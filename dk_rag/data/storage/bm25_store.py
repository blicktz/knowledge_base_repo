"""
BM25 Index Storage and Management

This module provides BM25 index creation, persistence, and search functionality
using the bm25s library for fast keyword-based retrieval.
"""

import json
import pickle
import gzip
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np

import bm25s
from bm25s.tokenization import Tokenizer

# Optional Chinese tokenization
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

from ...utils.logging import get_logger


class BM25Store:
    """
    Manages BM25 index creation, persistence, and search.
    
    Uses bm25s for 500x faster performance compared to rank-bm25,
    with numba backend for additional 2x speedup.
    """
    
    def __init__(
        self,
        index_path: str,
        tokenizer_type: str = "default",
        k1: float = 1.5,
        b: float = 0.75,
        language: str = "en"
    ):
        """
        Initialize BM25 store.

        Args:
            index_path: Path to store/load the index
            tokenizer_type: Type of tokenizer to use
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
            language: Language code ('en' for English, 'zh' for Chinese)
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(__name__)
        self.language = language.strip() if language else "en"

        # Setup language-aware tokenizer
        if self.language == "zh":
            if not JIEBA_AVAILABLE:
                self.logger.warning("jieba not available for Chinese tokenization, falling back to English")
                self.tokenizer = Tokenizer(stemmer=None, stopwords="en")
            else:
                # Create custom tokenizer for Chinese using jieba
                self.logger.info("Using jieba tokenizer for Chinese language")
                self.tokenizer = Tokenizer(
                    stemmer=None,
                    stopwords="zh",  # Chinese stopwords
                    splitter=self._jieba_tokenize
                )
        else:
            # Default English tokenizer
            self.tokenizer = Tokenizer(stemmer=None, stopwords="en")

        # BM25 parameters
        self.k1 = k1
        self.b = b
        
        # Index and document mapping
        self.bm25_index = None
        self.doc_ids = []
        self.doc_texts = []
        self.tokenized_docs = None
        
        # Paths for persistence
        self.index_file = self.index_path / "bm25_index.pkl.gz"
        self.metadata_file = self.index_path / "bm25_metadata.json"
        
        # Try to load existing index
        self._load_index()

    def _jieba_tokenize(self, text: str) -> List[str]:
        """
        Tokenize Chinese text using jieba.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        import jieba
        return list(jieba.cut(text))

    def index_exists(self) -> bool:
        """Check if BM25 index exists on disk."""
        return self.index_file.exists() and self.metadata_file.exists()
    
    def build_index(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        rebuild: bool = False
    ) -> None:
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of document texts
            doc_ids: Optional document IDs
            rebuild: Whether to rebuild from scratch
        """
        if self.bm25_index is not None and not rebuild:
            self.logger.info("BM25 index already exists. Use rebuild=True to force rebuild.")
            return
        
        self.logger.info(f"Building BM25 index for {len(documents)} documents...")
        
        # Defensive: Filter out None/empty documents
        valid_documents = []
        valid_doc_ids = []
        skipped = 0
        
        for i, doc in enumerate(documents):
            if doc is None or doc == '':
                skipped += 1
                self.logger.warning(f"Skipping null/empty document at index {i}")
                continue
            valid_documents.append(doc)
            if doc_ids and i < len(doc_ids):
                valid_doc_ids.append(doc_ids[i])
            else:
                valid_doc_ids.append(str(len(valid_documents) - 1))
        
        if skipped > 0:
            self.logger.warning(f"Filtered out {skipped} null/empty documents before indexing")
        
        # Store documents and IDs
        self.doc_texts = valid_documents
        self.doc_ids = valid_doc_ids
        
        # Tokenize documents
        self.logger.debug("Tokenizing documents...")
        self.tokenized_docs = self.tokenizer.tokenize(valid_documents, return_as="ids")
        
        # Build BM25 index
        self.logger.debug("Creating BM25 index...")
        self.bm25_index = bm25s.BM25(k1=self.k1, b=self.b)
        self.bm25_index.index(self.tokenized_docs)
        
        # Save index
        self._save_index()
        
        self.logger.info(f"BM25 index built and saved to {self.index_path}")
    
    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None
    ) -> None:
        """
        Add new documents to existing index.
        
        Note: This requires rebuilding the entire index as BM25
        statistics need to be recalculated.
        
        Args:
            documents: New documents to add
            doc_ids: Optional IDs for new documents
        """
        self.logger.info(f"Adding {len(documents)} documents to BM25 index...")
        
        # Prepare new document IDs
        if doc_ids is None:
            start_id = len(self.doc_texts)
            doc_ids = [str(start_id + i) for i in range(len(documents))]
        
        # Combine with existing documents
        all_documents = self.doc_texts + documents
        all_doc_ids = self.doc_ids + doc_ids
        
        # Rebuild index with all documents
        self.build_index(all_documents, all_doc_ids, rebuild=True)
    
    def search(
        self,
        query: str,
        k: int = 20,
        return_docs: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Search BM25 index.
        
        Args:
            query: Search query
            k: Number of results to return
            return_docs: Whether to return document texts
            
        Returns:
            List of (doc_id, score) or (doc_id, score, doc_text) tuples
        """
        if self.bm25_index is None:
            self.logger.warning("BM25 index not built. Returning empty results.")
            return []
        
        self.logger.debug(f"BM25 search for query: {query[:100]}...")
        
        # Tokenize query
        query_tokens = self.tokenizer.tokenize([query], return_as="ids")[0]
        
        # Get scores for all documents
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        if k >= len(scores):
            top_k_indices = np.argsort(scores)[::-1]
        else:
            # Use argpartition for efficiency with large collections
            top_k_indices = np.argpartition(scores, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])][::-1]
        
        # Build results
        results = []
        for idx in top_k_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                doc_id = self.doc_ids[idx]
                score = float(scores[idx])
                
                if return_docs:
                    doc_text = self.doc_texts[idx]
                    results.append((doc_id, score, doc_text))
                else:
                    results.append((doc_id, score))
        
        self.logger.debug(f"Found {len(results)} results with non-zero scores")
        return results
    
    def batch_search(
        self,
        queries: List[str],
        k: int = 20,
        return_docs: bool = False
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: List of search queries
            k: Number of results per query
            return_docs: Whether to return document texts
            
        Returns:
            List of result lists, one per query
        """
        if self.bm25_index is None:
            self.logger.warning("BM25 index not built. Returning empty results.")
            return [[] for _ in queries]
        
        self.logger.info(f"Batch BM25 search for {len(queries)} queries...")
        
        # Process each query individually (fallback for missing get_batch_scores)
        all_results = []
        for query in queries:
            # Use single query search method
            query_results = self.search(query, k=k, return_docs=return_docs)
            all_results.append(query_results)
        
        return all_results
    
    def get_document_by_id(self, doc_id: str) -> Optional[str]:
        """
        Get document text by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document text or None if not found
        """
        try:
            idx = self.doc_ids.index(doc_id)
            return self.doc_texts[idx]
        except ValueError:
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        if self.bm25_index is None:
            return {"status": "not_built"}
        
        return {
            "status": "built",
            "num_documents": len(self.doc_texts),
            "avg_doc_length": np.mean([len(doc.split()) for doc in self.doc_texts]),
            "index_size_bytes": self.index_file.stat().st_size if self.index_file.exists() else 0,
            "k1": self.k1,
            "b": self.b
        }
    
    def _save_index(self) -> None:
        """Save BM25 index and metadata to disk."""
        if self.bm25_index is None:
            self.logger.warning("No index to save")
            return
        
        self.logger.debug("Saving BM25 index...")
        
        # Save the index using pickle with compression
        index_data = {
            "bm25_index": self.bm25_index,
            "tokenized_docs": self.tokenized_docs,
            "doc_texts": self.doc_texts,
            "doc_ids": self.doc_ids
        }
        
        with gzip.open(self.index_file, 'wb') as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save metadata separately for easy inspection
        metadata = {
            "num_documents": len(self.doc_texts),
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids[:10]  # Save first 10 for reference
        }
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.debug(f"Index saved to {self.index_file}")
    
    def _load_index(self) -> bool:
        """
        Load BM25 index from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.index_file.exists():
            self.logger.debug("No existing BM25 index found")
            return False
        
        try:
            self.logger.info(f"Loading BM25 index from {self.index_file}...")
            
            with gzip.open(self.index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            self.bm25_index = index_data["bm25_index"]
            self.tokenized_docs = index_data["tokenized_docs"]
            self.doc_texts = index_data["doc_texts"]
            self.doc_ids = index_data["doc_ids"]
            
            self.logger.info(f"Loaded BM25 index with {len(self.doc_texts)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load BM25 index: {e}")
            return False
    
    def clear_index(self) -> None:
        """Clear the BM25 index and remove saved files."""
        self.logger.info("Clearing BM25 index...")
        
        # Clear in-memory index
        self.bm25_index = None
        self.doc_ids = []
        self.doc_texts = []
        self.tokenized_docs = None
        
        # Remove saved files
        if self.index_file.exists():
            self.index_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        
        self.logger.info("BM25 index cleared")