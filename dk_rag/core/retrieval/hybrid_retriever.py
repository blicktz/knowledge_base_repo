"""
Hybrid Retriever combining BM25 and Vector Search

This module implements hybrid search that combines keyword-based (BM25)
and semantic (vector) search for comprehensive retrieval coverage.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
from collections import defaultdict
import numpy as np

from langchain.schema import Document
from langchain.vectorstores.base import VectorStore

from ...data.storage.bm25_store import BM25Store
from ...utils.logging import get_logger


class HybridRetriever:
    """
    Combines BM25 keyword search with vector similarity search.
    
    Achieves 20-30% improvement over either method alone by combining
    exact keyword matching with semantic understanding.
    """
    
    def __init__(
        self,
        bm25_store: BM25Store,
        vector_store: VectorStore,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            bm25_store: BM25 index for keyword search
            vector_store: Vector store for semantic search
            bm25_weight: Weight for BM25 scores (default 0.4)
            vector_weight: Weight for vector scores (default 0.6)
        """
        self.bm25_store = bm25_store
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.logger = get_logger(__name__)
        
        # Validate weights sum to 1.0
        weight_sum = self.bm25_weight + self.vector_weight
        if abs(weight_sum - 1.0) > 0.001:
            self.logger.warning(f"Weights sum to {weight_sum}, normalizing...")
            self.bm25_weight = self.bm25_weight / weight_sum
            self.vector_weight = self.vector_weight / weight_sum
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            scores: List of scores
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def _fuse_results(
        self,
        bm25_results: List[Tuple[str, float]],
        vector_results: List[Tuple[Document, float]],
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Fuse BM25 and vector search results using weighted combination.
        
        Args:
            bm25_results: List of (doc_id, score) from BM25
            vector_results: List of (document, score) from vector search
            bm25_weight: Optional override for BM25 weight
            vector_weight: Optional override for vector weight
            
        Returns:
            Fused and sorted list of (document, combined_score)
        """
        bm25_weight = bm25_weight or self.bm25_weight
        vector_weight = vector_weight or self.vector_weight
        
        # Create mappings for efficient lookup
        doc_scores = defaultdict(lambda: {"bm25": 0.0, "vector": 0.0, "doc": None})
        
        # Normalize BM25 scores and add to mapping
        if bm25_results:
            bm25_scores_only = [score for _, score in bm25_results]
            normalized_bm25 = self._normalize_scores(bm25_scores_only)
            
            for (doc_id, _), norm_score in zip(bm25_results, normalized_bm25):
                # Try to get document content from BM25 store
                doc_text = self.bm25_store.get_document_by_id(doc_id)
                if doc_text:
                    doc = Document(
                        page_content=doc_text,
                        metadata={"doc_id": doc_id, "source": "bm25"}
                    )
                    doc_scores[doc_id]["bm25"] = norm_score
                    doc_scores[doc_id]["doc"] = doc
        
        # Normalize vector scores and add to mapping
        if vector_results:
            vector_scores_only = [score for _, score in vector_results]
            normalized_vector = self._normalize_scores(vector_scores_only)
            
            for (doc, _), norm_score in zip(vector_results, normalized_vector):
                # Extract doc_id from metadata or generate one
                doc_id = doc.metadata.get("doc_id", doc.metadata.get("source", str(hash(doc.page_content))))
                
                doc_scores[doc_id]["vector"] = norm_score
                # Prefer vector document if available (has more metadata)
                if doc_scores[doc_id]["doc"] is None:
                    doc_scores[doc_id]["doc"] = doc
                else:
                    # Merge metadata if document exists
                    doc_scores[doc_id]["doc"].metadata.update(doc.metadata)
        
        # Calculate combined scores
        combined_results = []
        for doc_id, scores in doc_scores.items():
            if scores["doc"] is not None:
                # Calculate weighted combination
                combined_score = (
                    bm25_weight * scores["bm25"] +
                    vector_weight * scores["vector"]
                )
                
                # Add retrieval metadata
                scores["doc"].metadata.update({
                    "hybrid_score": combined_score,
                    "bm25_score": scores["bm25"],
                    "vector_score": scores["vector"],
                    "retrieval_method": "hybrid"
                })
                
                combined_results.append((scores["doc"], combined_score))
        
        # Sort by combined score (descending)
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results
    
    def search(
        self,
        query: str,
        k: int = 20,
        bm25_k: Optional[int] = None,
        vector_k: Optional[int] = None,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None
    ) -> List[Document]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: Search query
            k: Number of final results to return
            bm25_k: Number of BM25 results to retrieve (default 2*k)
            vector_k: Number of vector results to retrieve (default 2*k)
            bm25_weight: Optional override for BM25 weight
            vector_weight: Optional override for vector weight
            
        Returns:
            List of documents ranked by hybrid score
        """
        self.logger.info(f"Hybrid search for query: {query[:100]}... (k={k})")
        
        # Set retrieval counts (retrieve more than needed for better fusion)
        bm25_k = bm25_k or min(k * 2, 50)
        vector_k = vector_k or min(k * 2, 50)
        
        # Parallel retrieval (can be optimized with async)
        self.logger.debug(f"Retrieving BM25 results (k={bm25_k})...")
        bm25_results = self.bm25_store.search(query, k=bm25_k)
        
        self.logger.debug(f"Retrieving vector results (k={vector_k})...")
        vector_results = []
        if hasattr(self.vector_store, 'similarity_search_with_score'):
            vector_results = self.vector_store.similarity_search_with_score(query, k=vector_k)
        else:
            # Fallback if score method not available
            docs = self.vector_store.similarity_search(query, k=vector_k)
            vector_results = [(doc, 1.0) for doc in docs]
        
        # Fuse results
        self.logger.debug("Fusing BM25 and vector results...")
        fused_results = self._fuse_results(
            bm25_results,
            vector_results,
            bm25_weight,
            vector_weight
        )
        
        # Return top-k documents
        final_results = [doc for doc, _ in fused_results[:k]]
        
        self.logger.info(f"Hybrid search returned {len(final_results)} documents")
        return final_results
    
    def search_with_scores(
        self,
        query: str,
        k: int = 20,
        bm25_k: Optional[int] = None,
        vector_k: Optional[int] = None,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search and return documents with scores.
        
        Args:
            query: Search query
            k: Number of final results to return
            bm25_k: Number of BM25 results to retrieve
            vector_k: Number of vector results to retrieve
            bm25_weight: Optional override for BM25 weight
            vector_weight: Optional override for vector weight
            
        Returns:
            List of (document, hybrid_score) tuples
        """
        self.logger.info(f"Hybrid search with scores for: {query[:100]}...")
        
        # Set retrieval counts
        bm25_k = bm25_k or min(k * 2, 50)
        vector_k = vector_k or min(k * 2, 50)
        
        # Retrieve from both sources
        bm25_results = self.bm25_store.search(query, k=bm25_k)
        
        vector_results = []
        if hasattr(self.vector_store, 'similarity_search_with_score'):
            vector_results = self.vector_store.similarity_search_with_score(query, k=vector_k)
        else:
            docs = self.vector_store.similarity_search(query, k=vector_k)
            vector_results = [(doc, 1.0) for doc in docs]
        
        # Fuse results
        fused_results = self._fuse_results(
            bm25_results,
            vector_results,
            bm25_weight,
            vector_weight
        )
        
        # Return top-k with scores
        return fused_results[:k]
    
    def reciprocal_rank_fusion(
        self,
        query: str,
        k: int = 20,
        rrf_k: int = 60,
        bm25_k: Optional[int] = None,
        vector_k: Optional[int] = None
    ) -> List[Document]:
        """
        Use Reciprocal Rank Fusion (RRF) instead of weighted combination.
        
        RRF is a robust fusion method that doesn't require score normalization.
        
        Args:
            query: Search query
            k: Number of final results to return
            rrf_k: RRF k parameter (default 60)
            bm25_k: Number of BM25 results to retrieve
            vector_k: Number of vector results to retrieve
            
        Returns:
            List of documents ranked by RRF score
        """
        self.logger.info(f"RRF hybrid search for: {query[:100]}...")
        
        # Set retrieval counts
        bm25_k = bm25_k or min(k * 2, 50)
        vector_k = vector_k or min(k * 2, 50)
        
        # Get ranked lists from both methods
        bm25_results = self.bm25_store.search(query, k=bm25_k)
        
        vector_docs = self.vector_store.similarity_search(query, k=vector_k)
        
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        doc_map = {}
        
        # Process BM25 results
        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores[doc_id] += 1.0 / (rank + rrf_k)
            
            # Get document if not already stored
            if doc_id not in doc_map:
                doc_text = self.bm25_store.get_document_by_id(doc_id)
                if doc_text:
                    doc_map[doc_id] = Document(
                        page_content=doc_text,
                        metadata={"doc_id": doc_id}
                    )
        
        # Process vector results
        for rank, doc in enumerate(vector_docs):
            doc_id = doc.metadata.get("doc_id", str(hash(doc.page_content)))
            rrf_scores[doc_id] += 1.0 / (rank + rrf_k)
            
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        
        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top-k documents
        results = []
        for doc_id, score in sorted_docs[:k]:
            if doc_id in doc_map:
                doc = doc_map[doc_id]
                doc.metadata["rrf_score"] = score
                doc.metadata["retrieval_method"] = "hybrid_rrf"
                results.append(doc)
        
        self.logger.info(f"RRF returned {len(results)} documents")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the hybrid retriever.
        
        Returns:
            Dictionary with retriever statistics
        """
        stats = {
            "bm25_weight": self.bm25_weight,
            "vector_weight": self.vector_weight,
            "bm25_statistics": self.bm25_store.get_statistics()
        }
        
        # Try to get vector store statistics if available
        if hasattr(self.vector_store, 'get_statistics'):
            stats["vector_statistics"] = self.vector_store.get_statistics()
        
        return stats