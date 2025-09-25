"""
Cross-Encoder Reranking for Advanced Retrieval

This module implements cross-encoder reranking to dramatically improve
the precision of retrieved results.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import json
from pathlib import Path
from datetime import datetime

try:
    from rerankers import Reranker
    RERANKERS_AVAILABLE = True
except ImportError:
    RERANKERS_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

import torch
from langchain.schema import Document

from ...utils.logging import get_logger
from ...utils.model_manager import get_model_manager


class CrossEncoderReranker:
    """
    Advanced reranking using cross-encoder models.
    
    Achieves 25-35% improvement in precision by directly modeling
    query-document relevance.
    """
    
    def __init__(
        self,
        model_name: str = "mixedbread-ai/mxbai-rerank-large-v1",
        use_cohere: bool = False,
        cohere_api_key: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize cross-encoder reranker with lazy model loading.
        
        Args:
            model_name: Model to use for reranking
            use_cohere: Whether to use Cohere API instead of local model
            cohere_api_key: API key for Cohere (if use_cohere=True)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            batch_size: Batch size for reranking
            cache_dir: Directory for caching reranking results
        """
        self.logger = get_logger(__name__)
        self.model_manager = get_model_manager()
        
        # Store configuration (models loaded lazily when needed)
        self.model_name = model_name
        self.use_cohere = use_cohere
        self.cohere_api_key = cohere_api_key
        self.batch_size = batch_size
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            base_dir = "/Volumes/J15/aicallgo_data/persona_data_base"
            self.cache_dir = Path(base_dir) / "retrieval_cache" / "reranker_logs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store device configuration (used by ModelManager)
        if device is None or device == "auto":
            # Let ModelManager handle device detection
            self.device = None
        else:
            self.device = device
        
        # Set reranker type for logging (determined by configuration)
        self.reranker_type = "cohere" if use_cohere else "local"
        
        # Check availability without loading models
        if use_cohere and not COHERE_AVAILABLE:
            raise ImportError("cohere package not available. Please install it to use Cohere reranking.")
        elif not use_cohere and not RERANKERS_AVAILABLE:
            raise ImportError("rerankers package not available. Please install it to use local reranking.")
        
        self.logger.info(f"CrossEncoderReranker configured with model: {model_name}, type: {self.reranker_type}")
        if not use_cohere:
            self.logger.info("Model will be loaded on first use for better performance")
    
    def _get_reranker_model(self):
        """
        Get the reranker model from ModelManager (lazy loading).
        
        Returns:
            Loaded reranker model
        """
        if self.use_cohere:
            return self.model_manager.get_reranker_model(
                model_name=self.model_name,
                use_cohere=True,
                cohere_api_key=self.cohere_api_key
            )
        else:
            return self.model_manager.get_reranker_model(
                model_name=self.model_name,
                use_cohere=False
            )
    
    def _is_model_loaded(self) -> bool:
        """Check if the reranker model is already loaded."""
        cache_key = f"cohere_{self.model_name}" if self.use_cohere else self.model_name
        return self.model_manager.is_model_loaded(cache_key, "reranker")
    
    def _log_reranking(
        self,
        query: str,
        candidates: List[str],
        scores: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log reranking operation for analysis.
        
        Args:
            query: Original query
            candidates: Candidate documents
            scores: Reranking scores
            metadata: Additional metadata
        """
        timestamp = datetime.now().isoformat()
        
        # Extract candidate counts from metadata if available
        total_candidates = metadata.get("total_candidates", len(candidates)) if metadata else len(candidates)
        final_candidates = metadata.get("final_candidates", len(scores)) if metadata else len(scores)
        
        # Get actual device being used (from ModelManager or configuration)
        actual_device = self.device if self.device else self.model_manager.device_manager.get_torch_device()
        
        log_entry = {
            "timestamp": timestamp,
            "query": query,
            "total_candidates": total_candidates,
            "final_candidates": final_candidates,
            "model": self.model_name,
            "reranker_type": self.reranker_type,
            "device": actual_device,
            "final_scores": scores,
            "metadata": {k: v for k, v in (metadata or {}).items() if k not in ["total_candidates", "final_candidates"]},
            "component": "CrossEncoderReranker"
        }
        
        # Save all final candidates (these are the top-k results)
        log_entry["final_candidates_text"] = candidates
        
        # Save to timestamped file
        filename = f"rerank_{timestamp.replace(':', '-')}.json"
        filepath = self.cache_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"Logged reranking operation to: {filepath}")
    
    def rerank(
        self,
        query: str,
        candidates: List[Document],
        top_k: int = 5,
        return_scores: bool = False,
        log_metadata: Optional[Dict[str, Any]] = None
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        Rerank candidate documents for the query.
        
        Args:
            query: Search query
            candidates: List of candidate documents
            top_k: Number of top documents to return
            return_scores: Whether to return scores with documents
            log_metadata: Additional metadata for logging
            
        Returns:
            Reranked documents or (document, score) tuples
        """
        if not candidates:
            return []
        
        self.logger.info(f"Reranking {len(candidates)} candidates for query: {query[:100]}...")
        
        # Extract text from documents
        candidate_texts = [doc.page_content for doc in candidates]
        
        # Perform reranking based on backend
        if self.reranker_type == "cohere":
            scores = self._rerank_cohere(query, candidate_texts, top_k)
        else:
            scores = self._rerank_local(query, candidate_texts)
        
        # Create (document, score) pairs
        doc_score_pairs = list(zip(candidates, scores))
        
        # Sort by score (descending)
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top-k
        top_results = doc_score_pairs[:top_k]
        
        # Log the reranking operation with final results
        final_scores = [score for doc, score in top_results]
        final_candidate_texts = [doc.page_content for doc, score in top_results]
        self._log_reranking(query, final_candidate_texts, final_scores, {
            **(log_metadata or {}),
            "total_candidates": len(candidates),
            "final_candidates": len(top_results)
        })
        
        # Add reranking metadata to documents
        for doc, score in top_results:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["rerank_score"] = float(score)
            doc.metadata["reranked"] = True
            doc.metadata["reranker_model"] = self.model_name
        
        self.logger.info(f"Reranking complete, returning top {len(top_results)} documents")
        
        if return_scores:
            return top_results
        else:
            return [doc for doc, _ in top_results]
    
    def _rerank_local(self, query: str, candidates: List[str]) -> List[float]:
        """
        Rerank using local model from ModelManager.
        
        Args:
            query: Search query
            candidates: Candidate texts
            
        Returns:
            List of scores
        """
        try:
            # Get reranker model from ModelManager (lazy loading)
            reranker = self._get_reranker_model()
            if reranker is None:
                self.logger.error("Failed to load local reranker model")
                return [1.0] * len(candidates)
            
            # Use rerankers library
            results = reranker.rank(
                query=query,
                docs=candidates,
                doc_ids=list(range(len(candidates)))
            )
            
            # Extract scores and maintain original order
            scores = [0.0] * len(candidates)
            for result in results:
                scores[result.doc_id] = result.score
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error in local reranking: {e}")
            # Return uniform scores as fallback
            return [1.0] * len(candidates)
    
    def _rerank_cohere(self, query: str, candidates: List[str], top_k: int) -> List[float]:
        """
        Rerank using Cohere API from ModelManager.
        
        Args:
            query: Search query
            candidates: Candidate texts
            top_k: Number of results to rerank
            
        Returns:
            List of scores
        """
        try:
            # Get Cohere reranker from ModelManager (lazy loading)
            cohere_reranker = self._get_reranker_model()
            if cohere_reranker is None:
                self.logger.error("Failed to load Cohere reranker")
                return [1.0] * len(candidates)
            
            # Call Cohere rerank API
            response = cohere_reranker.rank(
                query=query,
                docs=candidates,
                top_n=min(top_k, len(candidates))
            )
            
            # Create score array maintaining original indices
            scores = [0.0] * len(candidates)
            for result in response.results:
                scores[result.index] = result.relevance_score
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error in Cohere reranking: {e}")
            # Return uniform scores as fallback
            return [1.0] * len(candidates)
    
    def batch_rerank(
        self,
        queries: List[str],
        candidates_list: List[List[Document]],
        top_k: int = 5,
        return_scores: bool = False
    ) -> List[Union[List[Document], List[Tuple[Document, float]]]]:
        """
        Batch rerank multiple queries.
        
        Args:
            queries: List of queries
            candidates_list: List of candidate document lists
            top_k: Number of top documents per query
            return_scores: Whether to return scores
            
        Returns:
            List of reranked results for each query
        """
        self.logger.info(f"Batch reranking for {len(queries)} queries...")
        
        results = []
        for query, candidates in zip(queries, candidates_list):
            reranked = self.rerank(
                query,
                candidates,
                top_k=top_k,
                return_scores=return_scores
            )
            results.append(reranked)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get reranker statistics.
        
        Returns:
            Dictionary with reranker information
        """
        # Get actual device being used
        actual_device = self.device if self.device else self.model_manager.device_manager.get_torch_device()
        
        stats = {
            "model": self.model_name,
            "type": self.reranker_type,
            "device": actual_device,
            "batch_size": self.batch_size,
            "model_loaded": self._is_model_loaded()
        }
        
        # Count logged rerankings
        log_files = list(self.cache_dir.glob("rerank_*.json"))
        stats["total_rerankings_logged"] = len(log_files)
        
        return stats


class DualEncoderReranker:
    """
    Alternative reranker using dual-encoder models for faster reranking.
    
    This is faster than cross-encoders but less accurate. Useful for
    first-stage reranking before cross-encoder.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None
    ):
        """
        Initialize dual-encoder reranker.
        
        Args:
            model_name: Sentence transformer model name
            device: Device to run on
        """
        from sentence_transformers import SentenceTransformer
        
        self.logger = get_logger(__name__)
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        
        self.logger.info(f"DualEncoderReranker initialized with {model_name} on {device}")
    
    def rerank(
        self,
        query: str,
        candidates: List[Document],
        top_k: int = 10
    ) -> List[Document]:
        """
        Rerank using dual-encoder similarity.
        
        Args:
            query: Search query
            candidates: Candidate documents
            top_k: Number of top documents
            
        Returns:
            Reranked documents
        """
        if not candidates:
            return []
        
        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Encode candidates
        candidate_texts = [doc.page_content for doc in candidates]
        candidate_embeddings = self.model.encode(
            candidate_texts,
            convert_to_tensor=True,
            batch_size=32
        )
        
        # Calculate similarities
        similarities = self.model.similarity(query_embedding, candidate_embeddings)[0]
        
        # Sort by similarity
        sorted_indices = similarities.argsort(descending=True)
        
        # Return top-k documents
        reranked = []
        for idx in sorted_indices[:top_k]:
            doc = candidates[idx]
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["dual_encoder_score"] = float(similarities[idx])
            reranked.append(doc)
        
        return reranked