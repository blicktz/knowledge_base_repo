"""
Mental Models Retrieval Pipeline

Simplified pipeline for mental models: Vector Search → Reranking
Optimized for structured framework and methodology retrieval.
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

from langchain.schema import Document

from ..reranker import CrossEncoderReranker
from ..cache import MultiKnowledgeRetrievalCache
from ....models.knowledge_types import KnowledgeType
from ....models.knowledge_results import MentalModelResult
from ....data.storage.mental_models_store import MentalModelsStore
from ....utils.logging import get_logger, get_component_logger


class MentalModelsPipeline:
    """
    Simplified retrieval pipeline for mental models.
    
    Pipeline: Vector Search → Cross-Encoder Reranking
    
    Optimized for retrieving problem-solving frameworks, methodologies,
    and structured approaches with high precision.
    """
    
    def __init__(
        self,
        vector_store: MentalModelsStore,
        reranker: CrossEncoderReranker,
        persona_id: str,
        cache: Optional[MultiKnowledgeRetrievalCache] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize mental models pipeline.
        
        Args:
            vector_store: Mental models vector store
            reranker: Cross-encoder reranker
            persona_id: Persona identifier
            cache: Multi-knowledge cache instance
            cache_dir: Cache directory for pipeline logs
        """
        self.vector_store = vector_store
        self.reranker = reranker
        self.persona_id = persona_id
        self.cache = cache
        self.knowledge_type = KnowledgeType.MENTAL_MODELS
        
        self.logger = get_component_logger("MMPipe", persona_id)
        
        # Setup cache directory for pipeline logs
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            base_dir = "/Volumes/J15/aicallgo_data/persona_data_base"
            self.cache_dir = Path(base_dir) / "retrieval_cache" / "mental_models" / "pipeline_logs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Mental models pipeline initialized for persona: {persona_id}")
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        retrieval_k: int = 20,
        use_reranking: bool = True,
        return_scores: bool = False,
        min_confidence_score: float = 0.0,
        filter_by_categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Union[List[MentalModelResult], List[Tuple[MentalModelResult, float]]]:
        """
        Execute mental models retrieval pipeline.
        
        Args:
            query: User query
            k: Number of final results to return
            retrieval_k: Number of candidates before reranking
            use_reranking: Whether to use cross-encoder reranking
            return_scores: Whether to return relevance scores
            min_confidence_score: Minimum confidence score filter
            filter_by_categories: Optional category filters
            metadata: Additional metadata for logging
            
        Returns:
            List of MentalModelResult objects or (result, score) tuples
        """
        self.logger.info(f"Mental models retrieval for query: {query[:100]}... (k={k})")
        
        start_time = time.time()
        stage_timings = {}
        pipeline_metadata = metadata or {}
        
        try:
            # Stage 1: Vector Search
            candidates = self._vector_search(
                query=query,
                k=retrieval_k,
                min_confidence_score=min_confidence_score,
                filter_by_categories=filter_by_categories
            )
            stage_timings["vector_search"] = time.time() - start_time
            
            if not candidates:
                self.logger.warning(f"No candidates found for query: {query[:100]}...")
                self._log_pipeline_execution(query, stage_timings, {}, pipeline_metadata)
                return []
            
            self.logger.debug(f"Vector search found {len(candidates)} candidates")
            
            # Stage 2: Cross-Encoder Reranking (optional)
            if use_reranking and candidates:
                rerank_start = time.time()
                final_results = self._rerank_candidates(
                    query=query,
                    candidates=candidates,
                    top_k=k,
                    return_scores=return_scores
                )
                stage_timings["reranking"] = time.time() - rerank_start
                
                self.logger.debug(f"Reranking returned {len(final_results)} results")
            else:
                # No reranking, convert to MentalModelResult and take top k
                final_results = self._convert_to_results(candidates[:k], return_scores)
            
            # Log pipeline execution
            self._log_pipeline_execution(
                query=query,
                stage_timings=stage_timings,
                num_documents={
                    "after_vector_search": len(candidates),
                    "final_results": len(final_results)
                },
                metadata=pipeline_metadata
            )
            
            total_time = sum(stage_timings.values())
            self.logger.info(
                f"Mental models pipeline complete: {len(final_results)} results "
                f"(total time: {total_time:.3f}s)"
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Mental models pipeline error: {e}")
            
            # Log error
            self._log_pipeline_execution(
                query=query,
                stage_timings=stage_timings,
                num_documents={},
                metadata={**pipeline_metadata, "error": str(e)}
            )
            
            # Return empty results
            return []
    
    def _vector_search(
        self,
        query: str,
        k: int,
        min_confidence_score: float = 0.0,
        filter_by_categories: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Perform vector similarity search on mental models.
        
        Args:
            query: Search query
            k: Number of results
            min_confidence_score: Minimum confidence filter
            filter_by_categories: Category filters
            
        Returns:
            List of candidate documents
        """
        try:
            # Build metadata filter
            filter_metadata = {}
            if min_confidence_score > 0:
                filter_metadata["confidence_score"] = {"$gte": min_confidence_score}
            if filter_by_categories:
                # Filter by categories (exact match or contains)
                filter_metadata["categories"] = {"$in": filter_by_categories}
            
            # Use cache if available
            if self.cache:
                @self.cache.cache_vector_search(self.knowledge_type, self.persona_id)
                def cached_search(query_text: str, search_k: int):
                    return self.vector_store.search(
                        query=query_text,
                        k=search_k,
                        filter_metadata=filter_metadata if filter_metadata else None
                    )
                
                return cached_search(query, k)
            else:
                # Direct search without caching
                return self.vector_store.search(
                    query=query,
                    k=k,
                    filter_metadata=filter_metadata if filter_metadata else None
                )
                
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
    
    def _rerank_candidates(
        self,
        query: str,
        candidates: List[Document],
        top_k: int,
        return_scores: bool = False
    ) -> Union[List[MentalModelResult], List[Tuple[MentalModelResult, float]]]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Original query
            candidates: Candidate documents
            top_k: Number of results to return
            return_scores: Whether to return scores
            
        Returns:
            Reranked MentalModelResult objects
        """
        try:
            # Use cache if available
            if self.cache:
                @self.cache.cache_reranking(self.knowledge_type, self.persona_id)
                def cached_rerank(query_text: str, candidate_docs, k: int):
                    return self.reranker.rerank(
                        query=query_text,
                        candidates=candidate_docs,
                        top_k=k,
                        return_scores=True,  # Always get scores for conversion
                        log_metadata={
                            "knowledge_type": self.knowledge_type.value,
                            "persona_id": self.persona_id
                        }
                    )
                
                reranked_with_scores = cached_rerank(query, candidates, top_k)
            else:
                # Direct reranking without caching
                reranked_with_scores = self.reranker.rerank(
                    query=query,
                    candidates=candidates,
                    top_k=top_k,
                    return_scores=True,
                    log_metadata={
                        "knowledge_type": self.knowledge_type.value,
                        "persona_id": self.persona_id
                    }
                )
            
            # Convert to MentalModelResult objects
            results = []
            for doc, score in reranked_with_scores:
                result = MentalModelResult.from_document(
                    doc=doc,
                    persona_id=self.persona_id,
                    retrieval_method="vector_rerank",
                    score=score
                )
                
                if return_scores:
                    results.append((result, score))
                else:
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            # Fallback to converting top candidates without reranking
            return self._convert_to_results(candidates[:top_k], return_scores)
    
    def _convert_to_results(
        self,
        documents: List[Document],
        return_scores: bool = False
    ) -> Union[List[MentalModelResult], List[Tuple[MentalModelResult, float]]]:
        """
        Convert documents to MentalModelResult objects.
        
        Args:
            documents: Documents to convert
            return_scores: Whether to return scores (will be None if no scores available)
            
        Returns:
            List of MentalModelResult objects or (result, None) tuples
        """
        results = []
        
        for doc in documents:
            result = MentalModelResult.from_document(
                doc=doc,
                persona_id=self.persona_id,
                retrieval_method="vector_only"
            )
            
            if return_scores:
                results.append((result, None))  # No score available
            else:
                results.append(result)
        
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 5,
        **kwargs
    ) -> List[List[MentalModelResult]]:
        """
        Batch retrieval for multiple queries.
        
        Args:
            queries: List of queries
            k: Number of results per query
            **kwargs: Additional arguments for retrieve()
            
        Returns:
            List of result lists, one per query
        """
        self.logger.info(f"Batch mental models retrieval for {len(queries)} queries")
        
        results = []
        for i, query in enumerate(queries):
            self.logger.debug(f"Processing batch query {i+1}/{len(queries)}")
            query_results = self.retrieve(query, k, **kwargs)
            results.append(query_results)
        
        return results
    
    def get_frameworks_by_category(
        self,
        category: str,
        k: int = 10
    ) -> List[MentalModelResult]:
        """
        Get mental models by specific category.
        
        Args:
            category: Category to search for
            k: Number of results
            
        Returns:
            List of mental models in the category
        """
        self.logger.info(f"Retrieving {k} mental models for category: {category}")
        
        # Use category as the query with category filter
        return self.retrieve(
            query=f"frameworks and methods for {category}",
            k=k,
            filter_by_categories=[category.lower()],
            use_reranking=True
        )
    
    def _log_pipeline_execution(
        self,
        query: str,
        stage_timings: Dict[str, float],
        num_documents: Dict[str, int],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log complete pipeline execution for analysis.
        
        Args:
            query: Original query
            stage_timings: Timing for each stage
            num_documents: Document counts at each stage
            metadata: Additional metadata
        """
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "query": query,
            "knowledge_type": self.knowledge_type.value,
            "persona_id": self.persona_id,
            "pipeline": "simplified_mental_models",
            "stages": ["vector_search", "reranking"],
            "timings": stage_timings,
            "document_counts": num_documents,
            "total_time": sum(stage_timings.values()),
            "metadata": metadata or {},
            "component": "MentalModelsPipeline"
        }
        
        # Save to timestamped file
        filename = f"mental_models_{timestamp.replace(':', '-')}.json"
        filepath = self.cache_dir / filename
        
        try:
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Logged mental models pipeline execution to: {filepath}")
        except Exception as e:
            self.logger.warning(f"Failed to log pipeline execution: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            "pipeline_type": "simplified_mental_models",
            "knowledge_type": self.knowledge_type.value,
            "persona_id": self.persona_id,
            "components": ["vector_search", "reranking"],
            "cache_enabled": self.cache is not None
        }
        
        # Count pipeline executions
        if self.cache_dir.exists():
            log_files = list(self.cache_dir.glob("mental_models_*.json"))
            stats["total_executions"] = len(log_files)
            
            # Calculate average timings from recent executions
            if log_files:
                recent_files = sorted(log_files, key=lambda f: f.stat().st_mtime)[-10:]
                total_times = []
                vector_times = []
                rerank_times = []
                
                try:
                    import json
                    for log_file in recent_files:
                        with open(log_file, 'r') as f:
                            data = json.load(f)
                            total_times.append(data.get('total_time', 0))
                            
                            timings = data.get('timings', {})
                            if 'vector_search' in timings:
                                vector_times.append(timings['vector_search'])
                            if 'reranking' in timings:
                                rerank_times.append(timings['reranking'])
                    
                    if total_times:
                        stats["avg_total_time"] = sum(total_times) / len(total_times)
                    if vector_times:
                        stats["avg_vector_search_time"] = sum(vector_times) / len(vector_times)
                    if rerank_times:
                        stats["avg_reranking_time"] = sum(rerank_times) / len(rerank_times)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to calculate timing statistics: {e}")
        else:
            stats["total_executions"] = 0
        
        return stats