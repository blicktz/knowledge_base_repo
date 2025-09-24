"""
Core Beliefs Retrieval Pipeline

Simplified pipeline for core beliefs: Vector Search → Reranking
Optimized for retrieving foundational principles and belief statements.
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

from langchain.schema import Document

from ..reranker import CrossEncoderReranker
from ..cache import MultiKnowledgeRetrievalCache
from ....models.knowledge_types import KnowledgeType
from ....models.knowledge_results import CoreBeliefResult
from ....data.storage.core_beliefs_store import CoreBeliefsStore
from ....utils.logging import get_logger


class CoreBeliefsPipeline:
    """
    Simplified retrieval pipeline for core beliefs.
    
    Pipeline: Vector Search → Cross-Encoder Reranking
    
    Optimized for retrieving foundational principles, values, and belief
    statements with supporting evidence and high confidence scoring.
    """
    
    def __init__(
        self,
        vector_store: CoreBeliefsStore,
        reranker: CrossEncoderReranker,
        persona_id: str,
        cache: Optional[MultiKnowledgeRetrievalCache] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize core beliefs pipeline.
        
        Args:
            vector_store: Core beliefs vector store
            reranker: Cross-encoder reranker
            persona_id: Persona identifier
            cache: Multi-knowledge cache instance
            cache_dir: Cache directory for pipeline logs
        """
        self.vector_store = vector_store
        self.reranker = reranker
        self.persona_id = persona_id
        self.cache = cache
        self.knowledge_type = KnowledgeType.CORE_BELIEFS
        
        self.logger = get_logger(__name__)
        
        # Setup cache directory for pipeline logs
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            base_dir = "/Volumes/J15/aicallgo_data/persona_data_base"
            self.cache_dir = Path(base_dir) / "retrieval_cache" / "core_beliefs" / "pipeline_logs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Core beliefs pipeline initialized for persona: {persona_id}")
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        retrieval_k: int = 20,
        use_reranking: bool = True,
        return_scores: bool = False,
        min_confidence_score: float = 0.0,
        filter_by_category: Optional[str] = None,
        conviction_level: Optional[str] = None,
        include_evidence: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Union[List[CoreBeliefResult], List[Tuple[CoreBeliefResult, float]]]:
        """
        Execute core beliefs retrieval pipeline.
        
        Args:
            query: User query
            k: Number of final results to return
            retrieval_k: Number of candidates before reranking
            use_reranking: Whether to use cross-encoder reranking
            return_scores: Whether to return relevance scores
            min_confidence_score: Minimum confidence score filter
            filter_by_category: Optional category filter
            conviction_level: Filter by conviction level (very_high, high, moderate, etc.)
            include_evidence: Whether to include supporting evidence
            metadata: Additional metadata for logging
            
        Returns:
            List of CoreBeliefResult objects or (result, score) tuples
        """
        self.logger.info(f"Core beliefs retrieval for query: {query[:100]}... (k={k})")
        
        start_time = time.time()
        stage_timings = {}
        pipeline_metadata = metadata or {}
        
        try:
            # Stage 1: Vector Search
            candidates = self._vector_search(
                query=query,
                k=retrieval_k,
                min_confidence_score=min_confidence_score,
                filter_by_category=filter_by_category,
                conviction_level=conviction_level
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
                    return_scores=return_scores,
                    weight_by_confidence=True
                )
                stage_timings["reranking"] = time.time() - rerank_start
                
                self.logger.debug(f"Reranking returned {len(final_results)} results")
            else:
                # No reranking, convert to CoreBeliefResult and take top k
                final_results = self._convert_to_results(candidates[:k], return_scores)
            
            # Filter evidence if requested
            if not include_evidence:
                final_results = self._filter_evidence(final_results)
            
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
                f"Core beliefs pipeline complete: {len(final_results)} results "
                f"(total time: {total_time:.3f}s)"
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Core beliefs pipeline error: {e}")
            
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
        filter_by_category: Optional[str] = None,
        conviction_level: Optional[str] = None
    ) -> List[Document]:
        """
        Perform vector similarity search on core beliefs.
        
        Args:
            query: Search query
            k: Number of results
            min_confidence_score: Minimum confidence filter
            filter_by_category: Category filter
            conviction_level: Conviction level filter
            
        Returns:
            List of candidate documents
        """
        try:
            # Build metadata filter
            filter_metadata = {}
            
            if min_confidence_score > 0:
                filter_metadata["confidence_score"] = {"$gte": min_confidence_score}
            
            if filter_by_category:
                filter_metadata["semantic_category"] = filter_by_category.lower()
            
            if conviction_level:
                filter_metadata["conviction_level"] = conviction_level
            
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
        return_scores: bool = False,
        weight_by_confidence: bool = True
    ) -> Union[List[CoreBeliefResult], List[Tuple[CoreBeliefResult, float]]]:
        """
        Rerank candidates using cross-encoder with confidence weighting.
        
        Args:
            query: Original query
            candidates: Candidate documents
            top_k: Number of results to return
            return_scores: Whether to return scores
            weight_by_confidence: Whether to weight by confidence scores
            
        Returns:
            Reranked CoreBeliefResult objects
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
                        return_scores=True,  # Always get scores for weighting
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
            
            # Apply confidence weighting if requested
            if weight_by_confidence:
                reranked_with_scores = self._apply_confidence_weighting(reranked_with_scores)
            
            # Convert to CoreBeliefResult objects
            results = []
            for doc, score in reranked_with_scores:
                result = CoreBeliefResult.from_document(
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
    
    def _apply_confidence_weighting(
        self, 
        scored_documents: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        Apply confidence score weighting to reranking results.
        
        Args:
            scored_documents: Documents with reranking scores
            
        Returns:
            Documents with confidence-weighted scores
        """
        weighted_results = []
        
        for doc, rerank_score in scored_documents:
            try:
                confidence_score = doc.metadata.get('confidence_score', 0.5)
                
                # Combine reranking score with confidence (60% rerank, 40% confidence)
                weighted_score = (rerank_score * 0.6) + (confidence_score * 0.4)
                
                weighted_results.append((doc, weighted_score))
                
            except Exception as e:
                self.logger.warning(f"Failed to apply confidence weighting: {e}")
                weighted_results.append((doc, rerank_score))
        
        # Re-sort by weighted score
        weighted_results.sort(key=lambda x: x[1], reverse=True)
        
        return weighted_results
    
    def _convert_to_results(
        self,
        documents: List[Document],
        return_scores: bool = False
    ) -> Union[List[CoreBeliefResult], List[Tuple[CoreBeliefResult, float]]]:
        """
        Convert documents to CoreBeliefResult objects.
        
        Args:
            documents: Documents to convert
            return_scores: Whether to return scores (will be None if no scores available)
            
        Returns:
            List of CoreBeliefResult objects or (result, None) tuples
        """
        results = []
        
        for doc in documents:
            result = CoreBeliefResult.from_document(
                doc=doc,
                persona_id=self.persona_id,
                retrieval_method="vector_only"
            )
            
            if return_scores:
                results.append((result, None))  # No score available
            else:
                results.append(result)
        
        return results
    
    def _filter_evidence(
        self,
        results: Union[List[CoreBeliefResult], List[Tuple[CoreBeliefResult, float]]]
    ) -> Union[List[CoreBeliefResult], List[Tuple[CoreBeliefResult, float]]]:
        """
        Remove supporting evidence from results if not requested.
        
        Args:
            results: Results to filter
            
        Returns:
            Results with evidence removed
        """
        filtered_results = []
        
        for item in results:
            if isinstance(item, tuple):
                result, score = item
                result.supporting_evidence = []
                filtered_results.append((result, score))
            else:
                item.supporting_evidence = []
                filtered_results.append(item)
        
        return filtered_results
    
    def get_beliefs_by_category(
        self,
        category: str,
        k: int = 10,
        min_confidence: float = 0.7
    ) -> List[CoreBeliefResult]:
        """
        Get core beliefs by specific category.
        
        Args:
            category: Category to search for
            k: Number of results
            min_confidence: Minimum confidence score
            
        Returns:
            List of core beliefs in the category
        """
        self.logger.info(f"Retrieving {k} core beliefs for category: {category}")
        
        # Use category as the query with category filter
        return self.retrieve(
            query=f"beliefs and principles about {category}",
            k=k,
            filter_by_category=category.lower(),
            min_confidence_score=min_confidence,
            use_reranking=True
        )
    
    def get_high_conviction_beliefs(
        self,
        query: str,
        k: int = 10
    ) -> List[CoreBeliefResult]:
        """
        Get high-conviction beliefs for a query.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of high-conviction core beliefs
        """
        return self.retrieve(
            query=query,
            k=k,
            conviction_level="high",
            min_confidence_score=0.8,
            use_reranking=True
        )
    
    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 5,
        **kwargs
    ) -> List[List[CoreBeliefResult]]:
        """
        Batch retrieval for multiple queries.
        
        Args:
            queries: List of queries
            k: Number of results per query
            **kwargs: Additional arguments for retrieve()
            
        Returns:
            List of result lists, one per query
        """
        self.logger.info(f"Batch core beliefs retrieval for {len(queries)} queries")
        
        results = []
        for i, query in enumerate(queries):
            self.logger.debug(f"Processing batch query {i+1}/{len(queries)}")
            query_results = self.retrieve(query, k, **kwargs)
            results.append(query_results)
        
        return results
    
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
            "pipeline": "simplified_core_beliefs",
            "stages": ["vector_search", "reranking"],
            "timings": stage_timings,
            "document_counts": num_documents,
            "total_time": sum(stage_timings.values()),
            "metadata": metadata or {},
            "component": "CoreBeliefsPipeline"
        }
        
        # Save to timestamped file
        filename = f"core_beliefs_{timestamp.replace(':', '-')}.json"
        filepath = self.cache_dir / filename
        
        try:
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Logged core beliefs pipeline execution to: {filepath}")
        except Exception as e:
            self.logger.warning(f"Failed to log pipeline execution: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            "pipeline_type": "simplified_core_beliefs",
            "knowledge_type": self.knowledge_type.value,
            "persona_id": self.persona_id,
            "components": ["vector_search", "reranking"],
            "cache_enabled": self.cache is not None
        }
        
        # Count pipeline executions
        if self.cache_dir.exists():
            log_files = list(self.cache_dir.glob("core_beliefs_*.json"))
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