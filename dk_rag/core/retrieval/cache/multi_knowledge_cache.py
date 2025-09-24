"""
Multi-Knowledge Retrieval Cache

Provides separate caching for different knowledge types while sharing
the underlying cache infrastructure. Each knowledge type gets its own
cache namespace to prevent cross-contamination.
"""

import hashlib
import json
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union

from ....models.knowledge_types import KnowledgeType, CacheOperation, validate_knowledge_type
from ....data.storage.retrieval_cache import RetrievalCache
from ....utils.logging import get_logger


class MultiKnowledgeRetrievalCache:
    """
    Multi-knowledge retrieval cache with separate namespaces per knowledge type.
    
    Manages individual RetrievalCache instances for each knowledge type
    while providing a unified interface for cache operations.
    """
    
    def __init__(
        self,
        base_cache_dir: str,
        cache_size: int = 128,
        ttl_hours: int = 168  # 1 week default
    ):
        """
        Initialize multi-knowledge cache.
        
        Args:
            base_cache_dir: Base directory for all cache storage
            cache_size: LRU cache size per knowledge type
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.base_cache_dir = Path(base_cache_dir)
        self.cache_size = cache_size
        self.ttl_hours = ttl_hours
        self.logger = get_logger(__name__)
        
        # Individual cache instances per knowledge type
        self._caches: Dict[KnowledgeType, RetrievalCache] = {}
        
        # Initialize all knowledge type caches
        self._initialize_caches()
        
        self.logger.info(
            f"Multi-knowledge cache initialized at {base_cache_dir} "
            f"with {len(self._caches)} knowledge type caches"
        )
    
    def _initialize_caches(self):
        """Initialize individual cache instances for each knowledge type."""
        for knowledge_type in KnowledgeType:
            cache_dir = self.base_cache_dir / knowledge_type.cache_directory
            
            # Create cache instance
            cache = RetrievalCache(
                cache_dir=str(cache_dir),
                cache_size=self.cache_size,
                ttl_hours=self.ttl_hours
            )
            
            self._caches[knowledge_type] = cache
            
            self.logger.debug(f"Initialized cache for {knowledge_type.display_name} at {cache_dir}")
    
    def get_cache(self, knowledge_type: KnowledgeType) -> RetrievalCache:
        """
        Get the cache instance for a specific knowledge type.
        
        Args:
            knowledge_type: Type of knowledge to get cache for
            
        Returns:
            RetrievalCache instance for the knowledge type
        """
        knowledge_type = validate_knowledge_type(knowledge_type)
        
        if knowledge_type not in self._caches:
            raise ValueError(f"No cache initialized for knowledge type: {knowledge_type}")
        
        return self._caches[knowledge_type]
    
    def generate_cache_key(
        self,
        knowledge_type: KnowledgeType,
        persona_id: str,
        operation: CacheOperation,
        content: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate cache key with knowledge type namespace.
        
        Args:
            knowledge_type: Type of knowledge for namespacing
            persona_id: Persona identifier
            operation: Type of cache operation
            content: Primary content to hash
            additional_context: Additional context for key generation
            
        Returns:
            Unique cache key string
        """
        knowledge_type = validate_knowledge_type(knowledge_type)
        
        # Build content for hashing
        key_components = {
            'knowledge_type': knowledge_type.value,
            'persona_id': persona_id,
            'operation': operation.value,
            'content': content
        }
        
        if additional_context:
            key_components['context'] = additional_context
        
        # Create deterministic hash
        content_str = json.dumps(key_components, sort_keys=True)
        content_hash = hashlib.sha256(content_str.encode('utf-8')).hexdigest()[:16]
        
        # Format: {knowledge_type}:{persona_id}:{operation}:{hash}
        cache_key = f"{knowledge_type.value}:{persona_id}:{operation.value}:{content_hash}"
        
        return cache_key
    
    def cache_vector_search(
        self,
        knowledge_type: KnowledgeType,
        persona_id: str
    ) -> Callable:
        """
        Decorator to cache vector search results.
        
        Args:
            knowledge_type: Type of knowledge being searched
            persona_id: Persona identifier
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(query: str, k: int = 5, *args, **kwargs):
                # Generate cache key
                cache_key = self.generate_cache_key(
                    knowledge_type=knowledge_type,
                    persona_id=persona_id,
                    operation=CacheOperation.VECTOR_SEARCH,
                    content=query,
                    additional_context={'k': k}
                )
                
                # Try to get from cache
                cache = self.get_cache(knowledge_type)
                cached_result = cache.get_cached_result(cache_key)
                
                if cached_result is not None:
                    self.logger.debug(
                        f"Cache hit for {knowledge_type.display_name} vector search: {query[:50]}..."
                    )
                    return cached_result
                
                # Execute function and cache result
                result = func(query, k, *args, **kwargs)
                
                # Cache the result
                cache.cache_result(
                    cache_key,
                    result,
                    metadata={
                        'operation': 'vector_search',
                        'knowledge_type': knowledge_type.value,
                        'persona_id': persona_id,
                        'query': query[:100],  # Truncate for storage
                        'k': k
                    }
                )
                
                self.logger.debug(
                    f"Cached {knowledge_type.display_name} vector search result for: {query[:50]}..."
                )
                
                return result
            
            return wrapper
        return decorator
    
    def cache_reranking(
        self,
        knowledge_type: KnowledgeType,
        persona_id: str
    ) -> Callable:
        """
        Decorator to cache reranking results.
        
        Args:
            knowledge_type: Type of knowledge being reranked
            persona_id: Persona identifier
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(query: str, candidates, top_k: int = 5, *args, **kwargs):
                # Generate cache key based on query and candidate hashes
                candidates_hash = self._hash_candidates(candidates)
                
                cache_key = self.generate_cache_key(
                    knowledge_type=knowledge_type,
                    persona_id=persona_id,
                    operation=CacheOperation.RERANKING,
                    content=query,
                    additional_context={
                        'candidates_hash': candidates_hash,
                        'top_k': top_k
                    }
                )
                
                # Try to get from cache
                cache = self.get_cache(knowledge_type)
                cached_result = cache.get_cached_result(cache_key)
                
                if cached_result is not None:
                    self.logger.debug(
                        f"Cache hit for {knowledge_type.display_name} reranking: {query[:50]}..."
                    )
                    return cached_result
                
                # Execute function and cache result
                result = func(query, candidates, top_k, *args, **kwargs)
                
                # Cache the result
                cache.cache_result(
                    cache_key,
                    result,
                    metadata={
                        'operation': 'reranking',
                        'knowledge_type': knowledge_type.value,
                        'persona_id': persona_id,
                        'query': query[:100],
                        'candidates_count': len(candidates),
                        'top_k': top_k
                    }
                )
                
                self.logger.debug(
                    f"Cached {knowledge_type.display_name} reranking result for: {query[:50]}..."
                )
                
                return result
            
            return wrapper
        return decorator
    
    def cache_pipeline_execution(
        self,
        knowledge_type: KnowledgeType,
        persona_id: str,
        pipeline_config: Dict[str, Any]
    ):
        """
        Cache complete pipeline execution results.
        
        Args:
            knowledge_type: Type of knowledge being processed
            persona_id: Persona identifier
            pipeline_config: Pipeline configuration for cache key
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(query: str, *args, **kwargs):
                # Generate cache key
                cache_key = self.generate_cache_key(
                    knowledge_type=knowledge_type,
                    persona_id=persona_id,
                    operation=CacheOperation.PIPELINE_EXECUTION,
                    content=query,
                    additional_context=pipeline_config
                )
                
                # Try to get from cache
                cache = self.get_cache(knowledge_type)
                cached_result = cache.get_cached_result(cache_key)
                
                if cached_result is not None:
                    self.logger.debug(
                        f"Cache hit for {knowledge_type.display_name} pipeline: {query[:50]}..."
                    )
                    return cached_result
                
                # Execute function and cache result
                result = func(query, *args, **kwargs)
                
                # Cache the result with pipeline metadata
                cache.cache_result(
                    cache_key,
                    result,
                    metadata={
                        'operation': 'pipeline_execution',
                        'knowledge_type': knowledge_type.value,
                        'persona_id': persona_id,
                        'query': query[:100],
                        'pipeline_config': pipeline_config
                    }
                )
                
                self.logger.debug(
                    f"Cached {knowledge_type.display_name} pipeline result for: {query[:50]}..."
                )
                
                return result
            
            return wrapper
        return decorator
    
    def save_llm_interaction(
        self,
        knowledge_type: KnowledgeType,
        prompt: str,
        response: str,
        model: str,
        component: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save LLM interaction to knowledge-type specific cache.
        
        Args:
            knowledge_type: Type of knowledge for the interaction
            prompt: LLM prompt
            response: LLM response
            model: Model name
            component: Component that made the call
            metadata: Additional metadata
        """
        knowledge_type = validate_knowledge_type(knowledge_type)
        cache = self.get_cache(knowledge_type)
        
        # Add knowledge type to metadata
        interaction_metadata = metadata.copy() if metadata else {}
        interaction_metadata.update({
            'knowledge_type': knowledge_type.value,
            'component': component
        })
        
        cache.save_llm_interaction(
            prompt=prompt,
            response=response,
            model=model,
            component=component,
            metadata=interaction_metadata
        )
    
    def get_cache_statistics(self, knowledge_type: Optional[KnowledgeType] = None) -> Dict[str, Any]:
        """
        Get cache statistics for one or all knowledge types.
        
        Args:
            knowledge_type: Specific knowledge type or None for all
            
        Returns:
            Dictionary with cache statistics
        """
        if knowledge_type:
            knowledge_type = validate_knowledge_type(knowledge_type)
            cache = self.get_cache(knowledge_type)
            stats = cache.get_cache_statistics()
            stats['knowledge_type'] = knowledge_type.value
            return stats
        
        # Get stats for all knowledge types
        all_stats = {
            'summary': {
                'total_cache_entries': 0,
                'total_cache_size_mb': 0.0,
                'avg_hit_rate': 0.0,
                'knowledge_types_count': len(self._caches)
            },
            'by_knowledge_type': {}
        }
        
        hit_rates = []
        
        for kt, cache in self._caches.items():
            stats = cache.get_cache_statistics()
            stats['knowledge_type'] = kt.value
            
            all_stats['by_knowledge_type'][kt.value] = stats
            
            # Aggregate summary stats
            all_stats['summary']['total_cache_entries'] += stats.get('total_cached_items', 0)
            all_stats['summary']['total_cache_size_mb'] += stats.get('cache_size_mb', 0.0)
            
            hit_rate = stats.get('hit_rate', 0.0)
            if hit_rate > 0:
                hit_rates.append(hit_rate)
        
        # Calculate average hit rate
        if hit_rates:
            all_stats['summary']['avg_hit_rate'] = sum(hit_rates) / len(hit_rates)
        
        return all_stats
    
    def cleanup_expired(self, knowledge_type: Optional[KnowledgeType] = None):
        """
        Clean up expired cache entries.
        
        Args:
            knowledge_type: Specific knowledge type or None for all
        """
        if knowledge_type:
            knowledge_type = validate_knowledge_type(knowledge_type)
            cache = self.get_cache(knowledge_type)
            cache.cleanup_expired()
            self.logger.info(f"Cleaned up expired entries for {knowledge_type.display_name}")
        else:
            # Clean up all caches
            for kt, cache in self._caches.items():
                cache.cleanup_expired()
            self.logger.info("Cleaned up expired entries for all knowledge types")
    
    def clear_cache(self, knowledge_type: Optional[KnowledgeType] = None):
        """
        Clear cache for one or all knowledge types.
        
        Args:
            knowledge_type: Specific knowledge type or None for all
        """
        if knowledge_type:
            knowledge_type = validate_knowledge_type(knowledge_type)
            cache = self.get_cache(knowledge_type)
            cache.clear_cache()
            self.logger.info(f"Cleared cache for {knowledge_type.display_name}")
        else:
            # Clear all caches
            for kt, cache in self._caches.items():
                cache.clear_cache()
            self.logger.info("Cleared all knowledge type caches")
    
    def _hash_candidates(self, candidates) -> str:
        """
        Generate hash for candidate documents.
        
        Args:
            candidates: List of candidate documents
            
        Returns:
            Hash string representing the candidates
        """
        try:
            # Extract content from candidates for hashing
            if hasattr(candidates[0], 'page_content'):
                # LangChain Document objects
                content_list = [doc.page_content for doc in candidates]
            elif isinstance(candidates[0], str):
                # String candidates
                content_list = candidates
            else:
                # Fallback to string representation
                content_list = [str(candidate) for candidate in candidates]
            
            # Create deterministic hash
            content_str = '|||'.join(content_list)
            return hashlib.sha256(content_str.encode('utf-8')).hexdigest()[:16]
            
        except Exception as e:
            self.logger.warning(f"Failed to hash candidates: {e}")
            # Fallback hash based on length and type
            return hashlib.sha256(
                f"candidates_{len(candidates)}_{type(candidates[0])}".encode('utf-8')
            ).hexdigest()[:16]