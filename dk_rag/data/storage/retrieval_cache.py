"""
Retrieval Cache and LLM Logging

This module provides caching for expensive retrieval operations and
comprehensive logging of all LLM interactions.
"""

import json
import hashlib
import pickle
import gzip
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import time

from ...utils.logging import get_logger


class RetrievalCache:
    """
    LRU caching and persistent storage for expensive retrieval operations.
    
    Provides:
    - In-memory LRU caching for fast access
    - Persistent disk caching for long-term storage
    - Complete LLM interaction logging
    - Performance metrics tracking
    """
    
    def __init__(
        self,
        cache_dir: str,
        cache_size: int = 128,
        ttl_hours: int = 168,  # 1 week default
        enable_compression: bool = True
    ):
        """
        Initialize retrieval cache.
        
        Args:
            cache_dir: Directory for cache storage
            cache_size: Size of in-memory LRU cache
            ttl_hours: Time-to-live for cached items in hours
            enable_compression: Whether to compress cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_size = cache_size
        self.ttl_hours = ttl_hours
        self.enable_compression = enable_compression
        self.logger = get_logger(__name__)
        
        # Create cache subdirectories
        self.hyde_cache_dir = self.cache_dir / "hyde_cache"
        self.rerank_cache_dir = self.cache_dir / "rerank_cache"
        self.llm_logs_dir = self.cache_dir / "llm_logs"
        self.metrics_dir = self.cache_dir / "metrics"
        
        for dir_path in [self.hyde_cache_dir, self.rerank_cache_dir, self.llm_logs_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "loads": 0,
            "errors": 0
        }
        
        self.logger.info(f"RetrievalCache initialized at {self.cache_dir}")
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from function arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            MD5 hash of arguments
        """
        # Create a string representation of arguments
        key_data = {
            "args": str(args),
            "kwargs": str(sorted(kwargs.items()))
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_type: str, cache_key: str) -> Path:
        """
        Get cache file path.
        
        Args:
            cache_type: Type of cache ('hyde', 'rerank', etc.)
            cache_key: Cache key
            
        Returns:
            Path to cache file
        """
        if cache_type == "hyde":
            cache_dir = self.hyde_cache_dir
        elif cache_type == "rerank":
            cache_dir = self.rerank_cache_dir
        else:
            cache_dir = self.cache_dir / cache_type
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        extension = ".pkl.gz" if self.enable_compression else ".pkl"
        return cache_dir / f"{cache_key}{extension}"
    
    def _save_to_cache(self, cache_type: str, cache_key: str, data: Any):
        """
        Save data to persistent cache.
        
        Args:
            cache_type: Type of cache
            cache_key: Cache key
            data: Data to cache
        """
        try:
            cache_path = self._get_cache_path(cache_type, cache_key)
            
            cache_entry = {
                "timestamp": datetime.now().isoformat(),
                "data": data,
                "cache_key": cache_key
            }
            
            if self.enable_compression:
                with gzip.open(cache_path, 'wb') as f:
                    pickle.dump(cache_entry, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_entry, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.stats["saves"] += 1
            self.logger.debug(f"Saved to cache: {cache_type}/{cache_key}")
            
        except Exception as e:
            self.logger.error(f"Cache save error: {e}")
            self.stats["errors"] += 1
    
    def _load_from_cache(self, cache_type: str, cache_key: str) -> Optional[Any]:
        """
        Load data from persistent cache.
        
        Args:
            cache_type: Type of cache
            cache_key: Cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        try:
            cache_path = self._get_cache_path(cache_type, cache_key)
            
            if not cache_path.exists():
                return None
            
            if self.enable_compression:
                with gzip.open(cache_path, 'rb') as f:
                    cache_entry = pickle.load(f)
            else:
                with open(cache_path, 'rb') as f:
                    cache_entry = pickle.load(f)
            
            # Check TTL
            cached_time = datetime.fromisoformat(cache_entry["timestamp"])
            if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
                self.logger.debug(f"Cache expired: {cache_type}/{cache_key}")
                cache_path.unlink()  # Remove expired cache
                return None
            
            self.stats["loads"] += 1
            self.logger.debug(f"Loaded from cache: {cache_type}/{cache_key}")
            return cache_entry["data"]
            
        except Exception as e:
            self.logger.error(f"Cache load error: {e}")
            self.stats["errors"] += 1
            return None
    
    def cache_hyde_generation(self, func: Callable) -> Callable:
        """
        Decorator for caching HyDE hypothesis generation.
        
        Args:
            func: Function to cache
            
        Returns:
            Cached function
        """
        # Create LRU cache for in-memory caching
        cached_func = lru_cache(maxsize=self.cache_size)(func)
        
        @wraps(func)
        def wrapper(query: str, *args, **kwargs):
            # Generate cache key
            cache_key = self._get_cache_key(query, *args, **kwargs)
            
            # Try to load from persistent cache first
            cached_result = self._load_from_cache("hyde", cache_key)
            if cached_result is not None:
                self.stats["hits"] += 1
                return cached_result
            
            # Not in cache, generate new
            self.stats["misses"] += 1
            result = cached_func(query, *args, **kwargs)
            
            # Save to persistent cache
            self._save_to_cache("hyde", cache_key, result)
            
            return result
        
        # Expose cache clearing
        wrapper.cache_clear = cached_func.cache_clear
        wrapper.cache_info = cached_func.cache_info
        
        return wrapper
    
    def cache_reranking(self, func: Callable) -> Callable:
        """
        Decorator for caching reranking results.
        
        Args:
            func: Function to cache
            
        Returns:
            Cached function
        """
        cached_func = lru_cache(maxsize=self.cache_size)(func)
        
        @wraps(func)
        def wrapper(query: str, candidates: List, *args, **kwargs):
            # Create cache key from query and candidate hashes
            candidates_hash = hashlib.md5(
                str([c.page_content[:100] for c in candidates]).encode()
            ).hexdigest()
            cache_key = self._get_cache_key(query, candidates_hash, *args, **kwargs)
            
            # Try to load from cache
            cached_result = self._load_from_cache("rerank", cache_key)
            if cached_result is not None:
                self.stats["hits"] += 1
                return cached_result
            
            # Not in cache, rerank
            self.stats["misses"] += 1
            result = func(query, candidates, *args, **kwargs)
            
            # Save to cache
            self._save_to_cache("rerank", cache_key, result)
            
            return result
        
        return wrapper
    
    def save_llm_interaction(
        self,
        prompt: str,
        response: str,
        model: str,
        component: str,
        metadata: Optional[Dict[str, Any]] = None,
        timing: Optional[float] = None
    ):
        """
        Save complete LLM interaction for analysis.
        
        Args:
            prompt: Input prompt
            response: LLM response
            model: Model used
            component: Component that made the call
            metadata: Additional metadata
            timing: Execution time
        """
        timestamp = datetime.now().isoformat()
        
        interaction = {
            "timestamp": timestamp,
            "component": component,
            "model": model,
            "prompt": prompt,
            "response": response,
            "prompt_tokens": len(prompt.split()),  # Approximate
            "response_tokens": len(response.split()),  # Approximate
            "timing_seconds": timing,
            "metadata": metadata or {}
        }
        
        # Save to timestamped file
        filename = f"llm_{component}_{timestamp.replace(':', '-')}.json"
        filepath = self.llm_logs_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(interaction, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved LLM interaction to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save LLM interaction: {e}")
    
    def save_performance_metrics(
        self,
        operation: str,
        timing: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save performance metrics.
        
        Args:
            operation: Operation name
            timing: Execution time
            metadata: Additional metadata
        """
        timestamp = datetime.now().isoformat()
        
        metric = {
            "timestamp": timestamp,
            "operation": operation,
            "timing_seconds": timing,
            "metadata": metadata or {}
        }
        
        # Append to daily metrics file
        date_str = datetime.now().strftime("%Y-%m-%d")
        metrics_file = self.metrics_dir / f"metrics_{date_str}.jsonl"
        
        try:
            with open(metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metric) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = self.stats.copy()
        
        # Calculate hit rate
        total_requests = stats["hits"] + stats["misses"]
        if total_requests > 0:
            stats["hit_rate"] = stats["hits"] / total_requests
        else:
            stats["hit_rate"] = 0.0
        
        # Count cached files
        stats["hyde_cached_items"] = len(list(self.hyde_cache_dir.glob("*")))
        stats["rerank_cached_items"] = len(list(self.rerank_cache_dir.glob("*")))
        stats["llm_logs_count"] = len(list(self.llm_logs_dir.glob("*.json")))
        
        # Calculate cache size
        total_size = 0
        for cache_file in self.cache_dir.rglob("*"):
            if cache_file.is_file():
                total_size += cache_file.stat().st_size
        
        stats["cache_size_mb"] = total_size / (1024 * 1024)
        
        return stats
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache.
        
        Args:
            cache_type: Specific cache type to clear, or None for all
        """
        if cache_type == "hyde":
            dirs_to_clear = [self.hyde_cache_dir]
        elif cache_type == "rerank":
            dirs_to_clear = [self.rerank_cache_dir]
        elif cache_type is None:
            dirs_to_clear = [self.hyde_cache_dir, self.rerank_cache_dir]
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
        
        for cache_dir in dirs_to_clear:
            for cache_file in cache_dir.glob("*"):
                if cache_file.is_file():
                    cache_file.unlink()
        
        self.logger.info(f"Cleared cache: {cache_type or 'all'}")
    
    def cleanup_expired(self):
        """Remove expired cache entries."""
        expired_count = 0
        
        for cache_dir in [self.hyde_cache_dir, self.rerank_cache_dir]:
            for cache_file in cache_dir.glob("*"):
                try:
                    # Check file age
                    file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if file_age > timedelta(hours=self.ttl_hours):
                        cache_file.unlink()
                        expired_count += 1
                except Exception as e:
                    self.logger.error(f"Error cleaning up {cache_file}: {e}")
        
        if expired_count > 0:
            self.logger.info(f"Cleaned up {expired_count} expired cache entries")
    
    def get_cached_result(self, cache_key: str, cache_type: str = "general") -> Optional[Any]:
        """
        Public method to get cached result.
        
        Args:
            cache_key: Cache key to look up
            cache_type: Type of cache (default: "general")
            
        Returns:
            Cached result or None if not found/expired
        """
        return self._load_from_cache(cache_type, cache_key)
    
    def cache_result(self, cache_key: str, result: Any, cache_type: str = "general", metadata: Optional[Dict[str, Any]] = None):
        """
        Public method to cache result.
        
        Args:
            cache_key: Cache key
            result: Result to cache
            cache_type: Type of cache (default: "general")
            metadata: Additional metadata (currently unused but kept for API compatibility)
        """
        self._save_to_cache(cache_type, cache_key, result)


def timed_cache(cache: RetrievalCache, operation: str):
    """
    Decorator that times operations and saves metrics.
    
    Args:
        cache: RetrievalCache instance
        operation: Operation name
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            cache.save_performance_metrics(
                operation,
                elapsed,
                {"args_sample": str(args)[:100]}
            )
            
            return result
        return wrapper
    return decorator