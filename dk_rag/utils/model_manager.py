"""
Persistent Model Manager for Server-Ready RAG Operations

This module provides a singleton ModelManager that loads and caches ML models 
once per process lifetime, enabling efficient resource usage for both server
deployments and long-running CLI operations.
"""

import threading
from typing import Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from .logging import get_logger, get_component_logger
from .device_manager import get_device_manager


@dataclass
class ModelInfo:
    """Information about a loaded model"""
    model_name: str
    model_type: str  # 'embedding', 'reranker', 'chroma_embedding'
    device: str
    loaded_at: str
    memory_usage_mb: Optional[float] = None
    is_loaded: bool = True


class ModelManager:
    """
    Singleton model manager for persistent ML model loading.
    
    Provides thread-safe access to cached models across the application,
    avoiding repeated model loading and enabling efficient resource usage.
    """
    
    def __init__(self):
        self.logger = get_component_logger("ModelMgr")
        self.device_manager = get_device_manager()
        
        # Thread-safe model storage
        self._lock = threading.RLock()
        self._embedding_models: Dict[str, SentenceTransformer] = {}
        self._reranker_models: Dict[str, Union[Reranker, 'CohereReranker']] = {}
        self._chroma_embedding_functions: Dict[str, Any] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        
        # Configuration
        self.default_embedding_model = "sentence-transformers/all-mpnet-base-v2"
        self.default_reranker_model = "mixedbread-ai/mxbai-rerank-large-v1"
        
        self.logger.info("🤖 ModelManager initialized")
    
    def get_embedding_model(
        self, 
        model_name: Optional[str] = None
    ) -> Optional[SentenceTransformer]:
        """
        Get or load a sentence transformer embedding model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded SentenceTransformer model or None if unavailable
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning("sentence-transformers not available")
            return None
        
        model_name = model_name or self.default_embedding_model
        
        with self._lock:
            # Return cached model if available
            if model_name in self._embedding_models:
                self.logger.debug(f"Using cached embedding model: {model_name}")
                return self._embedding_models[model_name]
            
            # Load new model
            self.logger.info(f"🚀 Loading embedding model: {model_name}")
            try:
                device = self.device_manager.get_sentence_transformers_device()
                model = SentenceTransformer(model_name, device=device)
                
                # Cache the model
                self._embedding_models[model_name] = model
                
                # Store model info
                device_str = device or "auto"
                self._model_info[f"embedding_{model_name}"] = ModelInfo(
                    model_name=model_name,
                    model_type="embedding",
                    device=device_str,
                    loaded_at=datetime.now().isoformat()
                )
                
                self.device_manager.log_library_device_usage(
                    "SentenceTransformer", 
                    device_str.upper() if device else "AUTO"
                )
                
                self.logger.info(f"✅ Loaded embedding model: {model_name}")
                return model
                
            except Exception as e:
                self.logger.error(f"Failed to load embedding model {model_name}: {e}")
                return None
    
    def get_reranker_model(
        self,
        model_name: Optional[str] = None,
        use_cohere: bool = False,
        cohere_api_key: Optional[str] = None
    ) -> Optional[Union[Reranker, 'CohereReranker']]:
        """
        Get or load a reranker model.
        
        Args:
            model_name: Name of the model to load
            use_cohere: Whether to use Cohere API
            cohere_api_key: Cohere API key if using API
            
        Returns:
            Loaded reranker model or None if unavailable
        """
        model_name = model_name or self.default_reranker_model
        cache_key = f"cohere_{model_name}" if use_cohere else model_name
        
        with self._lock:
            # Return cached model if available
            if cache_key in self._reranker_models:
                self.logger.debug(f"Using cached reranker: {cache_key}")
                return self._reranker_models[cache_key]
            
            # Load new model
            if use_cohere:
                return self._load_cohere_reranker(model_name, cohere_api_key, cache_key)
            else:
                return self._load_local_reranker(model_name, cache_key)
    
    def _load_local_reranker(self, model_name: str, cache_key: str) -> Optional[Reranker]:
        """Load local reranker model."""
        if not RERANKERS_AVAILABLE:
            self.logger.warning("rerankers library not available")
            return None
        
        self.logger.info(f"🚀 Loading local reranker: {model_name}")
        try:
            device = self.device_manager.get_torch_device()
            reranker = Reranker(model_name, device=device)
            
            # Cache the model
            self._reranker_models[cache_key] = reranker
            
            # Store model info
            self._model_info[f"reranker_{cache_key}"] = ModelInfo(
                model_name=model_name,
                model_type="reranker",
                device=device,
                loaded_at=datetime.now().isoformat()
            )
            
            self.device_manager.log_library_device_usage("Reranker", device.upper())
            self.logger.info(f"✅ Loaded local reranker: {model_name}")
            return reranker
            
        except Exception as e:
            self.logger.error(f"Failed to load local reranker {model_name}: {e}")
            return None
    
    def _load_cohere_reranker(
        self, 
        model_name: str, 
        api_key: Optional[str], 
        cache_key: str
    ) -> Optional['CohereReranker']:
        """Load Cohere API reranker."""
        if not COHERE_AVAILABLE:
            self.logger.warning("cohere library not available")
            return None
        
        if not api_key:
            self.logger.error("Cohere API key required for Cohere reranker")
            return None
        
        self.logger.info(f"🚀 Loading Cohere reranker: {model_name}")
        try:
            from .cohere_wrapper import CohereReranker  # Local wrapper class
            reranker = CohereReranker(api_key=api_key, model_name=model_name)
            
            # Cache the model
            self._reranker_models[cache_key] = reranker
            
            # Store model info
            self._model_info[f"reranker_{cache_key}"] = ModelInfo(
                model_name=model_name,
                model_type="reranker",
                device="api",
                loaded_at=datetime.now().isoformat()
            )
            
            self.logger.info(f"✅ Loaded Cohere reranker: {model_name}")
            return reranker
            
        except Exception as e:
            self.logger.error(f"Failed to load Cohere reranker {model_name}: {e}")
            return None
    
    def get_chroma_embedding_function(
        self, 
        model_name: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get or create a ChromaDB embedding function.
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            ChromaDB embedding function or None if unavailable
        """
        if not CHROMADB_AVAILABLE:
            self.logger.warning("chromadb not available")
            return None
        
        model_name = model_name or self.default_embedding_model
        
        with self._lock:
            # Return cached function if available
            if model_name in self._chroma_embedding_functions:
                self.logger.debug(f"Using cached ChromaDB embedding function: {model_name}")
                return self._chroma_embedding_functions[model_name]
            
            # Create new embedding function
            self.logger.info(f"🚀 Creating ChromaDB embedding function: {model_name}")
            try:
                embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=model_name
                )
                
                # Cache the function
                self._chroma_embedding_functions[model_name] = embedding_function
                
                # Store model info
                self._model_info[f"chroma_{model_name}"] = ModelInfo(
                    model_name=model_name,
                    model_type="chroma_embedding",
                    device="auto",
                    loaded_at=datetime.now().isoformat()
                )
                
                self.logger.info(f"✅ Created ChromaDB embedding function: {model_name}")
                return embedding_function
                
            except Exception as e:
                self.logger.error(f"Failed to create ChromaDB embedding function {model_name}: {e}")
                return None
    
    def is_model_loaded(self, model_name: str, model_type: str = "embedding") -> bool:
        """
        Check if a model is already loaded.
        
        Args:
            model_name: Name of the model
            model_type: Type of model ('embedding', 'reranker', 'chroma_embedding')
            
        Returns:
            True if model is loaded, False otherwise
        """
        with self._lock:
            if model_type == "embedding":
                return model_name in self._embedding_models
            elif model_type == "reranker":
                return model_name in self._reranker_models
            elif model_type == "chroma_embedding":
                return model_name in self._chroma_embedding_functions
            else:
                return False
    
    def get_loaded_models(self) -> Dict[str, ModelInfo]:
        """
        Get information about all loaded models.
        
        Returns:
            Dictionary mapping model keys to ModelInfo objects
        """
        with self._lock:
            return self._model_info.copy()
    
    def clear_models(self, model_type: Optional[str] = None):
        """
        Clear loaded models from cache.
        
        Args:
            model_type: Type of models to clear, or None for all models
        """
        with self._lock:
            if model_type is None or model_type == "embedding":
                self._embedding_models.clear()
            
            if model_type is None or model_type == "reranker":
                self._reranker_models.clear()
            
            if model_type is None or model_type == "chroma_embedding":
                self._chroma_embedding_functions.clear()
            
            # Clear corresponding model info
            keys_to_remove = []
            for key, info in self._model_info.items():
                if model_type is None or info.model_type == model_type:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._model_info[key]
            
            self.logger.info(f"🧹 Cleared {model_type or 'all'} models from cache")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get approximate memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        stats = {
            "total_models_loaded": len(self._model_info),
            "embedding_models": len(self._embedding_models),
            "reranker_models": len(self._reranker_models),
            "chroma_embedding_functions": len(self._chroma_embedding_functions),
            "device_info": self.device_manager.get_device_summary()
        }
        
        return stats


# Simple Cohere wrapper class (to avoid circular imports)
class CohereReranker:
    """Simple wrapper for Cohere reranking API"""
    
    def __init__(self, api_key: str, model_name: str = "rerank-english-v3.0"):
        import cohere
        self.client = cohere.Client(api_key)
        self.model_name = model_name
    
    def rank(self, query: str, docs: list, top_n: int = None):
        """Rank documents using Cohere API"""
        response = self.client.rerank(
            query=query,
            documents=docs,
            model=self.model_name,
            top_n=top_n or len(docs)
        )
        return response


# Global model manager instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager