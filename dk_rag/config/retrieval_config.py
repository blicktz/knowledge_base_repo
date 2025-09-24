"""
Phase 2 Advanced Retrieval Configuration

This module contains all configuration settings for the Phase 2
advanced retrieval system.
"""

from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
import os


class HyDEConfig(BaseModel):
    """Configuration for HyDE (Hypothetical Document Embeddings)"""
    
    enabled: bool = Field(default=True, description="Enable HyDE retrieval")
    prompt_template: str = Field(default="default", description="Default prompt template type")
    cache_size: int = Field(default=128, description="LRU cache size for hypotheses")
    cache_ttl_hours: int = Field(default=168, description="Cache time-to-live in hours")
    auto_select_prompt: bool = Field(default=True, description="Auto-select best prompt based on query")
    

class HybridSearchConfig(BaseModel):
    """Configuration for Hybrid Search (BM25 + Vector)"""
    
    enabled: bool = Field(default=True, description="Enable hybrid search")
    bm25_weight: float = Field(default=0.4, description="Weight for BM25 scores")
    vector_weight: float = Field(default=0.6, description="Weight for vector scores")
    bm25_k1: float = Field(default=1.5, description="BM25 k1 parameter")
    bm25_b: float = Field(default=0.75, description="BM25 b parameter")
    retrieval_k_multiplier: int = Field(default=2, description="Retrieve k*multiplier candidates")
    use_rrf: bool = Field(default=False, description="Use Reciprocal Rank Fusion instead of weighted")
    rrf_k: int = Field(default=60, description="RRF k parameter")


class RerankingConfig(BaseModel):
    """Configuration for Cross-Encoder Reranking"""
    
    enabled: bool = Field(default=True, description="Enable reranking")
    model: str = Field(
        default="mixedbread-ai/mxbai-rerank-large-v1",
        description="Reranker model to use"
    )
    use_cohere: bool = Field(default=False, description="Use Cohere API instead of local model")
    cohere_api_key: Optional[str] = Field(default=None, description="Cohere API key")
    cohere_model: str = Field(default="rerank-english-v3.0", description="Cohere rerank model")
    top_k: int = Field(default=5, description="Number of top results after reranking")
    batch_size: int = Field(default=32, description="Batch size for reranking")
    device: str = Field(default="auto", description="Device: auto, cuda, mps, or cpu")


class CachingConfig(BaseModel):
    """Configuration for Caching and Logging"""
    
    enabled: bool = Field(default=True, description="Enable caching")
    hyde_cache_size: int = Field(default=128, description="HyDE cache size")
    rerank_cache_size: int = Field(default=256, description="Reranking cache size")
    cache_ttl_hours: int = Field(default=168, description="Cache TTL in hours")
    enable_compression: bool = Field(default=True, description="Compress cached data")
    log_llm_interactions: bool = Field(default=True, description="Log all LLM interactions")
    log_performance_metrics: bool = Field(default=True, description="Log performance metrics")


class StorageConfig(BaseModel):
    """Storage paths configuration"""
    
    base_storage_dir: str = Field(
        default="/Volumes/J15/aicallgo_data/persona_data_base",
        description="Base storage directory"
    )
    bm25_index_path: Optional[str] = Field(default=None, description="BM25 index path")
    cache_dir: Optional[str] = Field(default=None, description="Cache directory")
    
    def get_bm25_index_path(self, persona_id: Optional[str] = None) -> Path:
        """Get persona-specific BM25 index path"""
        if self.bm25_index_path:
            return Path(self.bm25_index_path)
        
        if not persona_id:
            raise ValueError("persona_id is required - single-tenant mode is no longer supported")
        
        base_dir = Path(self.base_storage_dir)
        return base_dir / "personas" / persona_id / "indexes" / "bm25"
    
    def get_cache_dir(self, persona_id: Optional[str] = None) -> Path:
        """Get persona-specific cache directory"""
        if self.cache_dir:
            return Path(self.cache_dir)
        
        if not persona_id:
            raise ValueError("persona_id is required - single-tenant mode is no longer supported")
        
        base_dir = Path(self.base_storage_dir)
        return base_dir / "personas" / persona_id / "retrieval_cache"


class PipelineConfig(BaseModel):
    """Advanced pipeline configuration"""
    
    default_k: int = Field(default=5, description="Default number of final results")
    default_retrieval_k: int = Field(default=25, description="Default candidates before reranking")
    enable_fallback: bool = Field(default=True, description="Enable fallback to basic search")
    log_pipeline_execution: bool = Field(default=True, description="Log pipeline executions")
    parallel_processing: bool = Field(default=False, description="Enable parallel processing")


class Phase2RetrievalConfig(BaseModel):
    """Complete Phase 2 Retrieval Configuration"""
    
    hyde: HyDEConfig = Field(default_factory=HyDEConfig)
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    caching: CachingConfig = Field(default_factory=CachingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    
    # Global settings
    enabled: bool = Field(default=True, description="Enable Phase 2 retrieval")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Phase2RetrievalConfig":
        """Create config from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> "Phase2RetrievalConfig":
        """Create config from environment variables"""
        config = cls()
        
        # Override with environment variables if present
        if cohere_key := os.getenv("COHERE_API_KEY"):
            config.reranking.cohere_api_key = cohere_key
            
        if storage_dir := os.getenv("RETRIEVAL_STORAGE_DIR"):
            config.storage.base_storage_dir = storage_dir
            
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump()


# Default configuration instance
DEFAULT_RETRIEVAL_CONFIG = Phase2RetrievalConfig()


def load_retrieval_config(config_path: Optional[str] = None) -> Phase2RetrievalConfig:
    """
    Load retrieval configuration from file or use defaults.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Phase2RetrievalConfig instance
    """
    if config_path and Path(config_path).exists():
        import yaml
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        # Extract retrieval section if present
        if "retrieval" in config_data:
            config_data = config_data["retrieval"]
            
        return Phase2RetrievalConfig.from_dict(config_data)
    
    # Try to load from environment
    return Phase2RetrievalConfig.from_env()


def update_persona_config_with_phase2(persona_config_path: str):
    """
    Update existing persona_config.yaml with Phase 2 retrieval settings.
    
    Args:
        persona_config_path: Path to persona_config.yaml
    """
    import yaml
    
    config_path = Path(persona_config_path)
    
    # Load existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add Phase 2 retrieval configuration
    config["retrieval"] = {
        "enabled": True,
        "hyde": {
            "enabled": True,
            "prompt_template": "default",
            "cache_size": 128,
            "auto_select_prompt": True
        },
        "hybrid_search": {
            "enabled": True,
            "bm25_weight": 0.4,
            "vector_weight": 0.6,
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "use_rrf": False
        },
        "reranking": {
            "enabled": True,
            "model": "mixedbread-ai/mxbai-rerank-large-v1",
            "use_cohere": False,
            "top_k": 5,
            "batch_size": 32,
            "device": "auto"
        },
        "caching": {
            "enabled": True,
            "hyde_cache_size": 128,
            "rerank_cache_size": 256,
            "cache_ttl_hours": 168,
            "enable_compression": True,
            "log_llm_interactions": True,
            "log_performance_metrics": True
        },
        "pipeline": {
            "default_k": 5,
            "default_retrieval_k": 25,
            "enable_fallback": True,
            "log_pipeline_execution": True
        }
    }
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Updated {config_path} with Phase 2 retrieval configuration")