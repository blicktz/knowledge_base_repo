"""
Configuration management for the Virtual Influencer Persona Agent
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Phase 2 retrieval configuration
from .retrieval_config import Phase2RetrievalConfig


class LLMConfig(BaseModel):
    """Configuration for LLM providers"""
    provider: str = Field(default="openrouter", description="LLM provider")
    config: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific configuration")
    
    @validator('config')
    def resolve_env_vars(cls, v):
        """Resolve environment variables in configuration"""
        resolved = {}
        for key, value in v.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                resolved[key] = os.getenv(env_var, '')
            else:
                resolved[key] = value
        return resolved


class VectorDBConfig(BaseModel):
    """Configuration for vector database"""
    provider: str = Field(default="chromadb", description="Vector database provider")
    config: Dict[str, Any] = Field(default_factory=dict, description="Database configuration")


class StatisticalAnalysisConfig(BaseModel):
    """Configuration for statistical analysis components"""
    spacy: Dict[str, Any] = Field(default_factory=dict, description="spaCy configuration")
    nltk: Dict[str, Any] = Field(default_factory=dict, description="NLTK configuration") 
    keywords: Dict[str, Any] = Field(default_factory=dict, description="Keyword extraction config")


class PersonaExtractionConfig(BaseModel):
    """Configuration for persona extraction process"""
    mental_models: Dict[str, Any] = Field(default_factory=dict, description="Mental models extraction config")
    core_beliefs: Dict[str, Any] = Field(default_factory=dict, description="Core beliefs extraction config")
    linguistic_style: Dict[str, Any] = Field(default_factory=dict, description="Linguistic style config")
    quality_thresholds: Dict[str, Any] = Field(default_factory=dict, description="Quality thresholds")


class DataProcessingConfig(BaseModel):
    """Configuration for data processing pipeline"""
    chunk_strategy: str = Field(default="semantic", description="Chunking strategy")
    chunk_size: int = Field(default=1000, description="Default chunk size")
    chunk_overlap: int = Field(default=200, description="Chunk overlap")
    min_chunk_length: int = Field(default=100, description="Minimum chunk length")
    max_chunk_length: int = Field(default=2000, description="Maximum chunk length")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Text filters")
    preprocessing: Dict[str, Any] = Field(default_factory=dict, description="Preprocessing options")


class StorageConfig(BaseModel):
    """Configuration for data storage"""
    base_storage_dir: str = Field(default="./data/storage", description="Base storage directory for all data")
    logs_dir: str = Field(default="./logs", description="Logs directory")
    backup: Dict[str, Any] = Field(default_factory=dict, description="Backup configuration")
    compression: Dict[str, Any] = Field(default_factory=dict, description="Compression settings")


class LoggingConfig(BaseModel):
    """Configuration for logging"""
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    handlers: Dict[str, Any] = Field(default_factory=dict, description="Log handlers")


class PerformanceConfig(BaseModel):
    """Configuration for performance optimization"""
    max_workers: int = Field(default=4, description="Maximum worker threads")
    batch_size: int = Field(default=100, description="Batch size for processing")
    cache_size: int = Field(default=1000, description="Cache size")
    memory_limit_mb: int = Field(default=2048, description="Memory limit in MB")
    optimization: Dict[str, Any] = Field(default_factory=dict, description="Optimization settings")


class ValidationConfig(BaseModel):
    """Configuration for validation"""
    strict_mode: bool = Field(default=False, description="Enable strict validation")
    auto_fix: bool = Field(default=True, description="Auto-fix validation issues")
    required_fields: Dict[str, List[str]] = Field(default_factory=dict, description="Required fields")
    quality_checks: Dict[str, Any] = Field(default_factory=dict, description="Quality check thresholds")


class DevelopmentConfig(BaseModel):
    """Configuration for development and testing"""
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    save_intermediate_results: bool = Field(default=True, description="Save intermediate results")
    profile_performance: bool = Field(default=False, description="Enable performance profiling")
    testing: Dict[str, Any] = Field(default_factory=dict, description="Testing configuration")


class MapReduceExtractionConfig(BaseModel):
    """Configuration for map-reduce extraction process"""
    enabled: bool = Field(default=True, description="Enable map-reduce extraction")
    skip_reduce: bool = Field(default=False, description="Skip reduce phase and output mapped results directly")
    llm_provider: str = Field(default="litellm", description="LLM provider for map-reduce")
    llm_model: str = Field(default="gemini/gemini-2.0-flash-exp", description="Default model for extraction")
    map_phase_model: str = Field(default="gemini/gemini-2.0-flash-exp", description="Model for map phase")
    reduce_phase_model: str = Field(default="gemini/gemini-2.0-flash-exp", description="Model for reduce phase")
    
    # Batch processing
    batch_size: int = Field(default=10, description="Documents per batch")
    max_tokens_per_batch: int = Field(default=30000, description="Max tokens per batch")
    parallel_batches: int = Field(default=3, description="Concurrent batch processes")
    
    # Caching
    cache_batch_results: bool = Field(default=True, description="Cache individual batch results")
    resume_from_cache: bool = Field(default=True, description="Resume from cached batches")
    cache_compression: bool = Field(default=True, description="Compress cached results")
    cache_ttl_hours: int = Field(default=168, description="Cache time-to-live in hours")
    
    # Processing
    max_retries: int = Field(default=2, description="Max retries for failed batches")
    retry_delay_seconds: int = Field(default=5, description="Delay between retries")
    timeout_seconds: int = Field(default=120, description="Timeout per batch")
    
    # Consolidation
    mental_models: Dict[str, Any] = Field(default_factory=dict, description="Mental models consolidation config")
    core_beliefs: Dict[str, Any] = Field(default_factory=dict, description="Core beliefs consolidation config")
    
    # Progress
    show_progress: bool = Field(default=True, description="Show progress bars")
    save_intermediate: bool = Field(default=True, description="Save after each batch")


# Phase 3 Configuration Classes
class AgentToolsConfig(BaseModel):
    """Configuration for agent tools"""
    persona_data: Dict[str, Any] = Field(default_factory=dict)
    mental_models: Dict[str, Any] = Field(default_factory=dict) 
    core_beliefs: Dict[str, Any] = Field(default_factory=dict)
    transcripts: Dict[str, Any] = Field(default_factory=dict)
    parallel_execution: bool = Field(default=True)
    timeout_seconds: int = Field(default=30)
    fail_fast: bool = Field(default=True)


class AgentLLMConfig(BaseModel):
    """Configuration for agent LLM calls"""
    llm_provider: str = Field(default="litellm")
    llm_model: str = Field(default="gemini/gemini-2.0-flash")
    temperature: float = Field(default=0.3)
    max_tokens: int = Field(default=1000)
    timeout_seconds: int = Field(default=20)
    max_retries: int = Field(default=2)
    cache_results: bool = Field(default=True)
    cache_ttl_hours: int = Field(default=24)
    log_interactions: bool = Field(default=True)


class AgentSynthesisConfig(BaseModel):
    """Configuration for response synthesis"""
    llm_provider: str = Field(default="litellm")
    llm_model: str = Field(default="gemini/gemini-2.5-pro")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4000)
    timeout_seconds: int = Field(default=60)
    max_retries: int = Field(default=2)
    use_chain_of_thought: bool = Field(default=True)
    include_scratchpad: bool = Field(default=True)
    log_interactions: bool = Field(default=True)


class AgentLLMLoggingConfig(BaseModel):
    """Configuration for persona-specific LLM logging"""
    enabled: bool = Field(default=True)
    directory_name: str = Field(default="llm_logging")
    save_prompts: bool = Field(default=True)
    save_responses: bool = Field(default=True)
    save_extracted: bool = Field(default=True)
    include_metadata: bool = Field(default=True)
    compression: bool = Field(default=True)
    retention_days: int = Field(default=30)


class AgentLoggingConfig(BaseModel):
    """Configuration for agent logging (legacy)"""
    enabled: bool = Field(default=True)
    save_prompts: bool = Field(default=True)
    save_responses: bool = Field(default=True)
    save_extracted: bool = Field(default=True)
    include_metadata: bool = Field(default=True)
    log_directory: str = Field(default="logging/llm_interactions")
    components: Dict[str, str] = Field(default_factory=dict)


class AgentPerformanceConfig(BaseModel):
    """Configuration for agent performance settings"""
    enable_progress_logging: bool = Field(default=True)
    log_timing_metrics: bool = Field(default=True)
    batch_tool_execution: bool = Field(default=False)
    max_concurrent_tools: int = Field(default=1)


class AgentErrorHandlingConfig(BaseModel):
    """Configuration for agent error handling"""
    use_fallbacks: bool = Field(default=False)
    throw_on_error: bool = Field(default=True)
    include_traceback: bool = Field(default=True)


class AgentMemoryConfig(BaseModel):
    """Configuration for agent memory management"""
    enabled: bool = Field(default=True)
    max_tokens: int = Field(default=6000)
    strategy: str = Field(default="last")
    include_system: bool = Field(default=True)
    start_on: str = Field(default="human")
    end_on: List[str] = Field(default_factory=lambda: ["human", "tool"])


class AgentConfig(BaseModel):
    """Configuration for Phase 3 agent system"""
    enabled: bool = Field(default=True)
    memory: AgentMemoryConfig = Field(default_factory=AgentMemoryConfig)
    llm_logging: AgentLLMLoggingConfig = Field(default_factory=AgentLLMLoggingConfig)
    query_analysis: AgentLLMConfig = Field(default_factory=AgentLLMConfig)
    synthesis: AgentSynthesisConfig = Field(default_factory=AgentSynthesisConfig)
    tools: AgentToolsConfig = Field(default_factory=AgentToolsConfig)
    logging: AgentLoggingConfig = Field(default_factory=AgentLoggingConfig)
    performance: AgentPerformanceConfig = Field(default_factory=AgentPerformanceConfig)
    error_handling: AgentErrorHandlingConfig = Field(default_factory=AgentErrorHandlingConfig)


class APISettingsConfig(BaseModel):
    """Configuration for API settings"""
    title: str = Field(default="Persona Agent API")
    version: str = Field(default="1.0.0")
    docs_url: str = Field(default="/docs")
    redoc_url: str = Field(default="/redoc")


class APIRateLimitConfig(BaseModel):
    """Configuration for API rate limiting"""
    enabled: bool = Field(default=False)
    requests_per_minute: int = Field(default=60)


class APICORSConfig(BaseModel):
    """Configuration for API CORS"""
    enabled: bool = Field(default=True)
    allow_origins: List[str] = Field(default_factory=lambda: ["*"])
    allow_methods: List[str] = Field(default_factory=lambda: ["GET", "POST"])
    allow_headers: List[str] = Field(default_factory=lambda: ["*"])


class APIHealthCheckConfig(BaseModel):
    """Configuration for API health check"""
    enabled: bool = Field(default=True)
    endpoint: str = Field(default="/health")
    include_version: bool = Field(default=True)


class APIConfig(BaseModel):
    """Configuration for FastAPI application"""
    enabled: bool = Field(default=True)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)
    workers: int = Field(default=1)
    settings: APISettingsConfig = Field(default_factory=APISettingsConfig)
    rate_limiting: APIRateLimitConfig = Field(default_factory=APIRateLimitConfig)
    cors: APICORSConfig = Field(default_factory=APICORSConfig)
    health_check: APIHealthCheckConfig = Field(default_factory=APIHealthCheckConfig)


class Settings(BaseModel):
    """Main settings class for the persona agent"""
    
    # Core configuration sections
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    statistical_analysis: StatisticalAnalysisConfig = Field(default_factory=StatisticalAnalysisConfig)
    persona_extraction: PersonaExtractionConfig = Field(default_factory=PersonaExtractionConfig)
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    map_reduce_extraction: MapReduceExtractionConfig = Field(default_factory=MapReduceExtractionConfig)
    retrieval: Phase2RetrievalConfig = Field(default_factory=Phase2RetrievalConfig)
    
    # Phase 3 configuration sections
    agent: AgentConfig = Field(default_factory=AgentConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    # Additional metadata
    version: str = Field(default="1.0.0", description="Configuration version")
    environment: str = Field(default="development", description="Environment name")
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """Initialize settings from config file or kwargs"""
        if config_path:
            config_data = self._load_config_file(config_path)
            super().__init__(**config_data, **kwargs)
        else:
            super().__init__(**kwargs)
        
        # Ensure directories exist
        self._ensure_directories()
    
    @staticmethod
    def _load_config_file(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return config_data or {}
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        # Only create logs directory - persona-specific directories are created by PersonaManager
        Path(self.storage.logs_dir).mkdir(parents=True, exist_ok=True)
    
    @property
    def base_storage_dir(self) -> str:
        """Convenience property to access base storage directory"""
        return self.storage.base_storage_dir
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration with resolved environment variables"""
        return self.llm.config
    
    def get_vector_db_path(self, persona_id: Optional[str] = None) -> str:
        """Get absolute path to vector database for a specific persona"""
        if not persona_id:
            raise ValueError("persona_id is required - single-tenant mode is no longer supported")
        base_dir = Path(self.storage.base_storage_dir) / "personas" / persona_id / "vector_db"
        return str(base_dir.resolve())
    
    def get_artifacts_path(self, persona_id: Optional[str] = None) -> str:
        """Get absolute path to artifacts directory for a specific persona"""
        if not persona_id:
            raise ValueError("persona_id is required - single-tenant mode is no longer supported")
        base_dir = Path(self.storage.base_storage_dir) / "personas" / persona_id / "artifacts"
        return str(base_dir.resolve())
    
    def get_persona_base_path(self, persona_id: str) -> str:
        """Get the base directory path for a specific persona"""
        base_dir = Path(self.storage.base_storage_dir) / "personas" / persona_id
        return str(base_dir.resolve())
    
    def get_logs_path(self) -> str:
        """Get absolute path to logs directory"""
        return str(Path(self.storage.logs_dir).resolve())
    
    def get_personas_base_dir(self) -> str:
        """Get absolute path to personas base directory"""
        return str(Path(self.storage.base_storage_dir) / "personas")
    
    def get_llm_logging_path(self, persona_id: str) -> str:
        """Get absolute path to LLM logging directory for a specific persona"""
        base_dir = Path(self.storage.base_storage_dir) / "personas" / persona_id / self.agent.llm_logging.directory_name
        return str(base_dir.resolve())
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return self.development.debug_mode
    
    def get_chunk_config(self) -> Dict[str, Any]:
        """Get chunking configuration"""
        return {
            "strategy": self.data_processing.chunk_strategy,
            "size": self.data_processing.chunk_size,
            "overlap": self.data_processing.chunk_overlap,
            "min_length": self.data_processing.min_chunk_length,
            "max_length": self.data_processing.max_chunk_length
        }
    
    def get_extraction_thresholds(self) -> Dict[str, Any]:
        """Get extraction quality thresholds"""
        return {
            "mental_models": self.persona_extraction.mental_models,
            "core_beliefs": self.persona_extraction.core_beliefs,
            "linguistic_style": self.persona_extraction.linguistic_style,
            "quality": self.persona_extraction.quality_thresholds
        }
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Check required environment variables
        api_key = self.llm.config.get('api_key')
        if not api_key:
            issues.append("LLM API key not configured")
        
        # Check base storage directory permissions
        try:
            base_storage = Path(self.storage.base_storage_dir)
            base_storage.mkdir(parents=True, exist_ok=True)
            test_file = base_storage / "test_write"
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            issues.append(f"No write permission to storage directory: {base_storage}")
        except Exception as e:
            issues.append(f"Cannot access storage directory: {e}")
        
        # Check model availability
        spacy_model = self.statistical_analysis.spacy.get('model', 'en_core_web_sm')
        try:
            import spacy
            spacy.load(spacy_model)
        except OSError:
            issues.append(f"spaCy model not found: {spacy_model}. Run: python -m spacy download {spacy_model}")
        except ImportError:
            issues.append("spaCy not installed")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return self.dict(by_alias=True, exclude_none=True)
    
    def save_to_file(self, output_path: str):
        """Save configuration to YAML file"""
        config_dict = self.to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def from_default_config(cls) -> "Settings":
        """Create settings from default configuration file"""
        default_config_path = Path(__file__).parent / "persona_config.yaml"
        if default_config_path.exists():
            return cls(config_path=str(default_config_path))
        else:
            return cls()
    
    @classmethod
    def from_file(cls, config_path: str) -> "Settings":
        """Create settings from configuration file"""
        return cls(config_path=config_path)


def load_settings(config_path: Optional[str] = None) -> Settings:
    """Convenience function to load settings"""
    if config_path:
        return Settings.from_file(config_path)
    else:
        return Settings.from_default_config()


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None, reload: bool = False) -> Settings:
    """Get global settings instance"""
    global _settings
    
    if _settings is None or reload:
        _settings = load_settings(config_path)
    
    return _settings