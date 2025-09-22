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
    artifacts_dir: str = Field(default="./data/storage/artifacts", description="Artifacts directory")
    vector_db_dir: str = Field(default="./data/storage/chroma_db", description="Vector DB directory") 
    cache_dir: str = Field(default="./data/storage/cache", description="Cache directory")
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
        """Ensure all required directories exist"""
        directories = [
            self.storage.artifacts_dir,
            self.storage.vector_db_dir,
            self.storage.cache_dir,
            self.storage.logs_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration with resolved environment variables"""
        return self.llm.config
    
    def get_vector_db_path(self) -> str:
        """Get absolute path to vector database"""
        return str(Path(self.storage.vector_db_dir).resolve())
    
    def get_artifacts_path(self) -> str:
        """Get absolute path to artifacts directory"""
        return str(Path(self.storage.artifacts_dir).resolve())
    
    def get_cache_path(self) -> str:
        """Get absolute path to cache directory"""
        return str(Path(self.storage.cache_dir).resolve())
    
    def get_logs_path(self) -> str:
        """Get absolute path to logs directory"""
        return str(Path(self.storage.logs_dir).resolve())
    
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
        
        # Check directory permissions
        try:
            test_file = Path(self.storage.artifacts_dir) / "test_write"
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            issues.append(f"No write permission to artifacts directory: {self.storage.artifacts_dir}")
        except Exception as e:
            issues.append(f"Cannot access artifacts directory: {e}")
        
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