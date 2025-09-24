"""
Knowledge Type Definitions and Enums

This module defines the knowledge types supported by the multi-RAG system
and related enumerations for type-safe knowledge base operations.
"""

from enum import Enum
from typing import Literal


class KnowledgeType(Enum):
    """
    Enumeration of supported knowledge types in the multi-RAG system.
    
    Each knowledge type has its own pipeline, storage, and caching strategy.
    """
    TRANSCRIPTS = "transcripts"
    MENTAL_MODELS = "mental_models" 
    CORE_BELIEFS = "core_beliefs"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def collection_suffix(self) -> str:
        """Get ChromaDB collection suffix for this knowledge type."""
        return f"_{self.value}"
    
    @property
    def cache_directory(self) -> str:
        """Get cache directory name for this knowledge type."""
        return self.value
        
    @property
    def display_name(self) -> str:
        """Get human-readable display name."""
        display_names = {
            KnowledgeType.TRANSCRIPTS: "Transcripts",
            KnowledgeType.MENTAL_MODELS: "Mental Models", 
            KnowledgeType.CORE_BELIEFS: "Core Beliefs"
        }
        return display_names[self]
    
    @property
    def uses_hyde(self) -> bool:
        """Whether this knowledge type uses HyDE hypothesis generation."""
        return self == KnowledgeType.TRANSCRIPTS
    
    @property
    def uses_bm25(self) -> bool:
        """Whether this knowledge type uses BM25 indexing."""
        return self == KnowledgeType.TRANSCRIPTS
    
    @property
    def uses_reranking(self) -> bool:
        """Whether this knowledge type uses cross-encoder reranking."""
        return True  # All knowledge types use reranking
    
    @classmethod
    def from_string(cls, value: str) -> 'KnowledgeType':
        """
        Create KnowledgeType from string value.
        
        Args:
            value: String representation of knowledge type
            
        Returns:
            KnowledgeType enum value
            
        Raises:
            ValueError: If value is not a valid knowledge type
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid_types = ", ".join([kt.value for kt in cls])
            raise ValueError(f"Invalid knowledge type: {value}. Valid types: {valid_types}")


class PipelineComponent(Enum):
    """
    Pipeline components used in retrieval.
    """
    HYDE = "hyde"
    VECTOR_SEARCH = "vector_search"
    BM25_SEARCH = "bm25_search" 
    HYBRID_SEARCH = "hybrid_search"
    RERANKING = "reranking"


class CacheOperation(Enum):
    """
    Types of cache operations for key generation.
    """
    HYDE_GENERATION = "hyde_generation"
    VECTOR_SEARCH = "vector_search"
    BM25_SEARCH = "bm25_search"
    RERANKING = "reranking"
    PIPELINE_EXECUTION = "pipeline_execution"


# Type aliases for better type hints
KnowledgeTypeStr = Literal["transcripts", "mental_models", "core_beliefs"]
PipelineComponentStr = Literal["hyde", "vector_search", "bm25_search", "hybrid_search", "reranking"]
CacheOperationStr = Literal["hyde_generation", "vector_search", "bm25_search", "reranking", "pipeline_execution"]


def get_pipeline_components(knowledge_type: KnowledgeType) -> list[PipelineComponent]:
    """
    Get the pipeline components used by a specific knowledge type.
    
    Args:
        knowledge_type: The knowledge type to get components for
        
    Returns:
        List of pipeline components used by the knowledge type
    """
    if knowledge_type == KnowledgeType.TRANSCRIPTS:
        # Full Phase 2 pipeline
        return [
            PipelineComponent.HYDE,
            PipelineComponent.HYBRID_SEARCH,  # BM25 + Vector
            PipelineComponent.RERANKING
        ]
    elif knowledge_type in [KnowledgeType.MENTAL_MODELS, KnowledgeType.CORE_BELIEFS]:
        # Simplified pipeline
        return [
            PipelineComponent.VECTOR_SEARCH,
            PipelineComponent.RERANKING
        ]
    else:
        raise ValueError(f"Unknown knowledge type: {knowledge_type}")


def get_cache_operations(knowledge_type: KnowledgeType) -> list[CacheOperation]:
    """
    Get the cache operations supported by a specific knowledge type.
    
    Args:
        knowledge_type: The knowledge type to get cache operations for
        
    Returns:
        List of cache operations supported by the knowledge type
    """
    if knowledge_type == KnowledgeType.TRANSCRIPTS:
        return [
            CacheOperation.HYDE_GENERATION,
            CacheOperation.BM25_SEARCH,
            CacheOperation.VECTOR_SEARCH,
            CacheOperation.RERANKING,
            CacheOperation.PIPELINE_EXECUTION
        ]
    elif knowledge_type in [KnowledgeType.MENTAL_MODELS, KnowledgeType.CORE_BELIEFS]:
        return [
            CacheOperation.VECTOR_SEARCH,
            CacheOperation.RERANKING,
            CacheOperation.PIPELINE_EXECUTION
        ]
    else:
        raise ValueError(f"Unknown knowledge type: {knowledge_type}")


def validate_knowledge_type(knowledge_type: str | KnowledgeType) -> KnowledgeType:
    """
    Validate and convert knowledge type to enum.
    
    Args:
        knowledge_type: Knowledge type as string or enum
        
    Returns:
        Validated KnowledgeType enum
        
    Raises:
        ValueError: If knowledge type is invalid
    """
    if isinstance(knowledge_type, str):
        return KnowledgeType.from_string(knowledge_type)
    elif isinstance(knowledge_type, KnowledgeType):
        return knowledge_type
    else:
        raise ValueError(f"Invalid knowledge type format: {type(knowledge_type)}")