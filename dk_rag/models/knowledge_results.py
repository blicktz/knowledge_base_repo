"""
Knowledge Result Data Models

This module defines the data models for results returned from different
knowledge base searches, providing type-safe and structured access to
mental models, core beliefs, and unified search results.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from langchain.schema import Document

from .knowledge_types import KnowledgeType


@dataclass
class BaseKnowledgeResult:
    """
    Base class for all knowledge search results.
    
    Contains common fields present in all result types.
    """
    # Core result data
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    
    # Retrieval metadata
    knowledge_type: KnowledgeType = field(default=KnowledgeType.TRANSCRIPTS)
    persona_id: str = ""
    retrieval_timestamp: datetime = field(default_factory=datetime.now)
    retrieval_method: str = "unknown"
    
    # Pipeline metadata
    used_hyde: bool = False
    used_bm25: bool = False
    used_reranking: bool = False
    
    @classmethod
    def from_document(
        cls,
        doc: Document,
        knowledge_type: KnowledgeType,
        persona_id: str,
        retrieval_method: str = "unknown",
        used_hyde: bool = False,
        used_bm25: bool = False,
        used_reranking: bool = False,
        score: Optional[float] = None
    ) -> 'BaseKnowledgeResult':
        """
        Create BaseKnowledgeResult from LangChain Document.
        
        Args:
            doc: LangChain Document object
            knowledge_type: Type of knowledge this result represents
            persona_id: ID of the persona
            retrieval_method: Method used for retrieval
            used_hyde: Whether HyDE was used
            used_bm25: Whether BM25 was used  
            used_reranking: Whether reranking was used
            score: Relevance score if available
            
        Returns:
            BaseKnowledgeResult instance
        """
        return cls(
            content=doc.page_content,
            metadata=doc.metadata.copy() if doc.metadata else {},
            score=score,
            knowledge_type=knowledge_type,
            persona_id=persona_id,
            retrieval_method=retrieval_method,
            used_hyde=used_hyde,
            used_bm25=used_bm25,
            used_reranking=used_reranking
        )


@dataclass
class MentalModelResult(BaseKnowledgeResult):
    """
    Result from mental models knowledge base search.
    
    Contains structured mental model information extracted from metadata.
    """
    # Mental model specific fields
    name: str = ""
    description: str = ""
    steps: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list) 
    confidence_score: float = 0.0
    frequency: int = 0
    
    def __post_init__(self):
        """Extract mental model fields from metadata after initialization."""
        if self.metadata:
            self.name = self.metadata.get('name', '')
            self.description = self.metadata.get('description', '')
            self.steps = self.metadata.get('steps', [])
            self.categories = self.metadata.get('categories', [])
            self.confidence_score = self.metadata.get('confidence_score', 0.0)
            self.frequency = self.metadata.get('frequency', 0)
    
    @classmethod
    def from_document(
        cls,
        doc: Document,
        persona_id: str,
        retrieval_method: str = "vector_rerank",
        score: Optional[float] = None
    ) -> 'MentalModelResult':
        """
        Create MentalModelResult from LangChain Document.
        """
        # Mental models don't use HyDE or BM25 (simplified pipeline)
        base_result = super().from_document(
            doc=doc,
            knowledge_type=KnowledgeType.MENTAL_MODELS,
            persona_id=persona_id,
            retrieval_method=retrieval_method,
            used_hyde=False,
            used_bm25=False,
            used_reranking=True,  # Always use reranking
            score=score
        )
        
        # Create MentalModelResult with base fields
        return cls(
            content=base_result.content,
            metadata=base_result.metadata,
            score=base_result.score,
            knowledge_type=base_result.knowledge_type,
            persona_id=base_result.persona_id,
            retrieval_timestamp=base_result.retrieval_timestamp,
            retrieval_method=base_result.retrieval_method,
            used_hyde=base_result.used_hyde,
            used_bm25=base_result.used_bm25,
            used_reranking=base_result.used_reranking
        )
    
    def get_formatted_steps(self) -> str:
        """Get formatted string of mental model steps."""
        if not self.steps:
            return "No steps available"
        return "\n".join([f"{i+1}. {step}" for i, step in enumerate(self.steps)])
    
    def get_categories_string(self) -> str:
        """Get formatted string of categories."""
        return ", ".join(self.categories) if self.categories else "Uncategorized"


@dataclass
class CoreBeliefResult(BaseKnowledgeResult):
    """
    Result from core beliefs knowledge base search.
    
    Contains structured core belief information extracted from metadata.
    """
    # Core belief specific fields
    statement: str = ""
    category: str = ""
    supporting_evidence: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    frequency: int = 0
    
    def __post_init__(self):
        """Extract core belief fields from metadata after initialization."""
        if self.metadata:
            self.statement = self.metadata.get('statement', '')
            self.category = self.metadata.get('category', '')
            self.supporting_evidence = self.metadata.get('supporting_evidence', [])
            self.confidence_score = self.metadata.get('confidence_score', 0.0)
            self.frequency = self.metadata.get('frequency', 0)
    
    @classmethod
    def from_document(
        cls,
        doc: Document,
        persona_id: str,
        retrieval_method: str = "vector_rerank",
        score: Optional[float] = None
    ) -> 'CoreBeliefResult':
        """
        Create CoreBeliefResult from LangChain Document.
        """
        # Core beliefs don't use HyDE or BM25 (simplified pipeline)
        base_result = super().from_document(
            doc=doc,
            knowledge_type=KnowledgeType.CORE_BELIEFS,
            persona_id=persona_id,
            retrieval_method=retrieval_method,
            used_hyde=False,
            used_bm25=False,
            used_reranking=True,  # Always use reranking
            score=score
        )
        
        # Create CoreBeliefResult with base fields
        return cls(
            content=base_result.content,
            metadata=base_result.metadata,
            score=base_result.score,
            knowledge_type=base_result.knowledge_type,
            persona_id=base_result.persona_id,
            retrieval_timestamp=base_result.retrieval_timestamp,
            retrieval_method=base_result.retrieval_method,
            used_hyde=base_result.used_hyde,
            used_bm25=base_result.used_bm25,
            used_reranking=base_result.used_reranking
        )
    
    def get_formatted_evidence(self) -> str:
        """Get formatted string of supporting evidence."""
        if not self.supporting_evidence:
            return "No evidence available"
        return "\n".join([f"• {evidence}" for evidence in self.supporting_evidence])
    
    def get_confidence_level(self) -> str:
        """Get human-readable confidence level."""
        if self.confidence_score >= 0.9:
            return "Very High"
        elif self.confidence_score >= 0.8:
            return "High"
        elif self.confidence_score >= 0.7:
            return "Medium"
        elif self.confidence_score >= 0.6:
            return "Low"
        else:
            return "Very Low"


@dataclass  
class IndexingResult:
    """
    Result from knowledge base indexing operations.
    
    Contains statistics and error information from building indexes.
    """
    # Success metrics
    documents_processed: int = 0
    documents_indexed: int = 0
    indexing_duration_seconds: float = 0.0
    
    # Storage information
    vector_store_created: bool = False
    bm25_index_created: bool = False
    index_size_mb: float = 0.0
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    knowledge_type: KnowledgeType = field(default=KnowledgeType.TRANSCRIPTS)
    persona_id: str = ""
    source_file: str = ""
    indexing_timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def success(self) -> bool:
        """Whether indexing completed successfully."""
        return len(self.errors) == 0 and self.documents_indexed > 0
    
    @property
    def partial_success(self) -> bool:
        """Whether indexing had some successes despite errors."""
        return self.documents_indexed > 0
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(f"[{datetime.now().isoformat()}] {error}")
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(f"[{datetime.now().isoformat()}] {warning}")
    
    def get_summary(self) -> str:
        """Get a human-readable summary of indexing results."""
        if self.success:
            return (f"Successfully indexed {self.documents_indexed} {self.knowledge_type.display_name.lower()} "
                   f"in {self.indexing_duration_seconds:.2f}s")
        elif self.partial_success:
            return (f"Partially successful: indexed {self.documents_indexed}/{self.documents_processed} "
                   f"{self.knowledge_type.display_name.lower()} with {len(self.errors)} errors")
        else:
            return f"Failed to index {self.knowledge_type.display_name.lower()}: {len(self.errors)} errors"


@dataclass
class SearchStatistics:
    """
    Statistics about a search operation across all knowledge types.
    """
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_latency_ms: float = 0.0
    
    # Per-knowledge-type stats
    transcripts_queries: int = 0
    mental_models_queries: int = 0
    core_beliefs_queries: int = 0
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def add_query(
        self, 
        knowledge_type: KnowledgeType, 
        latency_ms: float, 
        cache_hit: bool
    ):
        """Add statistics for a query."""
        self.total_queries += 1
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        # Update running average
        if self.total_queries == 1:
            self.average_latency_ms = latency_ms
        else:
            self.average_latency_ms = (
                (self.average_latency_ms * (self.total_queries - 1) + latency_ms) / 
                self.total_queries
            )
        
        # Update per-type counters
        if knowledge_type == KnowledgeType.TRANSCRIPTS:
            self.transcripts_queries += 1
        elif knowledge_type == KnowledgeType.MENTAL_MODELS:
            self.mental_models_queries += 1
        elif knowledge_type == KnowledgeType.CORE_BELIEFS:
            self.core_beliefs_queries += 1


# Type aliases for result unions
KnowledgeResult = Union[BaseKnowledgeResult, MentalModelResult, CoreBeliefResult]
AnyResult = Union[MentalModelResult, CoreBeliefResult, BaseKnowledgeResult]