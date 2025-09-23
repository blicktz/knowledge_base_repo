"""
Pydantic models for the Virtual Influencer Persona Agent system.

This module defines the core data structures for storing and validating
an influencer's persona constitution, including their linguistic style,
mental models, core beliefs, and metadata.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field, validator
from enum import Enum


class FormalityLevel(str, Enum):
    """Levels of formality in communication."""
    VERY_FORMAL = "very_formal"
    FORMAL = "formal"
    NEUTRAL = "neutral"
    INFORMAL = "informal"
    VERY_INFORMAL = "very_informal"


class DirectnessLevel(str, Enum):
    """Levels of directness in communication."""
    VERY_INDIRECT = "very_indirect"
    INDIRECT = "indirect"
    NEUTRAL = "neutral"
    DIRECT = "direct"
    VERY_DIRECT = "very_direct"


class FrequencyLevel(str, Enum):
    """Frequency levels for various communication aspects."""
    NEVER = "never"
    RARE = "rare"
    OCCASIONAL = "occasional"
    FREQUENT = "frequent"
    CONSTANT = "constant"


class CommunicationStyle(BaseModel):
    """Communication style characteristics."""
    formality: FormalityLevel = Field(..., description="Level of formality in communication")
    directness: DirectnessLevel = Field(..., description="Level of directness in communication")
    use_of_examples: FrequencyLevel = Field(..., description="Frequency of using examples")
    storytelling: FrequencyLevel = Field(..., description="Frequency of storytelling")
    humor: FrequencyLevel = Field(..., description="Frequency of humor usage")


class LinguisticStyle(BaseModel):
    """Linguistic style and communication patterns of the influencer."""
    
    tone: str = Field(..., description="Overall tone and energy of communication")
    catchphrases: List[str] = Field(
        default_factory=list,
        description="Signature phrases and expressions frequently used"
    )
    vocabulary: List[str] = Field(
        default_factory=list,
        description="Characteristic vocabulary and terminology"
    )
    sentence_structures: List[str] = Field(
        default_factory=list,
        description="Typical sentence patterns and structures"
    )
    communication_style: CommunicationStyle = Field(
        ...,
        description="Detailed communication style characteristics"
    )

    @validator('catchphrases')
    def validate_catchphrases(cls, v):
        """Ensure catchphrases are non-empty strings."""
        return [phrase.strip() for phrase in v if phrase.strip()]

    @validator('vocabulary')
    def validate_vocabulary(cls, v):
        """Ensure vocabulary items are non-empty strings."""
        return [word.strip() for word in v if word.strip()]


class MentalModel(BaseModel):
    """A mental model or framework used by the influencer."""
    
    name: str = Field(..., description="Name of the mental model or framework")
    description: str = Field(..., description="Description of the mental model")
    steps: List[str] = Field(
        default_factory=list,
        description="Step-by-step breakdown of the model"
    )
    categories: List[str] = Field(
        default_factory=list,
        description="Categories or domains where this model applies"
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this mental model (0-1)"
    )
    frequency: int = Field(
        default=1,
        ge=1,
        description="Number of times this model was referenced"
    )

    @validator('name')
    def validate_name(cls, v):
        """Ensure name is not empty."""
        if not v.strip():
            raise ValueError("Mental model name cannot be empty")
        return v.strip()

    @validator('steps')
    def validate_steps(cls, v):
        """Ensure steps are non-empty strings."""
        return [step.strip() for step in v if step.strip()]


class CoreBelief(BaseModel):
    """A core belief or principle held by the influencer."""
    
    statement: str = Field(..., description="The belief statement")
    category: str = Field(
        default="general",
        description="Category of the belief (e.g., productivity, business, life)"
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this belief (0-1)"
    )
    frequency: int = Field(
        default=1,
        ge=1,
        description="Number of times this belief was expressed"
    )
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="Examples or evidence supporting this belief"
    )

    @validator('statement')
    def validate_statement(cls, v):
        """Ensure statement is not empty."""
        if not v.strip():
            raise ValueError("Core belief statement cannot be empty")
        return v.strip()


class CollocationItem(BaseModel):
    """A single collocation item with metadata."""
    ngram: str = Field(..., description="The n-gram phrase")
    type: str = Field(..., description="Type of n-gram (bigram, trigram, etc.)")
    frequency: int = Field(..., ge=0, description="Frequency of occurrence")
    score: float = Field(..., description="Statistical significance score")


class StatisticalReport(BaseModel):
    """Statistical analysis report of the content."""
    
    total_words: int = Field(default=0, ge=0, description="Total word count")
    total_documents: int = Field(default=0, ge=0, description="Total number of documents")
    total_sentences: int = Field(default=0, ge=0, description="Total number of sentences")
    avg_words_per_document: float = Field(default=0.0, ge=0.0, description="Average words per document")
    
    # Linguistic metrics
    unique_words: int = Field(default=0, ge=0, description="Number of unique words")
    vocabulary_richness: float = Field(default=0.0, ge=0.0, le=1.0, description="Vocabulary richness ratio")
    avg_sentence_length: float = Field(default=0.0, ge=0.0, description="Average sentence length")
    
    # Content analysis - matching actual statistical_analyzer.py output format
    top_keywords: Dict[str, int] = Field(default_factory=dict, description="Most frequent keywords with counts")
    top_entities: Dict[str, int] = Field(default_factory=dict, description="Most frequent named entities with counts")  
    top_collocations: List[CollocationItem] = Field(default_factory=list, description="Most frequent collocations with metadata")
    
    # Advanced metrics
    readability_metrics: Dict[str, Any] = Field(default_factory=dict, description="Readability analysis results")
    linguistic_patterns: Dict[str, Any] = Field(default_factory=dict, description="Linguistic pattern analysis")
    sentiment_analysis: Dict[str, Any] = Field(default_factory=dict, description="Sentiment analysis results")
    
    # Processing metadata
    processing_time_seconds: float = Field(default=0.0, ge=0.0, description="Time taken for analysis")
    analysis_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When analysis was performed")
    
    class Config:
        """Pydantic configuration for proper JSON serialization."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class ExtractionMetadata(BaseModel):
    """Metadata about the persona extraction process."""
    
    extraction_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the extraction was performed"
    )
    total_processing_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total time taken for extraction"
    )
    llm_provider: str = Field(default="unknown", description="LLM provider used")
    llm_model: str = Field(default="unknown", description="LLM model used")
    source_documents_count: int = Field(default=0, ge=0, description="Number of source documents")
    total_tokens_processed: int = Field(default=0, ge=0, description="Total tokens processed")
    
    # Quality metrics
    mental_models_extracted: int = Field(default=0, ge=0, description="Number of mental models extracted")
    core_beliefs_extracted: int = Field(default=0, ge=0, description="Number of core beliefs extracted")
    avg_confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average confidence score across all extractions"
    )
    
    # Processing configuration
    use_map_reduce: bool = Field(default=False, description="Whether map-reduce extraction was used")
    batch_size: Optional[int] = Field(default=None, description="Batch size for map-reduce")
    parallel_batches: Optional[int] = Field(default=None, description="Number of parallel batches")
    
    # Cache information
    cache_hits: int = Field(default=0, ge=0, description="Number of cache hits during processing")
    cache_misses: int = Field(default=0, ge=0, description="Number of cache misses during processing")
    
    # Quality assessment scores (populated after extraction)
    quality_scores: Optional[Dict[str, float]] = Field(
        default=None, 
        description="Quality assessment scores for extraction components"
    )


class PersonaConstitution(BaseModel):
    """
    Complete persona constitution for an influencer.
    
    This is the main container for all extracted persona information,
    including linguistic style, mental models, core beliefs, and metadata.
    """
    
    # Core persona components
    linguistic_style: LinguisticStyle = Field(..., description="Communication style and patterns")
    mental_models: List[MentalModel] = Field(
        default_factory=list,
        description="Problem-solving frameworks and mental models"
    )
    core_beliefs: List[CoreBelief] = Field(
        default_factory=list,
        description="Fundamental beliefs and principles"
    )
    
    # Analysis and metadata
    statistical_report: Optional[StatisticalReport] = Field(
        default=None,
        description="Statistical analysis of the source content"
    )
    extraction_metadata: ExtractionMetadata = Field(
        default_factory=ExtractionMetadata,
        description="Metadata about the extraction process"
    )
    
    # Quality metrics
    overall_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall quality score of the persona extraction"
    )
    completeness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Completeness score based on available components"
    )

    @validator('mental_models')
    def validate_mental_models(cls, v):
        """Ensure mental models are sorted by confidence score."""
        return sorted(v, key=lambda x: x.confidence_score, reverse=True)

    @validator('core_beliefs')
    def validate_core_beliefs(cls, v):
        """Ensure core beliefs are sorted by confidence score."""
        return sorted(v, key=lambda x: x.confidence_score, reverse=True)

    def get_quality_summary(self) -> Dict[str, Any]:
        """Get a summary of the persona quality metrics."""
        return {
            "overall_quality": self.overall_quality_score,
            "completeness": self.completeness_score,
            "mental_models_count": len(self.mental_models),
            "core_beliefs_count": len(self.core_beliefs),
            "avg_mental_model_confidence": (
                sum(m.confidence_score for m in self.mental_models) / len(self.mental_models)
                if self.mental_models else 0.0
            ),
            "avg_core_belief_confidence": (
                sum(b.confidence_score for b in self.core_beliefs) / len(self.core_beliefs)
                if self.core_beliefs else 0.0
            ),
            "has_statistical_report": self.statistical_report is not None,
            "processing_time": self.extraction_metadata.total_processing_time_seconds,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the persona extraction results."""
        return {
            "total_mental_models": len(self.mental_models),
            "total_core_beliefs": len(self.core_beliefs),
            "total_catchphrases": len(self.linguistic_style.catchphrases),
            "total_vocabulary_terms": len(self.linguistic_style.vocabulary),
            "processing_time": f"{self.extraction_metadata.total_processing_time_seconds:.2f}s",
            "overall_quality": self.overall_quality_score,
            "completeness": self.completeness_score,
        }

    def get_top_mental_models(self, n: int = 10) -> List[MentalModel]:
        """Get the top N mental models by confidence score."""
        return self.mental_models[:n]

    def get_top_core_beliefs(self, n: int = 20) -> List[CoreBelief]:
        """Get the top N core beliefs by confidence score."""
        return self.core_beliefs[:n]

    def get_mental_models_by_category(self, category: str) -> List[MentalModel]:
        """Get mental models that belong to a specific category."""
        return [
            model for model in self.mental_models
            if category.lower() in [cat.lower() for cat in model.categories]
        ]

    def get_core_beliefs_by_category(self, category: str) -> List[CoreBelief]:
        """Get core beliefs that belong to a specific category."""
        return [
            belief for belief in self.core_beliefs
            if belief.category.lower() == category.lower()
        ]

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
        json_schema_extra = {
            "example": {
                "linguistic_style": {
                    "tone": "Energetic, direct, and conversational with entrepreneur mindset",
                    "catchphrases": ["What's up everybody", "The key takeaway is", "Here's the bottom line"],
                    "vocabulary": ["leverage", "framework", "execution", "value proposition"],
                    "sentence_structures": ["Short punchy statements", "Question and answer format"],
                    "communication_style": {
                        "formality": "informal",
                        "directness": "very_direct",
                        "use_of_examples": "frequent",
                        "storytelling": "frequent",
                        "humor": "occasional"
                    }
                },
                "mental_models": [
                    {
                        "name": "The 3-P Framework for Productivity",
                        "description": "A systematic approach to maximizing productivity",
                        "steps": [
                            "1. Plan your priorities for maximum impact",
                            "2. Protect your time from distractions",
                            "3. Perform with focused execution"
                        ],
                        "categories": ["productivity", "time management"],
                        "confidence_score": 0.95,
                        "frequency": 8
                    }
                ],
                "core_beliefs": [
                    {
                        "statement": "Consistency over intensity leads to long-term success",
                        "category": "productivity",
                        "confidence_score": 0.9,
                        "frequency": 15,
                        "supporting_evidence": ["Multiple references to daily habits", "Emphasis on long-term thinking"]
                    }
                ]
            }
        }