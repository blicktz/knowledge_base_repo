"""
Tests for the persona extractor module
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from dk_rag.data.models.persona_constitution import (
    PersonaConstitution,
    LinguisticStyle,
    MentalModel,
    CoreBelief,
    StatisticalReport,
    ExtractionMetadata
)
from dk_rag.config.settings import Settings


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            'content': """
            Listen, here's the truth about building a business. You need to understand 
            three fundamental principles. First, you must solve a real problem. Second, 
            you need to communicate value clearly. Third, you must deliver consistently.
            
            The key takeaway is this: Success isn't about perfection, it's about persistence.
            What's up everybody, today we're talking about the most important skill in business.
            
            Let me tell you a story about when I started my first company. We had no money,
            no connections, and no clear path. But we had one thing - determination.
            
            The 3-P Framework for Productivity has transformed thousands of businesses:
            1. Plan your priorities for maximum impact
            2. Protect your time from distractions  
            3. Perform with focused execution
            
            Remember: Consistency over intensity wins every time. Small daily actions
            compound into massive results. You need to understand that success is a marathon,
            not a sprint.
            """,
            'source': 'test_document_1.md'
        }
    ]


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    settings = Mock(spec=Settings)
    settings.llm.provider = "openrouter"
    settings.llm.config = {
        'api_key': 'test_key',
        'model': 'test_model',
        'temperature': 0.1,
        'max_tokens': 4000
    }
    settings.get_llm_config.return_value = settings.llm.config
    settings.persona_extraction.mental_models = {'min_confidence': 0.7}
    settings.persona_extraction.core_beliefs = {'min_confidence': 0.6}
    settings.persona_extraction.quality_thresholds = {
        'min_total_words': 1000
    }
    return settings


def test_persona_constitution_model():
    """Test PersonaConstitution data model"""
    
    # Create sample data
    linguistic_style = LinguisticStyle(
        tone="Energetic and direct",
        catchphrases=["What's up everybody", "The key takeaway is"],
        vocabulary=["leverage", "framework", "execution"],
        sentence_structures=["Listen, [statement]", "You need to [action]"]
    )
    
    mental_model = MentalModel(
        name="3-P Framework",
        description="A productivity framework",
        steps=["1. Plan", "2. Protect", "3. Perform"],
        confidence_score=0.9
    )
    
    core_belief = CoreBelief(
        statement="Consistency over intensity",
        category="productivity",
        frequency=5,
        confidence_score=0.8
    )
    
    stats_report = StatisticalReport(
        total_documents=1,
        total_words=1000,
        total_sentences=50
    )
    
    metadata = ExtractionMetadata(
        llm_model="test_model",
        total_processing_time=10.5,
        source_documents=["test.md"]
    )
    
    # Create PersonaConstitution
    persona = PersonaConstitution(
        linguistic_style=linguistic_style,
        mental_models=[mental_model],
        core_beliefs=[core_belief],
        statistical_report=stats_report,
        extraction_metadata=metadata
    )
    
    # Test basic properties
    assert persona.linguistic_style.tone == "Energetic and direct"
    assert len(persona.mental_models) == 1
    assert len(persona.core_beliefs) == 1
    assert persona.mental_models[0].name == "3-P Framework"
    assert persona.core_beliefs[0].statement == "Consistency over intensity"
    
    # Test summary generation
    summary = persona.get_summary()
    assert summary['total_mental_models'] == 1
    assert summary['total_core_beliefs'] == 1
    assert summary['total_catchphrases'] == 2
    
    # Test JSON serialization
    json_artifact = persona.to_json_artifact()
    assert 'linguistic_style' in json_artifact
    assert 'mental_models' in json_artifact
    assert 'core_beliefs' in json_artifact
    
    # Test JSON deserialization
    restored_persona = PersonaConstitution.from_json_artifact(json_artifact)
    assert restored_persona.linguistic_style.tone == persona.linguistic_style.tone
    assert len(restored_persona.mental_models) == len(persona.mental_models)


def test_mental_model_validation():
    """Test MentalModel validation"""
    
    # Valid mental model
    valid_model = MentalModel(
        name="Test Framework",
        description="A test framework",
        steps=["Step 1", "Step 2"],
        confidence_score=0.8
    )
    assert len(valid_model.steps) >= 2
    assert 0 <= valid_model.confidence_score <= 1
    
    # Test step formatting
    model_with_unnumbered_steps = MentalModel(
        name="Another Framework",
        description="Description",
        steps=["First step", "Second step", "Third step"]
    )
    # Steps should be automatically numbered
    assert model_with_unnumbered_steps.steps[0].startswith("1.")
    assert model_with_unnumbered_steps.steps[1].startswith("2.")


def test_core_belief_validation():
    """Test CoreBelief validation"""
    
    # Valid core belief
    valid_belief = CoreBelief(
        statement="Test belief statement",
        category="testing",
        frequency=3,
        confidence_score=0.7
    )
    assert valid_belief.frequency >= 1
    assert 0 <= valid_belief.confidence_score <= 1
    assert len(valid_belief.statement) >= 5


def test_linguistic_style_validation():
    """Test LinguisticStyle validation"""
    
    style = LinguisticStyle(
        tone="Professional and informative tone description",
        catchphrases=["", "  ", "Valid phrase", "Another valid phrase"],
        vocabulary=["", "a", "valid", "word"]
    )
    
    # Empty/whitespace catchphrases should be filtered
    assert "" not in style.catchphrases
    assert "  " not in style.catchphrases
    assert "Valid phrase" in style.catchphrases
    
    # Single character words should be filtered from vocabulary
    assert "" not in style.vocabulary
    assert "a" not in style.vocabulary
    assert "valid" in style.vocabulary


def test_persona_constitution_uniqueness_validation():
    """Test that PersonaConstitution enforces uniqueness"""
    
    # Create duplicate mental models and beliefs
    mental_models = [
        MentalModel(
            name="Same Framework",
            description="First",
            steps=["Step 1", "Step 2"]
        ),
        MentalModel(
            name="same framework",  # Different case, should be considered duplicate
            description="Second",
            steps=["Step A", "Step B"]
        ),
        MentalModel(
            name="Different Framework",
            description="Third",
            steps=["Step X", "Step Y"]
        )
    ]
    
    core_beliefs = [
        CoreBelief(
            statement="Same belief",
            category="test"
        ),
        CoreBelief(
            statement="  same belief  ",  # Should be considered duplicate
            category="test2"
        ),
        CoreBelief(
            statement="Different belief",
            category="test"
        )
    ]
    
    persona = PersonaConstitution(
        linguistic_style=LinguisticStyle(tone="Test tone"),
        mental_models=mental_models,
        core_beliefs=core_beliefs,
        statistical_report=StatisticalReport(
            total_documents=1,
            total_words=100,
            total_sentences=10
        ),
        extraction_metadata=ExtractionMetadata(
            llm_model="test",
            total_processing_time=1.0
        )
    )
    
    # Should have filtered duplicates
    assert len(persona.mental_models) == 2
    assert len(persona.core_beliefs) == 2


def test_statistical_report():
    """Test StatisticalReport model"""
    
    report = StatisticalReport(
        total_documents=5,
        total_words=10000,
        total_sentences=500,
        top_keywords={"keyword1": 100, "keyword2": 80},
        top_entities={"Entity1 (PERSON)": 50, "Entity2 (ORG)": 30},
        readability_metrics={
            "flesch_reading_ease": 65.5,
            "avg_sentence_length": 20.0
        },
        sentiment_analysis={
            "overall_sentiment": 0.7,
            "positive_ratio": 0.6
        }
    )
    
    assert report.total_documents == 5
    assert report.total_words == 10000
    assert len(report.top_keywords) == 2
    assert report.readability_metrics["flesch_reading_ease"] == 65.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])