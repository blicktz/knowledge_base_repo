"""
Validation utilities for persona constitution and data quality
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

from ..data.models.persona_constitution import PersonaConstitution
from ..config.settings import Settings


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def validate_persona(persona: PersonaConstitution, settings: Settings) -> List[str]:
    """
    Validate a persona constitution against quality requirements
    
    Args:
        persona: PersonaConstitution to validate
        settings: Settings with validation requirements
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    # Get validation thresholds
    quality_checks = settings.validation.quality_checks
    
    # Check mental models
    min_mental_models = quality_checks.get('min_mental_models', 3)
    if len(persona.mental_models) < min_mental_models:
        issues.append(f"Insufficient mental models: {len(persona.mental_models)} < {min_mental_models}")
    
    # Check core beliefs
    min_core_beliefs = quality_checks.get('min_core_beliefs', 5)
    if len(persona.core_beliefs) < min_core_beliefs:
        issues.append(f"Insufficient core beliefs: {len(persona.core_beliefs)} < {min_core_beliefs}")
    
    # Check catchphrases
    min_catchphrases = quality_checks.get('min_catchphrases', 3)
    if len(persona.linguistic_style.catchphrases) < min_catchphrases:
        issues.append(f"Insufficient catchphrases: {len(persona.linguistic_style.catchphrases)} < {min_catchphrases}")
    
    # Check vocabulary
    min_vocabulary = quality_checks.get('min_vocabulary_terms', 10)
    if len(persona.linguistic_style.vocabulary) < min_vocabulary:
        issues.append(f"Insufficient vocabulary terms: {len(persona.linguistic_style.vocabulary)} < {min_vocabulary}")
    
    # Check tone description
    if not persona.linguistic_style.tone or len(persona.linguistic_style.tone) < 10:
        issues.append("Missing or insufficient tone description")
    
    # Validate mental models
    for i, model in enumerate(persona.mental_models):
        if len(model.steps) < 2:
            issues.append(f"Mental model '{model.name}' has insufficient steps: {len(model.steps)}")
        if not model.description:
            issues.append(f"Mental model '{model.name}' missing description")
    
    # Validate core beliefs
    for i, belief in enumerate(persona.core_beliefs):
        if not belief.statement:
            issues.append(f"Core belief {i} missing statement")
        if not belief.category:
            issues.append(f"Core belief '{belief.statement[:30]}...' missing category")
    
    # Check statistical report
    if persona.statistical_report.total_words < 1000:
        issues.append(f"Insufficient source content: {persona.statistical_report.total_words} words")
    
    return issues


def validate_extraction_quality(persona: PersonaConstitution) -> Dict[str, float]:
    """
    Calculate quality scores for different aspects of extraction
    
    Args:
        persona: PersonaConstitution to evaluate
        
    Returns:
        Dictionary of quality scores (0-1 scale)
    """
    scores = {}
    
    # Linguistic style quality
    linguistic_score = 0.0
    if persona.linguistic_style.tone:
        linguistic_score += 0.25
    if len(persona.linguistic_style.catchphrases) >= 5:
        linguistic_score += 0.25
    if len(persona.linguistic_style.vocabulary) >= 20:
        linguistic_score += 0.25
    if persona.linguistic_style.communication_style:
        linguistic_score += 0.25
    scores['linguistic_style'] = min(1.0, linguistic_score)
    
    # Mental models quality
    mental_models_score = 0.0
    if len(persona.mental_models) >= 3:
        mental_models_score += 0.3
    if len(persona.mental_models) >= 5:
        mental_models_score += 0.2
    # Check average quality of models
    if persona.mental_models:
        avg_confidence = sum(m.confidence_score for m in persona.mental_models) / len(persona.mental_models)
        mental_models_score += avg_confidence * 0.5
    scores['mental_models'] = min(1.0, mental_models_score)
    
    # Core beliefs quality
    beliefs_score = 0.0
    if len(persona.core_beliefs) >= 5:
        beliefs_score += 0.3
    if len(persona.core_beliefs) >= 10:
        beliefs_score += 0.2
    # Check average quality of beliefs
    if persona.core_beliefs:
        avg_confidence = sum(b.confidence_score for b in persona.core_beliefs) / len(persona.core_beliefs)
        beliefs_score += avg_confidence * 0.5
    scores['core_beliefs'] = min(1.0, beliefs_score)
    
    # Statistical analysis quality
    stats_score = 0.0
    if persona.statistical_report.total_words >= 5000:
        stats_score += 0.3
    if persona.statistical_report.total_documents >= 3:
        stats_score += 0.2
    if persona.statistical_report.top_keywords:
        stats_score += 0.25
    if persona.statistical_report.top_collocations:
        stats_score += 0.25
    scores['statistical_analysis'] = min(1.0, stats_score)
    
    # Overall score
    scores['overall'] = sum(scores.values()) / len(scores)
    
    return scores


def validate_documents(documents: List[Dict[str, Any]]) -> List[str]:
    """
    Validate input documents for persona extraction
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        List of validation issues
    """
    issues = []
    
    if not documents:
        issues.append("No documents provided")
        return issues
    
    # Check for required fields
    for i, doc in enumerate(documents):
        if 'content' not in doc:
            issues.append(f"Document {i} missing 'content' field")
        elif not doc['content'].strip():
            issues.append(f"Document {i} has empty content")
        
        if 'source' not in doc:
            issues.append(f"Document {i} missing 'source' field")
    
    # Check total content
    total_words = sum(len(doc.get('content', '').split()) for doc in documents)
    if total_words < 1000:
        issues.append(f"Insufficient total content: {total_words} words (minimum 1000)")
    
    # Check for duplicates
    content_set = set()
    for i, doc in enumerate(documents):
        content = doc.get('content', '')[:100]  # Check first 100 chars
        if content in content_set:
            issues.append(f"Document {i} appears to be duplicate")
        content_set.add(content)
    
    return issues


def validate_config_file(config_path: str) -> List[str]:
    """
    Validate configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        List of validation issues
    """
    issues = []
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        issues.append(f"Configuration file not found: {config_path}")
        return issues
    
    try:
        # Try to load config
        from ..config.settings import Settings
        settings = Settings(config_path=config_path)
        
        # Validate configuration
        config_issues = settings.validate_configuration()
        issues.extend(config_issues)
        
    except Exception as e:
        issues.append(f"Failed to load configuration: {e}")
    
    return issues


def auto_fix_persona(persona: PersonaConstitution, settings: Settings) -> PersonaConstitution:
    """
    Attempt to auto-fix common persona issues
    
    Args:
        persona: PersonaConstitution to fix
        settings: Settings with validation requirements
        
    Returns:
        Fixed PersonaConstitution
    """
    if not settings.validation.auto_fix:
        return persona
    
    # Fix empty tone
    if not persona.linguistic_style.tone:
        persona.linguistic_style.tone = "Professional and informative"
    
    # Remove empty catchphrases
    persona.linguistic_style.catchphrases = [
        phrase for phrase in persona.linguistic_style.catchphrases 
        if phrase and len(phrase) > 2
    ]
    
    # Remove empty vocabulary
    persona.linguistic_style.vocabulary = [
        word for word in persona.linguistic_style.vocabulary
        if word and len(word) > 1
    ]
    
    # Fix mental models
    valid_models = []
    for model in persona.mental_models:
        if model.steps and len(model.steps) >= 2:
            if not model.description:
                model.description = f"A framework with {len(model.steps)} steps"
            valid_models.append(model)
    persona.mental_models = valid_models
    
    # Fix core beliefs
    valid_beliefs = []
    for belief in persona.core_beliefs:
        if belief.statement:
            if not belief.category:
                belief.category = "general"
            valid_beliefs.append(belief)
    persona.core_beliefs = valid_beliefs
    
    return persona


def validate_extraction_params(settings: Settings) -> List[str]:
    """
    Validate extraction parameters in settings
    
    Args:
        settings: Settings to validate
        
    Returns:
        List of validation issues
    """
    issues = []
    
    # Check LLM configuration
    if not settings.llm.config.get('api_key'):
        issues.append("LLM API key not configured")
    
    # Check chunking parameters
    chunk_config = settings.get_chunk_config()
    if chunk_config['size'] < 100:
        issues.append(f"Chunk size too small: {chunk_config['size']}")
    if chunk_config['overlap'] >= chunk_config['size']:
        issues.append("Chunk overlap cannot be >= chunk size")
    
    # Check extraction thresholds
    thresholds = settings.get_extraction_thresholds()
    mental_models_config = thresholds.get('mental_models', {})
    if mental_models_config.get('min_confidence', 0) > 1.0:
        issues.append("Mental models min_confidence cannot be > 1.0")
    
    return issues