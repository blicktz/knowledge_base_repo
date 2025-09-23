"""
Core modules for the Virtual Influencer Persona Agent
"""

# Import available modules with error handling
__all__ = []

try:
    from .persona_extractor import PersonaExtractor
    __all__.append("PersonaExtractor")
except ImportError as e:
    pass

try:
    from .knowledge_indexer import KnowledgeIndexer
    __all__.append("KnowledgeIndexer")
except ImportError as e:
    pass

try:
    from .statistical_analyzer import StatisticalAnalyzer
    __all__.append("StatisticalAnalyzer")
except ImportError as e:
    pass