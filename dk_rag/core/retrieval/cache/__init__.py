"""
Retrieval Cache System

Knowledge-aware caching system that provides separate cache namespaces
for different knowledge types while maintaining shared infrastructure.
"""

from .multi_knowledge_cache import MultiKnowledgeRetrievalCache

__all__ = [
    'MultiKnowledgeRetrievalCache'
]