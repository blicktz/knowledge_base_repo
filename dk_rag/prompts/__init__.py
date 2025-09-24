"""
Prompt templates and utilities for Phase 2 Advanced Retrieval

This module contains prompt templates for:
- HyDE hypothesis generation
- Query transformation
- Retrieval optimization
"""

from .hyde_prompts import HYDE_PROMPTS, get_hyde_prompt
from .query_templates import QUERY_TEMPLATES, transform_query

__all__ = [
    'HYDE_PROMPTS',
    'get_hyde_prompt',
    'QUERY_TEMPLATES',
    'transform_query'
]