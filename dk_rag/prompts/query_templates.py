"""
Query transformation templates for advanced retrieval

This module contains templates for transforming queries to improve
retrieval effectiveness.
"""

from typing import List, Dict, Optional


QUERY_TEMPLATES = {
    "step_back": """What are the underlying concepts, principles, or fundamentals behind the following question?
Question: {query}
Fundamental concepts:""",
    
    "decomposition": """Break down the following complex question into simpler sub-questions that need to be answered:
Complex Question: {query}
Sub-questions:
1.""",
    
    "context_expansion": """What related topics, concepts, or areas should be considered when answering the following question?
Question: {query}
Related topics:""",
    
    "keyword_extraction": """Extract the key terms, concepts, and entities from the following question:
Question: {query}
Key terms:""",
    
    "reformulation": """Rephrase the following question in a clearer, more specific way:
Original Question: {query}
Clearer Question:""",
    
    "generalization": """What is the more general form or category of the following specific question?
Specific Question: {query}
General Question:""",
    
    "specification": """Make the following general question more specific and concrete:
General Question: {query}
Specific Question:"""
}


def transform_query(query: str, transformation_type: str = "step_back") -> str:
    """
    Transform a query using a specific transformation template.
    
    Args:
        query: Original query
        transformation_type: Type of transformation to apply
        
    Returns:
        Transformed query prompt
    """
    template = QUERY_TEMPLATES.get(transformation_type, QUERY_TEMPLATES["step_back"])
    return template.format(query=query)


def generate_multi_queries(query: str, num_variations: int = 3) -> List[str]:
    """
    Generate multiple query variations for improved retrieval coverage.
    
    Args:
        query: Original query
        num_variations: Number of variations to generate
        
    Returns:
        List of query variations
    """
    variations = [query]  # Include original
    
    # Add different transformations
    transformations = ["reformulation", "context_expansion", "keyword_extraction"]
    for i in range(min(num_variations - 1, len(transformations))):
        variations.append(transform_query(query, transformations[i]))
    
    return variations


def create_query_expansion_prompt(query: str) -> str:
    """
    Create a prompt for LLM-based query expansion.
    
    Args:
        query: Original query
        
    Returns:
        Query expansion prompt
    """
    return f"""Generate 3 different variations of the following query that would help find relevant information.
The variations should:
1. Use different words but maintain the same intent
2. Be more specific or more general as appropriate
3. Include related concepts or terminology

Original Query: {query}

Query Variations:
1."""


def create_answer_extraction_prompt(query: str, context: str) -> str:
    """
    Create a prompt for extracting answers from retrieved context.
    
    Args:
        query: User query
        context: Retrieved context
        
    Returns:
        Answer extraction prompt
    """
    return f"""Based on the following context, provide a direct and comprehensive answer to the question.
If the context doesn't contain enough information, indicate what's missing.

Context:
{context}

Question: {query}

Answer:"""