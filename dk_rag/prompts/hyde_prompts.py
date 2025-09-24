"""
HyDE (Hypothetical Document Embeddings) Prompt Templates

This module contains various prompt templates for generating hypothetical
answers that improve retrieval quality.
"""

from typing import Dict, Optional


# HyDE prompt templates for different query types
HYDE_PROMPTS = {
    "default": """Please provide a comprehensive and detailed answer to the following question. 
Include relevant information, examples, explanations, and any important context that would help someone 
fully understand the topic. Be thorough and informative.

Question: {query}

Detailed Answer:""",
    
    "detailed_explanation": """You are an expert providing a thorough explanation. Write a detailed, 
informative response to the following question. Include:
- Key concepts and definitions
- Relevant examples and use cases
- Important considerations and best practices
- Common misconceptions to avoid

Question: {query}

Comprehensive Explanation:""",
    
    "technical_response": """Provide a technical and detailed answer to the following question. 
Your response should be accurate, specific, and include:
- Technical terminology and concepts
- Implementation details where relevant
- Practical examples or code snippets if applicable
- Common patterns and approaches

Technical Question: {query}

Technical Answer:""",
    
    "framework_focused": """Answer the following question by providing a structured response that includes:
- The main framework or methodology involved
- Step-by-step breakdown of the approach
- Key principles and best practices
- Real-world application examples

Question: {query}

Structured Framework Answer:""",
    
    "concept_explanation": """Explain the following concept or topic in detail. Your explanation should:
- Define the core concept clearly
- Explain how it works
- Discuss why it's important
- Provide relevant examples
- Compare with related concepts if applicable

Topic/Question: {query}

Detailed Concept Explanation:""",
    
    "problem_solution": """For the following problem or question, provide a comprehensive solution that includes:
- Clear understanding of the problem
- Systematic approach to solving it
- Detailed solution steps
- Alternative approaches if applicable
- Potential challenges and how to address them

Problem: {query}

Comprehensive Solution:""",
    
    "factual_response": """Provide a factual, informative answer to the following question. 
Focus on accuracy and completeness. Include:
- Verified facts and information
- Relevant statistics or data if applicable
- Sources of information
- Context and background

Question: {query}

Factual Answer:""",
    
    "tutorial_style": """Create a tutorial-style answer for the following question. 
Structure your response as if you're teaching someone:
- Start with the basics
- Build up complexity gradually
- Include examples at each step
- Provide practical tips
- Summarize key takeaways

Question: {query}

Tutorial-Style Answer:""",
    
    "comparison": """Answer the following question by providing a detailed comparison. Include:
- Key similarities and differences
- Advantages and disadvantages of each option
- Use cases for each
- Recommendations based on different scenarios

Question: {query}

Detailed Comparison:""",
    
    "best_practices": """Provide a comprehensive answer focusing on best practices for the following topic:
- Industry-standard approaches
- Common pitfalls to avoid
- Proven strategies and techniques
- Tips for optimization and efficiency
- Real-world examples of successful implementation

Topic: {query}

Best Practices Guide:"""
}


def get_hyde_prompt(prompt_type: str = "default", query: str = "") -> str:
    """
    Get a HyDE prompt template and optionally format it with a query.
    
    Args:
        prompt_type: Type of prompt template to use
        query: Optional query to format the template with
        
    Returns:
        Prompt template or formatted prompt
    """
    template = HYDE_PROMPTS.get(prompt_type, HYDE_PROMPTS["default"])
    
    if query:
        return template.format(query=query)
    return template


def select_best_prompt(query: str) -> str:
    """
    Automatically select the best HyDE prompt based on query characteristics.
    
    Args:
        query: User query
        
    Returns:
        Best prompt template key
    """
    query_lower = query.lower()
    
    # Check for specific patterns in the query
    if any(word in query_lower for word in ['how to', 'tutorial', 'guide', 'step by step']):
        return "tutorial_style"
    elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus', 'better']):
        return "comparison"
    elif any(word in query_lower for word in ['best practice', 'optimize', 'improve', 'efficient']):
        return "best_practices"
    elif any(word in query_lower for word in ['problem', 'issue', 'error', 'fix', 'solve']):
        return "problem_solution"
    elif any(word in query_lower for word in ['what is', 'define', 'explain', 'meaning']):
        return "concept_explanation"
    elif any(word in query_lower for word in ['framework', 'methodology', 'approach', 'process']):
        return "framework_focused"
    elif any(word in query_lower for word in ['technical', 'implement', 'code', 'api']):
        return "technical_response"
    elif any(word in query_lower for word in ['fact', 'data', 'statistic', 'research']):
        return "factual_response"
    else:
        return "detailed_explanation"


def format_hyde_prompt(query: str, prompt_type: Optional[str] = None) -> str:
    """
    Format a HyDE prompt for the given query, automatically selecting
    the best template if not specified.
    
    Args:
        query: User query
        prompt_type: Optional specific prompt type to use
        
    Returns:
        Formatted HyDE prompt
    """
    if prompt_type is None:
        prompt_type = select_best_prompt(query)
    
    return get_hyde_prompt(prompt_type, query)