"""
HyDE (Hypothetical Document Embeddings) Prompt Templates

This module contains various prompt templates for generating hypothetical
answers that improve retrieval quality.
"""

from typing import Dict, Optional


# HyDE prompt templates for different query types
HYDE_PROMPTS = {
    "default": """You are an expert AI assistant tasked with generating a hypothetical document to be used for a semantic search query.

Your goal is NOT to answer the user's question in a conversational way. Instead, your goal is to generate a concise, information-rich document that contains the types of keywords, concepts, and technical terms likely to be found in a perfect, detailed answer.

## USER QUESTION ##
{query}

## INSTRUCTIONS ##
1.  **Be Concise:** The entire document must be between **150 and 250 words**.
2.  **Be Concept-Dense:** To make the document dense and useful for search, ensure it includes relevant information, examples, explanations, and important context.
3.  **Be Factual and Objective:** Write as an expert explaining a topic. Do not use personal opinions or any conversational language.
4.  **Omit Filler:** Do not include any introductions, pleasantries, or concluding summaries. Begin the response directly with the core information.

## HYPOTHETICAL DOCUMENT ##
""",

    "detailed_explanation": """You are an expert AI assistant tasked with generating a hypothetical document for a semantic search query.

Your goal is to generate a concise, information-rich document containing the keywords and concepts likely to be found in a comprehensive explanation.

## USER QUESTION ##
{query}

## INSTRUCTIONS ##
1.  **Be Concise:** The entire document must be between **150 and 250 words**.
2.  **Be Concept-Dense:** Ensure the document's content touches upon the following aspects:
    * Key concepts and their definitions.
    * Relevant examples and primary use cases.
    * Important considerations or best practices.
    * Common misconceptions to clarify.
3.  **Be Factual and Objective:** Explain the topic clearly and directly.
4.  **Omit Filler:** Do not include conversational introductions or summaries.

## HYPOTHETICAL EXPLANATION ##
""",

    "technical_response": """You are an expert AI assistant tasked with generating a hypothetical document for a technical search query.

Your goal is to generate a concise, information-rich document containing the keywords and terminology likely to be found in a specific technical answer.

## TECHNICAL QUESTION ##
{query}

## INSTRUCTIONS ##
1.  **Be Concise:** The entire document must be between **150 and 250 words**.
2.  **Be Concept-Dense:** Ensure the document's content touches upon the following aspects:
    * Core technical terminology and concepts.
    * Key implementation details or logic.
    * Practical examples or brief code snippets where applicable.
    * Common patterns and established approaches.
3.  **Be Factual and Objective:** Provide accurate, technical information.
4.  **Omit Filler:** Do not include conversational introductions or summaries.

## HYPOTHETICAL TECHNICAL DOCUMENT ##
""",

    "framework_focused": """You are an expert AI assistant tasked with generating a hypothetical document for a search query about a framework or methodology.

Your goal is to generate a concise, information-rich document containing the keywords and structured concepts likely to be found in a detailed framework breakdown.

## USER QUESTION ##
{query}

## INSTRUCTIONS ##
1.  **Be Concise:** The entire document must be between **150 and 250 words**.
2.  **Be Concept-Dense:** Ensure the document's content touches upon the following aspects:
    * The name and purpose of the main framework.
    * A summary of its step-by-step breakdown.
    * Its key principles and associated best practices.
    * A brief real-world application example.
3.  **Be Factual and Objective:** Describe the framework systematically.
4.  **Omit Filler:** Do not include conversational introductions or summaries.

## HYPOTHETICAL FRAMEWORK OUTLINE ##
""",

    "concept_explanation": """You are an expert AI assistant tasked with generating a hypothetical document for a search query that asks to explain a concept.

Your goal is to generate a concise, information-rich document containing the keywords and definitions likely to be found in a clear concept explanation.

## TOPIC/QUESTION ##
{query}

## INSTRUCTIONS ##
1.  **Be Concise:** The entire document must be between **150 and 250 words**.
2.  **Be Concept-Dense:** Ensure the document's content touches upon the following aspects:
    * A clear definition of the core concept.
    * A brief explanation of its mechanism or how it works.
    * A discussion of its importance and primary examples.
    * A brief comparison to related concepts.
3.  **Be Factual and Objective:** Explain the concept with clarity and accuracy.
4.  **Omit Filler:** Do not include conversational introductions or summaries.

## HYPOTHETICAL CONCEPT EXPLANATION ##
""",

    "problem_solution": """You are an expert AI assistant tasked with generating a hypothetical document for a search query about solving a problem.

Your goal is to generate a concise, information-rich document containing the keywords and steps likely to be found in a comprehensive solution.

## PROBLEM ##
{query}

## INSTRUCTIONS ##
1.  **Be Concise:** The entire document must be between **150 and 250 words**.
2.  **Be Concept-Dense:** Ensure the document's content touches upon the following aspects:
    * A concise summary of the problem.
    * The core of a systematic approach to the solution.
    * An outline of the main solution steps.
    * A mention of potential challenges or alternative approaches.
3.  **Be Factual and Objective:** Present the solution in a clear, step-by-step manner.
4.  **Omit Filler:** Do not include conversational introductions or summaries.

## HYPOTHETICAL SOLUTION OUTLINE ##
""",

    "factual_response": """You are an expert AI assistant tasked with generating a hypothetical document for a fact-based search query.

Your goal is to generate a concise, information-rich document containing the specific facts, data, and statistics likely to be found in a definitive answer.

## USER QUESTION ##
{query}

## INSTRUCTIONS ##
1.  **Be Concise:** The entire document must be between **150 and 250 words**.
2.  **Be Concept-Dense:** Ensure the document's content touches upon the following aspects:
    * The most critical verified facts and information.
    * Relevant statistics or data points.
    * Key context or background information.
3.  **Be Factual and Objective:** State information clearly and accurately.
4.  **Omit Filler:** Do not include conversational introductions or summaries.

## HYPOTHETICAL FACT SHEET ##
""",

    "tutorial_style": """You are an expert AI assistant tasked with generating a hypothetical document for a "how-to" search query.

Your goal is to generate a concise, information-rich document containing the keywords and structure likely to be found in a step-by-step tutorial.

## USER QUESTION ##
{query}

## INSTRUCTIONS ##
1.  **Be Concise:** The entire document must be between **150 and 250 words**.
2.  **Be Concept-Dense:** Ensure the document's content touches upon the following aspects:
    * An outline of the basic steps, from simple to more complex.
    * Inclusion of examples or practical tips for the main steps.
    * A summary of the key process or takeaways.
3.  **Be Factual and Objective:** Present the steps in a clear, instructional format.
4.  **Omit Filler:** Do not include conversational introductions or summaries.

## HYPOTHETICAL TUTORIAL ##
""",

    "comparison": """You are an expert AI assistant tasked with generating a hypothetical document for a comparison-based search query.

Your goal is to generate a concise, information-rich document containing the keywords and points of comparison likely to be found in a detailed analysis.

## USER QUESTION ##
{query}

## INSTRUCTIONS ##
1.  **Be Concise:** The entire document must be between **150 and 250 words**.
2.  **Be Concept-Dense:** Ensure the document's content touches upon the following aspects:
    * The key similarities and differences between the items.
    * The primary advantages and disadvantages of each.
    * The most common use cases for each option.
3.  **Be Factual and Objective:** Compare the items based on clear criteria.
4.  **Omit Filler:** Do not include conversational introductions or summaries.

## HYPOTHETICAL COMPARISON ##
""",

    "best_practices": """You are an expert AI assistant tasked with generating a hypothetical document for a search query about best practices.

Your goal is to generate a concise, information-rich document containing the keywords and strategies likely to be found in an expert guide.

## TOPIC ##
{query}

## INSTRUCTIONS ##
1.  **Be Concise:** The entire document must be between **150 and 250 words**.
2.  **Be Concept-Dense:** Ensure the document's content touches upon the following aspects:
    * A summary of industry-standard approaches.
    * A mention of common pitfalls to avoid.
    * An outline of proven strategies and techniques.
    * Tips for optimization or efficiency.
3.  **Be Factual and Objective:** Present the best practices as clear, expert recommendations.
4.  **Omit Filler:** Do not include conversational introductions or summaries.

## HYPOTHETICAL BEST PRACTICES GUIDE ##
"""
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