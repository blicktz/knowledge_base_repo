"""
LLM Factory with Explicit Context Logging

This module provides a factory function for creating LLMs with explicit logging context,
eliminating the need for complex heuristics to determine what each LLM call is for.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from langchain_litellm import ChatLiteLLM

from ..config.settings import Settings
from ..agent.simple_llm_logger import SimpleLLMLoggingCallbackHandler
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_llm_with_logging(
    llm_config: Dict[str, Any],
    context_name: str,
    persona_id: str,
    settings: Settings
) -> ChatLiteLLM:
    """
    Create an LLM with explicit logging context - no guessing needed.
    
    Args:
        llm_config: LLM configuration (model, temperature, etc.)
        context_name: Explicit context name (e.g., "query_analysis", "hyde_generation", "synthesis")
        persona_id: The persona identifier
        settings: Application settings
        
    Returns:
        Configured ChatLiteLLM instance with logging callback
    """
    # Get the persona-specific logging directory
    logging_base_dir = settings.get_llm_logging_path(persona_id)
    
    # Ensure logging directory exists
    Path(logging_base_dir).mkdir(parents=True, exist_ok=True)
    
    # Create callback handler with explicit context
    callback_handler = SimpleLLMLoggingCallbackHandler(
        context_name=context_name,
        persona_id=persona_id,
        logging_base_dir=logging_base_dir,
        settings=settings
    )
    
    # Create LLM with callback
    llm = ChatLiteLLM(
        model=llm_config["llm_model"],
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 1000),
        callbacks=[callback_handler]
    )
    
    logger.info(f"Created LLM for {context_name} (persona: {persona_id}, model: {llm_config['llm_model']})")
    
    return llm


def create_agent_llm(persona_id: str, settings: Settings) -> ChatLiteLLM:
    """
    Create the main agent LLM for reasoning and tool calling.
    
    Args:
        persona_id: The persona identifier
        settings: Application settings
        
    Returns:
        Configured ChatLiteLLM instance for agent reasoning
    """
    llm_config = {
        "llm_model": settings.agent.query_analysis.llm_model,
        "temperature": settings.agent.query_analysis.temperature,
        "max_tokens": settings.agent.query_analysis.max_tokens
    }
    
    return create_llm_with_logging(
        llm_config=llm_config,
        context_name="agent_reasoning",
        persona_id=persona_id,
        settings=settings
    )


def create_query_analysis_llm(persona_id: str, settings: Settings) -> ChatLiteLLM:
    """
    Create LLM for query analysis preprocessing.
    
    Args:
        persona_id: The persona identifier
        settings: Application settings
        
    Returns:
        Configured ChatLiteLLM instance for query analysis
    """
    llm_config = {
        "llm_model": settings.agent.query_analysis.llm_model,
        "temperature": settings.agent.query_analysis.temperature,
        "max_tokens": settings.agent.query_analysis.max_tokens
    }
    
    return create_llm_with_logging(
        llm_config=llm_config,
        context_name="query_analysis",
        persona_id=persona_id,
        settings=settings
    )


def create_synthesis_llm(persona_id: str, settings: Settings) -> ChatLiteLLM:
    """
    Create LLM for final response synthesis.
    
    Args:
        persona_id: The persona identifier
        settings: Application settings
        
    Returns:
        Configured ChatLiteLLM instance for synthesis
    """
    llm_config = {
        "llm_model": settings.agent.synthesis.llm_model,
        "temperature": settings.agent.synthesis.temperature,
        "max_tokens": settings.agent.synthesis.max_tokens
    }
    
    return create_llm_with_logging(
        llm_config=llm_config,
        context_name="synthesis",
        persona_id=persona_id,
        settings=settings
    )


def create_hyde_llm(persona_id: str, settings: Settings) -> ChatLiteLLM:
    """
    Create LLM for HyDE hypothetical document generation.
    
    Args:
        persona_id: The persona identifier
        settings: Application settings
        
    Returns:
        Configured ChatLiteLLM instance for HyDE generation
    """
    hyde_config = settings.retrieval.hyde
    llm_config = {
        "llm_model": hyde_config.llm_model,
        "temperature": hyde_config.temperature,
        "max_tokens": hyde_config.max_tokens
    }
    
    return create_llm_with_logging(
        llm_config=llm_config,
        context_name="hyde_generation",
        persona_id=persona_id,
        settings=settings
    )