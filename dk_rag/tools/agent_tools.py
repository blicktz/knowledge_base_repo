"""
LangChain-native tools for persona agent system
Complete rewrite using @tool decorator pattern
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from ..config.settings import Settings
from ..core.knowledge_indexer import KnowledgeIndexer
from ..core.persona_manager import PersonaManager
from ..utils.logging import get_logger, get_component_logger
from ..utils.component_registry import get_component_registry

# Module-level logger for non-tool specific logging
logger = get_logger(__name__)

def _get_tool_logger(tool_name: str, persona_id: str):
    """Get a tool-specific component logger."""
    return get_component_logger(f"Tool:{tool_name}", persona_id)



@tool
def retrieve_mental_models(query: str, config: RunnableConfig = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant mental models using RAG pipeline.
    
    Args:
        query: The search query
        config: Runtime configuration containing persona_id and settings
    
    Returns:
        List of relevant mental model dictionaries
    """
    # Extract context from RunnableConfig
    persona_id = None
    settings = None
    rag_query = query  # Default to original query
    if config and "configurable" in config:
        persona_id = config["configurable"].get("persona_id")
        settings = config["configurable"].get("settings")
        # Use optimized rag_query if available
        rag_query = config["configurable"].get("rag_query", query)
    
    if not persona_id:
        raise ValueError("persona_id required in config")
        
    if not settings:
        from ..config.settings import Settings
        settings = Settings()
    
    # Use settings for retrieval parameters
    mm_config = settings.agent.tools.mental_models
    k = mm_config.get('k', 3)
    
    # Get tool-specific logger
    tool_logger = _get_tool_logger("MM", persona_id)
    tool_logger.info(f"Retrieving {k} mental models for persona: {persona_id}, rag_query: {rag_query}")
    
    try:
        # Get long-lived knowledge indexer from registry (server-optimized)
        component_registry = get_component_registry()
        knowledge_indexer = component_registry.get_knowledge_indexer(settings, persona_id)
        
        # Search mental models using optimized rag_query
        results = knowledge_indexer.search_mental_models(
            query=rag_query,
            persona_id=persona_id,
            k=k,
            use_reranking=True,
            return_scores=True
        )
        
        # Convert to serializable format
        formatted_results = []
        for i, result in enumerate(results):
            # Handle tuple format (item, score) when return_scores=True
            if isinstance(result, tuple):
                item, score = result
                score_value = score if score is not None else 0.0
            else:
                item = result
                score_value = 0.0
            
            formatted_results.append({
                'content': item.content if hasattr(item, 'content') else str(item),
                'score': score_value,
                'rank': i + 1,
                'metadata': item.metadata if hasattr(item, 'metadata') else {}
            })
        
        tool_logger.info(f"Retrieved {len(formatted_results)} mental models")
        return formatted_results
        
    except Exception as e:
        tool_logger.error(f"Mental models retrieval failed: {str(e)}")
        return []


@tool  
def retrieve_core_beliefs(query: str, config: RunnableConfig = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant core beliefs using RAG pipeline.
    
    Args:
        query: The search query
        config: Runtime configuration containing persona_id and settings
    
    Returns:
        List of relevant core belief dictionaries
    """
    # Extract context from RunnableConfig
    persona_id = None
    settings = None
    rag_query = query  # Default to original query
    if config and "configurable" in config:
        persona_id = config["configurable"].get("persona_id")
        settings = config["configurable"].get("settings")
        # Use optimized rag_query if available
        rag_query = config["configurable"].get("rag_query", query)
    
    if not persona_id:
        raise ValueError("persona_id required in config")
        
    if not settings:
        from ..config.settings import Settings
        settings = Settings()
        
    # Use settings for retrieval parameters
    cb_config = settings.agent.tools.core_beliefs
    k = cb_config.get('k', 5)
    
    # Get tool-specific logger
    tool_logger = _get_tool_logger("CB", persona_id)
    tool_logger.info(f"Retrieving {k} core beliefs for persona: {persona_id}, rag_query: {rag_query}")
    
    try:
        # Get long-lived knowledge indexer from registry (server-optimized)
        component_registry = get_component_registry()
        knowledge_indexer = component_registry.get_knowledge_indexer(settings, persona_id)
        
        # Search core beliefs using optimized rag_query
        results = knowledge_indexer.search_core_beliefs(
            query=rag_query,
            persona_id=persona_id,
            k=k,
            use_reranking=True,
            return_scores=True
        )
        
        # Convert to serializable format
        formatted_results = []
        for i, result in enumerate(results):
            # Handle tuple format (item, score) when return_scores=True
            if isinstance(result, tuple):
                item, score = result
                score_value = score if score is not None else 0.0
            else:
                item = result
                score_value = 0.0
            
            formatted_results.append({
                'content': item.content if hasattr(item, 'content') else str(item),
                'score': score_value,
                'rank': i + 1,
                'metadata': item.metadata if hasattr(item, 'metadata') else {}
            })
        
        tool_logger.info(f"Retrieved {len(formatted_results)} core beliefs")
        return formatted_results
        
    except Exception as e:
        tool_logger.error(f"Core beliefs retrieval failed: {str(e)}")
        return []


@tool
def retrieve_transcripts(query: str, config: RunnableConfig = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant transcript chunks using Phase 2 advanced pipeline.
    
    Args:
        query: The search query
        config: Runtime configuration containing persona_id and settings
    
    Returns:
        List of relevant transcript chunk dictionaries
    """
    # Extract context from RunnableConfig
    persona_id = None
    settings = None
    rag_query = query  # Default to original query
    if config and "configurable" in config:
        persona_id = config["configurable"].get("persona_id")
        settings = config["configurable"].get("settings")
        # Use optimized rag_query if available
        rag_query = config["configurable"].get("rag_query", query)
    
    if not persona_id:
        raise ValueError("persona_id required in config")
        
    if not settings:
        from ..config.settings import Settings
        settings = Settings()
        
    # Use settings for retrieval parameters
    ts_config = settings.agent.tools.transcripts
    k = ts_config.get('k', 5)
    retrieval_k = ts_config.get('retrieval_k', 25)
    
    # Get tool-specific logger
    tool_logger = _get_tool_logger("TS", persona_id)
    tool_logger.info(f"Retrieving {k} transcript chunks for persona: {persona_id}, rag_query: {rag_query}")
    
    try:
        # Get long-lived knowledge indexer from registry (server-optimized)
        component_registry = get_component_registry()
        knowledge_indexer = component_registry.get_knowledge_indexer(settings, persona_id)
        pipeline = knowledge_indexer.get_advanced_retrieval_pipeline(persona_id)
        
        # Use Phase 2 advanced pipeline with optimized rag_query
        results = pipeline.retrieve(
            query=rag_query,
            k=k,
            use_hyde=True,
            use_hybrid=True, 
            use_reranking=True,
            return_scores=True
        )
        
        # Convert to serializable format
        formatted_results = []
        for result in results:
            # Handle different tuple formats from advanced pipeline
            if isinstance(result, tuple) and len(result) == 2:
                # (Document, score) tuples from retrieve_with_scores()
                doc, score = result
                content = doc.page_content
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                score_value = score
            elif isinstance(result, tuple) and len(result) == 3:
                # (doc_id, score, doc_text) tuples from BM25 with return_docs=True
                doc_id, score, doc_text = result
                content = doc_text
                metadata = {'doc_id': doc_id}
                score_value = score
            elif hasattr(result, 'page_content'):
                # Document objects
                content = result.page_content
                metadata = result.metadata if hasattr(result, 'metadata') else {}
                score_value = metadata.get('similarity_score', None)
            else:
                # Dictionary or other format
                content = str(result)
                metadata = {}
                score_value = None
                
            formatted_results.append({
                'content': content,
                'metadata': metadata,
                'score': score_value
            })
        
        tool_logger.info(f"Retrieved {len(formatted_results)} transcript chunks")
        return formatted_results
        
    except Exception as e:
        tool_logger.error(f"Transcript retrieval failed: {str(e)}")
        return []


# Tool registry for easy access (query_analyzer and get_persona_data removed - they're now preprocessing steps)
PERSONA_TOOLS = [
    retrieve_mental_models,
    retrieve_core_beliefs,
    retrieve_transcripts
]


def get_tools_for_persona(persona_id: str, settings: Settings) -> List:
    """
    Get configured tools for a specific persona.
    
    Args:
        persona_id: The persona identifier  
        settings: Application settings
        
    Returns:
        List of LangChain tools configured for the persona
    """
    # Tools now handle context via RunnableConfig, so no wrapping needed
    return PERSONA_TOOLS