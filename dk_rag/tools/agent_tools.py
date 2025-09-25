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
from ..utils.artifact_discovery import ArtifactDiscovery
from ..utils.logging import get_logger

logger = get_logger(__name__)


@tool
def get_persona_data(config: RunnableConfig = None) -> Dict[str, Any]:
    """
    Load and extract static persona data from latest artifact.
    
    Args:
        config: Runtime configuration containing persona_id and settings
    
    Returns:
        Dictionary with linguistic_style, communication_patterns, and metadata
    """
    # Extract context from RunnableConfig
    persona_id = None
    settings = None
    if config and "configurable" in config:
        persona_id = config["configurable"].get("persona_id")
        settings = config["configurable"].get("settings")
    
    if not persona_id:
        raise ValueError("persona_id required in config")
        
    logger.info(f"Loading persona data for: {persona_id}")
    
    try:
        # Initialize artifact discovery
        if not settings:
            from ..config.settings import Settings
            settings = Settings()
            
        artifact_discovery = ArtifactDiscovery(settings)
        
        # Auto-discover and load latest artifact
        json_path, artifact_info = artifact_discovery.get_latest_artifact_json(persona_id)
        
        logger.info(f"Loading from artifact: {artifact_info.file_path.name}")
        
        # Load and extract relevant data
        with open(json_path, 'r') as f:
            full_data = json.load(f)
        
        extracted_data = {
            'linguistic_style': full_data.get('linguistic_style', {}),
            'communication_patterns': full_data.get('communication_patterns', {}),
            'persona_metadata': {
                'name': full_data.get('name'),
                'description': full_data.get('description'),
                'extraction_timestamp': artifact_info.timestamp.isoformat()
            }
        }
        
        # Cleanup temp file if needed
        artifact_discovery.cleanup_temp_file(json_path)
        
        logger.info("Persona data extraction completed")
        return extracted_data
        
    except Exception as e:
        logger.error(f"Persona data loading failed: {str(e)}")
        return {
            'linguistic_style': {},
            'communication_patterns': {},
            'persona_metadata': {'name': persona_id, 'description': '', 'extraction_timestamp': ''}
        }


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
    
    logger.info(f"Retrieving {k} mental models for persona: {persona_id}, rag_query: {rag_query}")
    
    try:
        # Initialize knowledge indexer
        persona_manager = PersonaManager(settings) 
        knowledge_indexer = KnowledgeIndexer(settings, persona_manager, persona_id)
        
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
        
        logger.info(f"Retrieved {len(formatted_results)} mental models")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Mental models retrieval failed: {str(e)}")
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
    
    logger.info(f"Retrieving {k} core beliefs for persona: {persona_id}, rag_query: {rag_query}")
    
    try:
        # Initialize knowledge indexer
        persona_manager = PersonaManager(settings) 
        knowledge_indexer = KnowledgeIndexer(settings, persona_manager, persona_id)
        
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
        
        logger.info(f"Retrieved {len(formatted_results)} core beliefs")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Core beliefs retrieval failed: {str(e)}")
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
    
    logger.info(f"Retrieving {k} transcript chunks for persona: {persona_id}, rag_query: {rag_query}")
    
    try:
        # Initialize knowledge indexer and Phase 2 pipeline
        persona_manager = PersonaManager(settings) 
        knowledge_indexer = KnowledgeIndexer(settings, persona_manager, persona_id)
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
        
        logger.info(f"Retrieved {len(formatted_results)} transcript chunks")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Transcript retrieval failed: {str(e)}")
        return []


# Tool registry for easy access (query_analyzer removed - it's now a preprocessing step)
PERSONA_TOOLS = [
    get_persona_data, 
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