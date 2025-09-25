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
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage

from ..config.settings import Settings
from ..core.knowledge_indexer import KnowledgeIndexer
from ..core.persona_manager import PersonaManager
from ..utils.artifact_discovery import ArtifactDiscovery
from ..utils.logging import get_logger
from ..utils.llm_utils import robust_json_loads


# Import robust JSON parsing library
from llm_output_parser import parse_json

logger = get_logger(__name__)


@tool
def query_analyzer(query: str, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    Analyze user queries to extract core tasks and generate RAG queries.
    
    Args:
        query: The user query to analyze
        persona_id: ID of the persona (for context)
        settings: Application settings
    
    Returns:
        Dictionary with core_task, rag_query, provided_context, and intent_type
    """
    logger.info(f"Analyzing query: {query[:100]}...")
    
    # Extract context from RunnableConfig
    settings = None
    if config and "configurable" in config:
        settings = config["configurable"].get("settings")
    
    if not settings:
        from ..config.settings import Settings
        settings = Settings()
    
    # Use settings for LLM initialization (light task - fast model)
    llm_config = settings.agent.query_analysis
    llm = ChatLiteLLM(
        model=llm_config.llm_model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens
    )
    
    # Build analysis prompt
    prompt = f"""You are a query analysis specialist. Analyze the following user query and extract structured information.

User Query: "{query}"

Extract and return a JSON object with the following fields:

1. "core_task": A clear, concise description of what the user wants to accomplish (1-2 sentences)
2. "rag_query": An optimized search query for RAG retrieval that captures the key concepts and terms
3. "provided_context": Any specific context, examples, or details the user provided
4. "intent_type": Classify the intent as one of:
   - "question" (asking for information)
   - "task" (requesting an action or creation)
   - "analysis" (requesting analysis or evaluation)
   - "advice" (seeking recommendations or guidance)

Return ONLY the JSON object, no other text.

Example response:
{{
    "core_task": "Create a sales email for a new product launch",
    "rag_query": "sales email product launch marketing copywriting persuasion",
    "provided_context": "New SaaS product for small businesses",
    "intent_type": "task"
}}

Now analyze the query and return the JSON:"""
    
    try:
        # Use LangChain message format
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        
        # Log raw response for debugging
        logger.info(f"DEBUG: Raw LLM response (first 500 chars): {response.content[:500] if response.content else 'None'}...")
        logger.info(f"DEBUG: Raw response total length: {len(response.content) if response.content else 0}")
        
        # Parse JSON response using robust parsing pattern
        try:
            # Try llm-output-parser first (handles markdown/mixed content better)
            extracted = parse_json(response.content)
            logger.debug(f"llm-output-parser successful")
        except Exception as parse_error:
            logger.debug(f"llm-output-parser failed: {str(parse_error)}, falling back to robust_json_loads")
            # Fallback to XML-aware extraction
            extracted = robust_json_loads(response.content, logger)
        
        # Validate required fields
        required_fields = ['core_task', 'rag_query', 'intent_type']
        for field in required_fields:
            if field not in extracted:
                extracted[field] = ""
        
        if 'provided_context' not in extracted:
            extracted['provided_context'] = ""
        
        logger.info(f"Query analysis completed: {extracted['core_task'][:50]}...")
        return extracted
        
    except Exception as e:
        logger.error(f"Query analysis failed: {str(e)}")
        return {
            "core_task": query,  # Fallback to original query
            "rag_query": query,
            "provided_context": "",
            "intent_type": "unknown"
        }


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
    if config and "configurable" in config:
        persona_id = config["configurable"].get("persona_id")
        settings = config["configurable"].get("settings")
    
    if not persona_id:
        raise ValueError("persona_id required in config")
        
    if not settings:
        from ..config.settings import Settings
        settings = Settings()
    
    # Use settings for retrieval parameters
    mm_config = settings.agent.tools.mental_models
    k = mm_config.get('k', 3)
    
    logger.info(f"Retrieving {k} mental models for persona: {persona_id}")
    
    try:
        # Initialize knowledge indexer
        persona_manager = PersonaManager(settings) 
        knowledge_indexer = KnowledgeIndexer(settings, persona_manager, persona_id)
        
        # Search mental models using KnowledgeIndexer method
        results = knowledge_indexer.search_mental_models(
            query=query,
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
    if config and "configurable" in config:
        persona_id = config["configurable"].get("persona_id")
        settings = config["configurable"].get("settings")
    
    if not persona_id:
        raise ValueError("persona_id required in config")
        
    if not settings:
        from ..config.settings import Settings
        settings = Settings()
        
    # Use settings for retrieval parameters
    cb_config = settings.agent.tools.core_beliefs
    k = cb_config.get('k', 5)
    
    logger.info(f"Retrieving {k} core beliefs for persona: {persona_id}")
    
    try:
        # Initialize knowledge indexer
        persona_manager = PersonaManager(settings) 
        knowledge_indexer = KnowledgeIndexer(settings, persona_manager, persona_id)
        
        # Search core beliefs using KnowledgeIndexer method
        results = knowledge_indexer.search_core_beliefs(
            query=query,
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
    if config and "configurable" in config:
        persona_id = config["configurable"].get("persona_id")
        settings = config["configurable"].get("settings")
    
    if not persona_id:
        raise ValueError("persona_id required in config")
        
    if not settings:
        from ..config.settings import Settings
        settings = Settings()
        
    # Use settings for retrieval parameters
    ts_config = settings.agent.tools.transcripts
    k = ts_config.get('k', 5)
    retrieval_k = ts_config.get('retrieval_k', 25)
    
    logger.info(f"Retrieving {k} transcript chunks for persona: {persona_id}")
    
    try:
        # Initialize knowledge indexer and Phase 2 pipeline
        persona_manager = PersonaManager(settings) 
        knowledge_indexer = KnowledgeIndexer(settings, persona_manager, persona_id)
        pipeline = knowledge_indexer.get_advanced_retrieval_pipeline(persona_id)
        
        # Use Phase 2 advanced pipeline with all enhancements
        results = pipeline.retrieve(
            query=query,
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


# Tool registry for easy access
PERSONA_TOOLS = [
    query_analyzer,
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