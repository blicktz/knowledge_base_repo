"""
LangChain-native tools for persona agent system
Complete rewrite using @tool decorator pattern
"""

import json
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage

from ..config.settings import Settings
from ..core.knowledge_indexer import KnowledgeIndexer
from ..core.persona_manager import PersonaManager
from ..data.discovery.artifact_discovery import ArtifactDiscovery
from ..utils.logging import get_logger

logger = get_logger(__name__)


@tool
def query_analyzer(query: str, persona_id: str = None, settings: Settings = None) -> Dict[str, Any]:
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
    
    if not settings:
        from ..config.settings import Settings
        settings = Settings()
    
    # Use config for LLM initialization (light task - fast model)
    config = settings.agent.query_analysis
    llm = ChatLiteLLM(
        model=config.llm_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens
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
        
        # Parse JSON response
        extracted = json.loads(response.content)
        
        # Validate required fields
        required_fields = ['core_task', 'rag_query', 'intent_type']
        for field in required_fields:
            if field not in extracted:
                extracted[field] = ""
        
        if 'provided_context' not in extracted:
            extracted['provided_context'] = ""
        
        logger.info(f"Query analysis completed: {extracted['core_task'][:50]}...")
        return extracted
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {str(e)}")
        return {
            "core_task": query,  # Fallback to original query
            "rag_query": query,
            "provided_context": "",
            "intent_type": "unknown"
        }
    except Exception as e:
        logger.error(f"Query analysis failed: {str(e)}")
        return {
            "core_task": query,
            "rag_query": query,
            "provided_context": "",
            "intent_type": "unknown"
        }


@tool
def get_persona_data(persona_id: str, settings: Settings = None) -> Dict[str, Any]:
    """
    Load and extract static persona data from latest artifact.
    
    Args:
        persona_id: The persona identifier
        settings: Application settings
    
    Returns:
        Dictionary with linguistic_style, communication_patterns, and metadata
    """
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
def retrieve_mental_models(query: str, persona_id: str, settings: Settings = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant mental models using RAG pipeline.
    
    Args:
        query: The search query
        persona_id: The persona identifier  
        settings: Application settings
    
    Returns:
        List of relevant mental model dictionaries
    """
    if not settings:
        from ..config.settings import Settings
        settings = Settings()
    
    # Use config for retrieval parameters
    config = settings.agent.tools.mental_models
    k = config.k
    
    logger.info(f"Retrieving {k} mental models for persona: {persona_id}")
    
    try:
        # Initialize knowledge indexer
        persona_manager = PersonaManager(settings)
        knowledge_indexer = KnowledgeIndexer(settings, persona_manager, persona_id)
        
        # Search mental models using existing tested pattern
        results, scores = knowledge_indexer.search_mental_models(
            query=query,
            k=k,
            use_reranking=config.use_reranking
        )
        
        # Convert to serializable format
        formatted_results = []
        for i, (result, score) in enumerate(zip(results, scores)):
            formatted_results.append({
                'content': result.get('content', str(result)),
                'score': score,
                'rank': i + 1,
                'metadata': result.get('metadata', {})
            })
        
        logger.info(f"Retrieved {len(formatted_results)} mental models")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Mental models retrieval failed: {str(e)}")
        return []


@tool  
def retrieve_core_beliefs(query: str, persona_id: str, settings: Settings = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant core beliefs using RAG pipeline.
    
    Args:
        query: The search query
        persona_id: The persona identifier
        settings: Application settings
    
    Returns:
        List of relevant core belief dictionaries
    """
    if not settings:
        from ..config.settings import Settings
        settings = Settings()
        
    # Use config for retrieval parameters
    config = settings.agent.tools.core_beliefs
    k = config.k
    
    logger.info(f"Retrieving {k} core beliefs for persona: {persona_id}")
    
    try:
        # Initialize knowledge indexer  
        persona_manager = PersonaManager(settings)
        knowledge_indexer = KnowledgeIndexer(settings, persona_manager, persona_id)
        
        # Search core beliefs using existing tested pattern
        results, scores = knowledge_indexer.search_core_beliefs(
            query=query,
            k=k,
            use_reranking=config.use_reranking
        )
        
        # Convert to serializable format
        formatted_results = []
        for i, (result, score) in enumerate(zip(results, scores)):
            formatted_results.append({
                'content': result.get('content', str(result)),
                'score': score,
                'rank': i + 1,
                'metadata': result.get('metadata', {})
            })
        
        logger.info(f"Retrieved {len(formatted_results)} core beliefs")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Core beliefs retrieval failed: {str(e)}")
        return []


@tool
def retrieve_transcripts(query: str, persona_id: str, settings: Settings = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant transcript chunks using Phase 2 advanced pipeline.
    
    Args:
        query: The search query
        persona_id: The persona identifier
        settings: Application settings
    
    Returns:
        List of relevant transcript chunk dictionaries
    """
    if not settings:
        from ..config.settings import Settings
        settings = Settings()
        
    # Use config for retrieval parameters
    config = settings.agent.tools.transcripts
    k = config.k
    retrieval_k = config.retrieval_k
    
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
        for doc in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': getattr(doc, 'score', None)
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
        List of configured LangChain tools
    """
    # Pre-configure tools with persona context
    def make_persona_tool(base_tool, persona_id, settings):
        """Create a persona-specific version of a tool"""
        def wrapper(*args, **kwargs):
            # Inject persona_id and settings if not provided
            kwargs.setdefault('persona_id', persona_id)
            kwargs.setdefault('settings', settings)
            return base_tool(*args, **kwargs)
        
        wrapper.__name__ = base_tool.__name__
        wrapper.__doc__ = base_tool.__doc__
        return wrapper
    
    # Create persona-specific tools
    configured_tools = []
    for tool in PERSONA_TOOLS:
        configured_tools.append(tool)
    
    return configured_tools