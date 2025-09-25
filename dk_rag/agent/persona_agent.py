"""
LangChain-native persona agent using create_react_agent
Complete rewrite leveraging LangChain's agent framework
"""

import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from ..config.settings import Settings
from ..tools.agent_tools import get_tools_for_persona
from ..utils.llm_utils import robust_json_loads
from ..utils.artifact_discovery import ArtifactDiscovery
# Will import llm factory functions inside methods to avoid circular imports
from ..utils.logging import get_logger

# Import robust JSON parsing library
from llm_output_parser import parse_json

logger = get_logger(__name__)


class LangChainPersonaAgent:
    """
    LangChain-native persona agent using ReAct pattern with conversation memory.
    
    This replaces the manual orchestration with LangChain's mature agent framework.
    """
    
    def __init__(self, persona_id: str, settings: Settings):
        self.persona_id = persona_id
        self.settings = settings
        self.logger = get_logger(f"{__name__}.{persona_id}")
        
        # Initialize LangChain LLM using factory (import here to avoid circular imports)
        from ..utils.llm_factory import create_agent_llm
        self.llm = create_agent_llm(persona_id, settings)
        
        # Get persona-specific tools
        self.tools = get_tools_for_persona(persona_id, settings)
        
        # Initialize memory for conversation context
        self.memory = MemorySaver()
        
        # Agent will be created dynamically per query with query analysis context
        self.agent_executor = None
        
        self.logger.info(f"LangChain PersonaAgent initialized for: {persona_id}")
        self.logger.info(f"Available tools: {[tool.name for tool in self.tools]}")
    
    def _initialize_llm(self) -> ChatLiteLLM:
        """Initialize LangChain-wrapped LLM"""
        # Use query_analysis config for agent reasoning (fast model)
        # The synthesis model will be used separately for final response generation
        model_config = self.settings.agent.query_analysis  # Use fast model for agent reasoning
        
        return ChatLiteLLM(
            model=model_config.llm_model,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens
        )
    
    def _create_agent(self, query_analysis: Optional[Dict[str, Any]] = None, persona_data: Optional[Dict[str, Any]] = None):
        """Create ReAct agent with tools and memory, optionally with query analysis and persona data context"""
        
        # Create system prompt that defines the persona behavior with query and persona context
        system_prompt = self._build_system_prompt(query_analysis, persona_data)
        
        # Create the ReAct agent with memory checkpointing and LLM logging
        agent_executor = create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.memory,
            prompt=system_prompt
        )
        
        return agent_executor
    
    def _build_system_prompt(self, query_analysis: Optional[Dict[str, Any]] = None, persona_data: Optional[Dict[str, Any]] = None) -> str:
        """Build system prompt for the persona agent with optional query analysis and persona data context"""
        
        persona_name = self.persona_id.replace('_', ' ').title()
        
        # Build query analysis section if available
        query_context = ""
        if query_analysis:
            core_task = query_analysis.get('core_task', '')
            intent_type = query_analysis.get('intent_type', '')
            provided_context = query_analysis.get('provided_context', '')
            
            query_context = f"""
USER QUERY ANALYSIS:
- Core Task: {core_task}
- Intent Type: {intent_type}
- User Context: {provided_context if provided_context else 'None provided'}

"""
        
        # Build persona data section if available
        persona_context = ""
        if persona_data:
            linguistic_style = persona_data.get('linguistic_style', {})
            
            # Extract key linguistic elements
            tone = linguistic_style.get('tone', '')
            catchphrases = linguistic_style.get('catchphrases', [])
            vocabulary = linguistic_style.get('vocabulary', [])
            comm_style = linguistic_style.get('communication_style', {})
            
            # Build persona context section
            persona_sections = []
            
            if tone:
                persona_sections.append(f"- Tone: {tone}")
            
            if catchphrases:
                catchphrase_list = ', '.join(catchphrases[:8])  # Limit to avoid overwhelming
                persona_sections.append(f"- Key Catchphrases: {catchphrase_list}")
            
            if vocabulary:
                vocab_list = ', '.join(vocabulary[:12])  # Limit to avoid overwhelming
                persona_sections.append(f"- Specialized Vocabulary: {vocab_list}")
            
            if comm_style:
                formality = comm_style.get('formality', '')
                directness = comm_style.get('directness', '')
                if formality:
                    persona_sections.append(f"- Communication Style: {formality}")
                if directness:
                    persona_sections.append(f"- Directness: {directness}")
            
            if persona_sections:
                persona_context = f"""
PERSONA LINGUISTIC PROFILE:
{chr(10).join(persona_sections)}

"""
        
        system_prompt = f"""You are a virtual AI persona of {persona_name}. Your goal is to respond authentically as {persona_name} would, using their tone, style, knowledge, and problem-solving approach.

CORE INSTRUCTIONS:
1. The user's query has been analyzed and you have access to your linguistic profile
2. Gather relevant context using the available retrieval tools:
   - retrieve_mental_models: Get relevant frameworks and methodologies you use
   - retrieve_core_beliefs: Get your foundational principles and beliefs
   - retrieve_transcripts: Get relevant examples of your actual words and ideas

3. Once you have gathered context from the tools, respond authentically as {persona_name}:
   - Use your specific tone, catchphrases, and vocabulary from your linguistic profile
   - Apply relevant mental models to structure your thinking
   - Ensure your reasoning aligns with your core beliefs
   - Ground your response in facts from your transcripts
   - NEVER break character or mention that you are an AI

4. Your response should feel like it's coming directly from {persona_name}, not an AI system.

{persona_context}
{query_context}
Remember: You are {persona_name}. Think, speak, and reason exactly as they would."""


        return system_prompt
    
    def _analyze_query(self, user_query: str) -> Dict[str, Any]:
        """Analyze user query and extract structured information as a preprocessing step."""
        self.logger.info(f"Analyzing query: {user_query[:100]}...")
        
        # Use the factory to create query analysis LLM with explicit context
        from ..utils.llm_factory import create_query_analysis_llm
        llm = create_query_analysis_llm(self.persona_id, self.settings)
        
        # Build analysis prompt
        prompt = f"""You are a query analysis specialist. Analyze the following user query and extract structured information.

User Query: "{user_query}"

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
            self.logger.info(f"DEBUG: Raw LLM response (first 500 chars): {response.content[:500] if response.content else 'None'}...")
            self.logger.info(f"DEBUG: Raw response total length: {len(response.content) if response.content else 0}")
            
            # Parse JSON response using robust parsing pattern
            try:
                # Try llm-output-parser first (handles markdown/mixed content better)
                extracted = parse_json(response.content)
                self.logger.debug(f"llm-output-parser successful")
            except Exception as parse_error:
                self.logger.debug(f"llm-output-parser failed: {str(parse_error)}, falling back to robust_json_loads")
                # Fallback to XML-aware extraction
                extracted = robust_json_loads(response.content, self.logger)
            
            # Validate required fields
            required_fields = ['core_task', 'rag_query', 'intent_type']
            for field in required_fields:
                if field not in extracted:
                    extracted[field] = ""
            
            if 'provided_context' not in extracted:
                extracted['provided_context'] = ""
            
            self.logger.info(f"Query analysis completed: {extracted['core_task'][:50]}...")
            return extracted
            
        except Exception as e:
            self.logger.error(f"Query analysis failed: {str(e)}")
            return {
                "core_task": user_query,  # Fallback to original query
                "rag_query": user_query,
                "provided_context": "",
                "intent_type": "unknown"
            }
    
    def _load_persona_data(self) -> Dict[str, Any]:
        """Load and extract static persona data from latest artifact (mirrors query analysis pattern)."""
        self.logger.info(f"Loading persona data for: {self.persona_id}")
        
        try:
            # Initialize artifact discovery
            artifact_discovery = ArtifactDiscovery(self.settings)
            
            # Auto-discover and load latest artifact
            json_path, artifact_info = artifact_discovery.get_latest_artifact_json(self.persona_id)
            
            self.logger.info(f"Loading from artifact: {artifact_info.file_path.name}")
            
            # Load and extract relevant data
            import json
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
            
            self.logger.info("Persona data loading completed")
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Persona data loading failed: {str(e)}")
            return {
                'linguistic_style': {},
                'communication_patterns': {},
                'persona_metadata': {'name': self.persona_id, 'description': '', 'extraction_timestamp': ''}
            }
    
    def process_query(self, user_query: str, session_id: Optional[str] = None) -> str:
        """
        Process a user query using the LangChain ReAct agent.
        
        Args:
            user_query: The user's input query
            session_id: Optional session ID for conversation continuity
            
        Returns:
            The agent's response as the persona
        """
        self.logger.info(f"Processing query: {user_query[:100]}...")
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Step 1: Always analyze the query first (preprocessing)
        query_analysis = self._analyze_query(user_query)
        
        # Step 2: Load persona data (static context)
        persona_data = self._load_persona_data()
        
        # Step 3: Create agent with query analysis and persona data context for better reasoning
        agent_executor = self._create_agent(query_analysis, persona_data)
        
        # Configure conversation thread with persona context, query analysis, and LLM logging
        config = {
            "configurable": {
                "thread_id": session_id,
                "persona_id": self.persona_id,
                "settings": self.settings,
                "query_analysis": query_analysis,  # Pass analysis to tools
                "rag_query": query_analysis.get('rag_query', user_query)  # Optimized search query
            },
            "max_concurrency": 1  # Execute tools sequentially to prevent model loading conflicts
        }
        
        try:
            # Invoke the agent with the user message
            messages = [HumanMessage(content=user_query)]
            
            # Stream the agent's execution for real-time feedback
            response_content = ""
            for step in agent_executor.stream(
                {"messages": messages}, 
                config=config, 
                stream_mode="values"
            ):
                # Get the latest message
                if step["messages"]:
                    latest_message = step["messages"][-1]
                    
                    # Log tool calls for debugging
                    if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                        for tool_call in latest_message.tool_calls:
                            self.logger.info(f"Tool called: {tool_call['name']}")
                    
                    # Capture the final AI response
                    if latest_message.type == 'ai' and latest_message.content:
                        response_content = latest_message.content
            
            if not response_content:
                # Fallback if no response captured
                final_state = agent_executor.invoke({"messages": messages}, config=config)
                if final_state["messages"]:
                    response_content = final_state["messages"][-1].content
                else:
                    response_content = f"I apologize, but I'm having trouble processing your request right now. Could you please rephrase your question?"
            
            self.logger.info("Query processing completed successfully")
            return response_content
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}", exc_info=True)
            
            # Provide a helpful error response in character
            error_response = f"I'm experiencing some technical difficulties right now. Could you please try rephrasing your question? I want to make sure I give you the best response possible."
            return error_response
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: The conversation session ID
            
        Returns:
            List of message dictionaries from the conversation
        """
        try:
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "persona_id": self.persona_id,
                    "settings": self.settings
                },
                "max_concurrency": 1  # Execute tools sequentially to prevent model loading conflicts
            }
            
            # Get checkpointed state
            checkpoint = self.memory.get(config)
            if checkpoint and "messages" in checkpoint:
                return [
                    {
                        "type": msg.type,
                        "content": msg.content,
                        "timestamp": getattr(msg, 'timestamp', None)
                    }
                    for msg in checkpoint["messages"]
                ]
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation history: {str(e)}")
            return []
    
    def clear_conversation(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: The conversation session ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "persona_id": self.persona_id,
                    "settings": self.settings
                },
                "max_concurrency": 1  # Execute tools sequentially to prevent model loading conflicts
            }
            
            # Clear the checkpoint
            self.memory.delete(config)
            self.logger.info(f"Cleared conversation history for session: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear conversation: {str(e)}")
            return False
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent configuration"""
        return {
            "persona_id": self.persona_id,
            "model": self.llm.model,
            "tools": [tool.name for tool in self.tools],
            "has_memory": True,
            "framework": "LangChain ReAct Agent",
            "dynamic_agent": True,  # Agent is created per query with query analysis context
        }


# Factory function for easy agent creation
def create_persona_agent(persona_id: str, settings: Settings = None) -> LangChainPersonaAgent:
    """
    Factory function to create a LangChain persona agent.
    
    Args:
        persona_id: The persona identifier
        settings: Application settings (will load defaults if not provided)
        
    Returns:
        Configured LangChain persona agent
    """
    if not settings:
        settings = Settings()
    
    return LangChainPersonaAgent(persona_id, settings)