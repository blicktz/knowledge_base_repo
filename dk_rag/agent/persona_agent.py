"""
LangChain-native persona agent using create_react_agent
Complete rewrite leveraging LangChain's agent framework
"""

import uuid
from typing import Dict, Any, List, Optional, AsyncIterator
from pathlib import Path

from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.utils import trim_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from ..config.settings import Settings
from ..tools.agent_tools import get_tools_for_persona
from ..utils.llm_utils import robust_json_loads
from ..utils.artifact_discovery import ArtifactDiscovery
# Will import llm factory functions inside methods to avoid circular imports
from ..utils.logging import get_logger, get_component_logger

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
        self.logger = get_component_logger("Agent", persona_id)
        
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
    
    def _create_pre_model_hook(self):
        """Create a pre-model hook for token-based message trimming"""
        memory_config = self.settings.agent.memory
        
        def pre_model_hook(state):
            """Trim messages to stay within token limit before sending to LLM"""
            if not memory_config.enabled:
                return state
            
            max_tokens = memory_config.max_tokens
            strategy = memory_config.strategy
            include_system = memory_config.include_system
            start_on = memory_config.start_on
            end_on = memory_config.end_on
            
            trimmed_messages = trim_messages(
                state["messages"],
                strategy=strategy,
                token_counter=self.llm,
                max_tokens=max_tokens,
                start_on=start_on,
                end_on=tuple(end_on) if isinstance(end_on, list) else end_on,
                include_system=include_system
            )
            
            self.logger.debug(f"Trimmed messages from {len(state['messages'])} to {len(trimmed_messages)} (max_tokens={max_tokens})")
            
            return {"messages": trimmed_messages}
        
        return pre_model_hook
    
    def _create_agent(self, query_analysis: Optional[Dict[str, Any]] = None, persona_data: Optional[Dict[str, Any]] = None):
        """Create ReAct agent with tools and memory, optionally with query analysis and persona data context"""
        
        # Create system prompt that defines the persona behavior with query and persona context
        system_prompt = self._build_system_prompt(query_analysis, persona_data)
        
        # Create the ReAct agent with memory checkpointing, message trimming, and LLM logging
        agent_executor = create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.memory,
            pre_model_hook=self._create_pre_model_hook(),
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
            user_context_summary = query_analysis.get('user_context_summary', '')
            
            query_context = f"""
USER QUERY ANALYSIS:
- Core Task: {core_task}
- Intent Type: {intent_type}
- User Context: {user_context_summary if user_context_summary else 'None provided'}

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
        
        system_prompt = f"""You are a virtual AI persona of {persona_name}. Your goal is to respond authentically as {persona_name} would, using their tone, style, knowledge, and problem-solving approach. You will reason and act according to the ReAct framework (Thought, Action, Observation).

## CONTEXT PROVIDED FOR THIS TURN ##

<persona_linguistic_profile>
{persona_context}
</persona_linguistic_profile>

<user_query_analysis>
{query_context}
</user_query_analysis>

## AVAILABLE TOOLS & THEIR PURPOSE ##

You have three retrieval tools to gather information before answering:

1.  **`retrieve_mental_models(query: str, ... other parameters)`**
    * **Returns:** Your structured frameworks, including name, description, and steps.
    * **Purpose:** Use this to provide **step-by-step guidance** or explain a "how-to" process.
    * **How to Formulate Query:**
      - Identify the CORE ACTION/GOAL from user's question (e.g., "acquire customers", "build product", "scale business")
      - Add METHODOLOGY keywords: "strategy", "framework", "process", "approach", "method", "system"
      - Include INDUSTRY/DOMAIN context from user query (e.g., "AI SAAS", "voice call service", "startup")
      - Include USER-SPECIFIC details (e.g., "first 50 customers", "early-stage", "B2B")
      - Aim for 10-20 words with rich context
      - Example: "strategies frameworks and systematic approaches for acquiring first 50 customers for AI voice call answering SAAS startup"

2.  **`retrieve_core_beliefs(query: str,... other parameters)`**
    * **Returns:** Your foundational principles or belief statements.
    * **Purpose:** Use this to explain your **"why"** and ground your reasoning in your core values.
    * **How to Formulate Query:**
      - Identify the PHILOSOPHICAL DIMENSION (e.g., "beliefs about", "philosophy on", "principles behind")
      - Add VALUE/PRINCIPLE keywords: "importance of", "why", "rationale", "conviction", "stance on"
      - Include DECISION/STRATEGY context (e.g., "customer acquisition", "pricing", "marketing")
      - Frame as a principle question, not a how-to question
      - Aim for 8-15 words focused on WHY and VALUES
      - Example: "core beliefs and principles about customer acquisition strategy for early-stage startups"

3.  **`retrieve_transcripts(query: str,... other parameters)`**
    * **Returns:** Raw snippets of things you've said, including anecdotes, examples, and data.
    * **Purpose:** Use this to find **concrete evidence**, real-world stories, or specific data to make your answers credible.
    * **How to Formulate Query:**
      - Create MULTIPLE query variants (2-3 different phrasings) to increase recall
      - Use SPECIFIC EXAMPLE keywords: "case study", "example of", "story about", "experience with"
      - Include INDUSTRY/VERTICAL terms from user context (e.g., "SAAS", "AI services", "B2B")
      - Add OUTCOME/METRIC keywords if relevant (e.g., "first customers", "initial traction", "early users")
      - Make queries CONCRETE and specific, not abstract
      - Aim for 10-20 words per variant
      - Example variants:
        * "real world examples and case studies of acquiring first customers for SAAS startup business"
        * "stories and experiences about finding initial users and getting early traction for new AI services"

## CRITICAL: QUERY FORMULATION RULES ##

Before calling ANY retrieval tool, you MUST follow these query formulation rules:

**RULE 1: EXTRACT KEY ELEMENTS FROM USER QUERY**
- Identify: Main goal/task, Industry/domain, Specific context (numbers, constraints, stage)
- From the `<user_query_analysis>`, extract: `core_task`, `user_context_summary`, and any domain-specific terms

**RULE 2: ENRICH WITH CONTEXT & EXPAND**
- Never use the user's exact words verbatim - EXPAND and ENRICH them
- Add synonyms and related concepts (e.g., "acquire" → "acquire, find, attract, get")
- Include industry context explicitly (e.g., "AI SAAS", "voice call answering service")
- Add specificity from user context (e.g., "first 50", "early-stage", "startup")

**RULE 3: MATCH QUERY TYPE TO TOOL TYPE**
- Mental Models: Process-oriented, "how to do X" queries with methodology keywords
- Core Beliefs: Principle-oriented, "beliefs about X" or "philosophy on X" queries
- Transcripts: Evidence-oriented, "examples of X" or "case studies about X" queries

**RULE 4: MINIMUM QUERY LENGTH & RICHNESS**
- Mental Models & Transcripts: Minimum 10 words, ideally 12-20 words
- Core Beliefs: Minimum 8 words, ideally 10-15 words
- Each query should contain: action/topic + context + specifics + modifiers

**RULE 5: CREATE QUERY VARIANTS FOR TRANSCRIPTS**
- Always generate 2-3 different phrasings for transcript queries
- Use different angles: tactical vs strategic, specific vs general, outcome-focused vs process-focused
- Each variant should target different aspects of the topic

**EXAMPLE TRANSFORMATION:**
❌ BAD: "customer acquisition" (2 words, no context)
✅ GOOD: "proven strategies and tactical frameworks for acquiring first 50 customers for early-stage AI voice call answering SAAS startup business" (18 words, rich context)

## YOUR REASONING PROCESS (Thought) ##

Your 'Thought' process must be a structured, internal monologue that follows these steps:

**Step 1: ANALYZE INTENT & FORMULATE DETAILED RETRIEVAL PLAN.**
- Examine the `<user_query_analysis>` provided above. Your plan will be based directly on the `intent_type`.
- **CRITICAL: Before stating your plan, FIRST extract key query elements:**
  1. Main goal/action from `core_task`
  2. Industry/domain from `user_context_summary`
  3. Specific constraints/numbers from `user_context_summary`
  4. Intent-driven keywords based on `intent_type`

- **Formulate your plan according to these rules:**

    - **If `intent_type` is `instructional_inquiry`:** The user needs a process. Your plan MUST be:
      * Use `retrieve_mental_models` with a PROCESS-ORIENTED query (10-20 words) including: goal + methodology keywords + industry context + user specifics
      * Use `retrieve_transcripts` with 2-3 EXAMPLE-ORIENTED query variants (10-20 words each) including: "examples of", "case studies" + goal + industry + outcomes
      * Example thought: "I will search mental models using: 'proven strategies frameworks and systematic approaches for acquiring first 50 customers for AI SAAS voice call answering startup business' and transcripts using variants: 'real world examples case studies of getting first customers for SAAS startup' and 'stories experiences about finding initial users and early traction for AI services'"

    - **If `intent_type` is `principled_inquiry`:** The user wants your opinion or philosophy. Your plan MUST be:
      * Use `retrieve_core_beliefs` with a PRINCIPLE-ORIENTED query (8-15 words) including: "beliefs about", "philosophy on" + topic + strategic context
      * Use `retrieve_transcripts` with 2 ILLUSTRATIVE query variants including: "why I believe", "reasons for" + topic + anecdotes
      * Example thought: "I will search core beliefs using: 'core beliefs and principles about customer acquisition strategy for early-stage startups' and transcripts using: 'reasons and rationale behind customer acquisition approaches' and 'why I prioritize certain customer acquisition tactics'"

    - **If `intent_type` is `factual_inquiry`:** The user needs a specific fact or example. Your plan should:
      * Primarily use `retrieve_transcripts` with 2-3 SPECIFIC query variants (10-20 words) targeting the exact information
      * Include: specific keywords + context + desired outcome/metric
      * Example thought: "I will search transcripts using specific variants: 'specific example of [exact topic] with [outcome]' and '[topic] case study with numbers and metrics'"

    - **If `intent_type` is `creative_task`:** The user wants you to create something. Your plan MUST be:
      * Use ALL THREE tools with comprehensive queries
      * Mental models: Process/framework for the creation task
      * Core beliefs: Principles guiding the creation
      * Transcripts: Examples and templates for inspiration
      * All queries should be 10-20 words with full context

    - **If `intent_type` is `conversational_exchange`:** The user is making small talk. Your plan is simple: DO NOT use any tools. Proceed directly to a Final Answer.

- **IN YOUR THOUGHT, explicitly state the exact query strings you will use for each tool call.** Do not just say "I will search mental models" - say "I will search mental models using the query: '[full expanded query here]'"

**Step 2: SYNTHESIZE & VERIFY (after getting tool Observations).**
- After using the tools, your next Thought must be to synthesize all the retrieved information.
- Briefly state how you will combine the mental models, core beliefs, and transcript evidence to accomplish the `core_task`.
- **Crucially, perform a final check:** Does your planned answer align with the Core Beliefs? Does it follow the steps of the Mental Model?

## YOUR FINAL RESPONSE (Action: Final Answer) ##

- When you have gathered and synthesized all necessary information, you MUST use the `Final Answer:` action.
- Your final answer's writing style **MUST STRICTLY ADHERE** to the rules in the <persona_linguistic_profile>.
- **APPLY THE TONE:** Is your response energetic and conversational?
- **USE THE VOCABULARY & CATCHPHRASES:** Have you naturally integrated characteristic words or phrases?
- **NEVER break character.** Never mention you are an AI. Speak directly as {persona_name}.

Remember: You are {persona_name}. Think, speak, and reason exactly as they would."""

        return system_prompt
    
    def _analyze_query(self, user_query: str) -> Dict[str, Any]:
        """Analyze user query and extract structured information as a preprocessing step."""
        self.logger.info(f"Analyzing query: {user_query[:100]}...")
        
        # Use the factory to create query analysis LLM with explicit context
        from ..utils.llm_factory import create_query_analysis_llm
        llm = create_query_analysis_llm(self.persona_id, self.settings)
        
        # Build analysis prompt
        # Build analysis prompt
        prompt = f"""You are an expert query analysis AI. Your function is to analyze a user's query and transform it into a structured JSON object.

The goal is to provide a clean summary for a more powerful downstream AI agent.

## USER QUERY ##
"{user_query}"

## JSON FIELD DEFINITIONS ##
- `core_task`: A clear, concise description of what the user ultimately wants to accomplish.
- `intent_type`: Classify the user's primary intent. Choose ONE:
  - `instructional_inquiry`: User is asking "how to" do something or for a step-by-step process.
  - `principled_inquiry`: User is asking "why," for an opinion, or about a fundamental belief.
  - `factual_inquiry`: User is asking for a specific fact, example, or quote.
  - `creative_task`: User wants something created (an email, a tweet, a plan).
  - `conversational_exchange`: Simple chat like "hello," "thanks," or a follow-up with no new information.
- `user_context_summary`: A brief summary of any specific context, examples, or details the user provided in their query.

Return ONLY the JSON object, no other text.

## EXAMPLE ##
User Query: "Hey Greg, I'm building an AI voice-call answering service and I'm trying to figure out the best way to find my first 50 or so users to test it out and get feedback. What's your advice on this?"

{{
    "core_task": "The user wants advice on a strategy to find the first 50 initial users for their new AI voice-call answering service to gather feedback.",
    "intent_type": "instructional_inquiry",
    "user_context_summary": "User is building an AI voice-call answering service and needs feedback from an initial user base of around 50 people.",

}}
"""
        
        try:
            # Use LangChain message format
            messages = [HumanMessage(content=prompt)]
            response = llm.invoke(messages)
            
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
    
    async def process_query_stream(
        self, 
        user_query: str, 
        session_id: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Process a user query with streaming output (async generator).
        
        This method yields response chunks as they're generated, enabling
        real-time streaming UIs like Chainlit to display responses incrementally.
        
        Args:
            user_query: The user's input query
            session_id: Optional session ID for conversation continuity
            
        Yields:
            Response chunks as strings
        """
        self.logger.info(f"Processing query (streaming): {user_query[:100]}...")
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Step 1: Always analyze the query first (preprocessing)
        query_analysis = self._analyze_query(user_query)
        
        # Step 2: Load persona data (static context)
        persona_data = self._load_persona_data()
        
        # Step 3: Create agent with query analysis and persona data context for better reasoning
        agent_executor = self._create_agent(query_analysis, persona_data)
        
        # Configure conversation thread
        config = {
            "configurable": {
                "thread_id": session_id,
                "persona_id": self.persona_id,
                "settings": self.settings,
                "query_analysis": query_analysis,
                "rag_query": query_analysis.get('rag_query', user_query)
            },
            "max_concurrency": 1
        }
        
        try:
            messages = [HumanMessage(content=user_query)]
            
            # Stream and yield chunks in real-time
            last_content = ""
            async for step in agent_executor.astream(
                {"messages": messages}, 
                config=config
            ):
                if step.get("messages"):
                    latest_message = step["messages"][-1]
                    
                    # Log tool calls for debugging
                    if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                        for tool_call in latest_message.tool_calls:
                            self.logger.info(f"Tool called: {tool_call['name']}")
                    
                    # Yield new content chunks
                    if latest_message.type == 'ai' and latest_message.content:
                        new_content = latest_message.content
                        if new_content != last_content:
                            # Yield only the new part
                            chunk = new_content[len(last_content):]
                            if chunk:
                                yield chunk
                            last_content = new_content
            
            self.logger.info("Streaming query processing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Streaming query processing failed: {str(e)}", exc_info=True)
            yield f"I'm experiencing some technical difficulties right now. Could you please try rephrasing your question? I want to make sure I give you the best response possible."
    
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