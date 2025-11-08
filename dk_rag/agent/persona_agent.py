"""
LangChain-native persona agent using create_react_agent
Complete rewrite leveraging LangChain's agent framework
"""

import uuid
from typing import Dict, Any, List, Optional, AsyncIterator, Union
from pathlib import Path
from dataclasses import dataclass

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


@dataclass
class StreamEvent:
    """Represents a structured streaming event for UI organization"""
    event_type: str  # "thinking", "tool_call", "tool_result", "final_answer"
    content: str
    metadata: Optional[Dict[str, Any]] = None


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
    
    def _validate_tool_calls(self, message) -> bool:
        """
        Validate tool calls to prevent concatenated or malformed tool names.
        Returns True if valid, False if invalid.
        """
        if not hasattr(message, 'tool_calls') or not message.tool_calls:
            return True  # No tool calls to validate
        
        valid_tool_names = {'retrieve_mental_models', 'retrieve_core_beliefs', 'retrieve_transcripts'}
        
        for tool_call in message.tool_calls:
            tool_name = tool_call.get('name', '')
            
            # Check for concatenated tool names (e.g., "retrieve_transcriptsretrieve_transcripts")
            if tool_name not in valid_tool_names:
                self.logger.error(f"INVALID TOOL CALL DETECTED: '{tool_name}'")
                self.logger.error(f"Valid tools are: {valid_tool_names}")
                
                # Check if it's a concatenation
                for valid_name in valid_tool_names:
                    if valid_name in tool_name and tool_name != valid_name:
                        self.logger.error(f"Tool name appears to be concatenated. Contains '{valid_name}' but full name is '{tool_name}'")
                
                return False
            
            # Check tool call arguments
            if 'args' not in tool_call:
                self.logger.error(f"Tool call missing 'args': {tool_call}")
                return False
            
            # Validate query parameter exists
            args = tool_call.get('args', {})
            if 'query' not in args:
                self.logger.error(f"Tool call missing 'query' parameter: {tool_call}")
                return False
        
        # Log successful validation
        tool_names = [tc.get('name') for tc in message.tool_calls]
        self.logger.info(f"Tool calls validated successfully: {tool_names}")
        return True
    
    def _check_tool_usage_policy(self, messages_history: List, intent_type: str) -> bool:
        """
        Check if tools were used when required based on intent type.
        Returns True if policy is satisfied, False if violated.
        """
        # conversational_exchange is the only intent that doesn't require tools
        if intent_type == 'conversational_exchange':
            return True  # No tools required
        
        # For all other intents, check if at least one tool was called
        tools_called = []
        for msg in messages_history:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get('name', '')
                    if tool_name in {'retrieve_mental_models', 'retrieve_core_beliefs', 'retrieve_transcripts'}:
                        tools_called.append(tool_name)
        
        if not tools_called:
            self.logger.error(f"TOOL USAGE POLICY VIOLATION: No tools were called for intent_type='{intent_type}'")
            self.logger.error(f"Expected at least one tool call for non-conversational queries")
            return False
        
        self.logger.info(f"Tool usage policy satisfied: {len(tools_called)} tool(s) called for intent_type='{intent_type}'")
        self.logger.info(f"Tools used: {tools_called}")
        return True
    
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
        
        # Determine if tools are mandatory based on intent type
        intent_type = query_analysis.get('intent_type', '') if query_analysis else ''
        
        if intent_type != 'conversational_exchange':
            tool_enforcement = """
## ⚠️ MANDATORY TOOL USAGE FOR THIS QUERY ⚠️

**CRITICAL: This query requires tool usage.**

You MUST call at least ONE tool before providing your final answer. The intent type of this query requires you to retrieve information from your knowledge base.

**DO NOT skip tools.** Even if you think you know the answer from earlier in the conversation, you MUST retrieve fresh information from the tools to ensure authenticity, accuracy, and to ground your response in your actual experience.

**If you attempt to provide a Final Answer without calling any tools first, the system will reject your response as a policy violation.**

Follow the tool calling sequence specified in your reasoning process below based on the intent type.

"""
        else:
            tool_enforcement = """
## TOOL USAGE: OPTIONAL FOR THIS QUERY

This query appears to be simple conversational exchange (greetings like "hi/hello/thanks" or brief acknowledgments like "ok/got it"). 

You may skip tools and respond directly ONLY if this is genuinely just small talk with no substantive content.

"""
        
        # Build query analysis section if available
        query_context = ""
        if query_analysis:
            core_task = query_analysis.get('core_task', '')
            intent_type_display = query_analysis.get('intent_type', '')
            user_context_summary = query_analysis.get('user_context_summary', '')
            
            query_context = f"""
USER QUERY ANALYSIS:
- Core Task: {core_task}
- Intent Type: {intent_type_display}
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

## LANGUAGE HANDLING ##
CRITICAL: Detect the primary language of the input and produce your ENTIRE output EXCLUSIVELY in that single language.

**STRICT RULES:**
1. **SINGLE LANGUAGE ONLY**: Use ONLY the detected input language. NO mixing of languages.
2. **NO TRANSLATIONS**: Do not provide translations, explanations, or parenthetical notes in other languages.
3. **NO ROMANIZATION**: If input is Chinese, do NOT include Pinyin romanization. If input is English, do NOT include IPA or phonetics.
4. **ASSUME FLUENT READER**: The reader is a native/fluent speaker of the input language and does not need assistance from other languages.

**Examples:**
- ✅ CORRECT for Chinese input: "恢复秩序和繁荣"
- ❌ WRONG for Chinese input: "恢复秩序 (huīfù zhìxù - restore order)"
- ✅ CORRECT for English input: "Restore order and prosperity"
- ❌ WRONG for English input: "Restore order (恢复秩序)"

Your output language must match the input language, NOT the language this prompt is written in.

{tool_enforcement}

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


## CRITICAL TOOL USAGE RULES ##

**YOU MUST FOLLOW THESE RULES EXACTLY:**

1. **ONE TOOL AT A TIME**: Call ONLY ONE tool per Action. NEVER call multiple tools simultaneously.
2. **WAIT FOR OBSERVATION**: After calling a tool, you MUST wait for the Observation before taking another Action.
3. **EXACT TOOL NAMES**: Use EXACTLY these tool names:
   - `retrieve_mental_models` (NOT retrieve_mental_modelsretrieve_mental_models)
   - `retrieve_core_beliefs` (NOT retrieve_core_beliefsretrieve_core_beliefs)
   - `retrieve_transcripts` (NOT retrieve_transcriptsretrieve_transcripts)
4. **PROPER FORMAT**: Each tool call must be in the correct format with the tool name as a single, non-repeated string.
5. **NO CONCATENATION**: Do NOT concatenate or repeat tool names. Each tool name appears ONCE per call.

If you violate these rules, the system will reject your tool call with an error.


## YOUR REASONING PROCESS (Thought) ##

Your 'Thought' process must be a structured, internal monologue that follows these steps:

**Step 1: ANALYZE INTENT & FORMULATE RETRIEVAL PLAN.**
- Examine the `<user_query_analysis>` provided above. Your plan will be based directly on the `intent_type`.
- **IMPORTANT**: You will call tools ONE AT A TIME, in SEQUENCE. Not all at once.
- **Extract key query elements:**
  1. Main goal/action from `core_task`
  2. Industry/domain from `user_context_summary`
  3. Specific constraints/numbers from `user_context_summary`
  4. Intent-driven keywords based on `intent_type`

- **Formulate your plan according to these rules (call tools sequentially):**

    - **If `intent_type` is `instructional_inquiry`:** [TOOLS MANDATORY] The user needs a process.
      * ⚠️ REQUIRED: You MUST call tools for this intent type
      * FIRST, call `retrieve_mental_models` with a PROCESS-ORIENTED query - MANDATORY
      * THEN, after receiving results, call `retrieve_transcripts` for examples - MANDATORY
      * Call tools ONE AT A TIME in this order
      * Example thought: "I will FIRST call retrieve_mental_models with: 'proven strategies frameworks for acquiring first 50 customers for AI SAAS startup', then WAIT for results, then call retrieve_transcripts for examples"

    - **If `intent_type` is `principled_inquiry`:** [TOOLS MANDATORY] The user wants your opinion or philosophy.
      * ⚠️ REQUIRED: You MUST call tools for this intent type
      * FIRST, call `retrieve_core_beliefs` with a PRINCIPLE-ORIENTED query - MANDATORY
      * THEN, after receiving results, call `retrieve_transcripts` for illustrative stories - MANDATORY
      * Call tools ONE AT A TIME in this order
      * Example thought: "I will FIRST call retrieve_core_beliefs with: 'core beliefs about customer acquisition for startups', then WAIT for results, then call retrieve_transcripts"

    - **If `intent_type` is `factual_inquiry`:** [TOOLS MANDATORY] The user needs a specific fact or example.
      * ⚠️ REQUIRED: You MUST call tools for this intent type
      * FIRST, call `retrieve_transcripts` with a SPECIFIC query - MANDATORY
      * If more context needed, THEN call other tools after receiving first results
      * Call tools ONE AT A TIME
      * Example thought: "I will FIRST call retrieve_transcripts with: 'specific proven lead magnet examples that worked with metrics and results'"

    - **If `intent_type` is `creative_task`:** [TOOLS MANDATORY - ALL THREE] The user wants you to create something (email, copy, plan, etc.).
      * ⚠️ REQUIRED: You MUST call ALL THREE tools for this intent type
      * FIRST, call `retrieve_mental_models` for framework - MANDATORY
      * THEN, call `retrieve_core_beliefs` for principles - MANDATORY
      * FINALLY, call `retrieve_transcripts` for examples - MANDATORY
      * Call tools ONE AT A TIME in this specific order
      * DO NOT SKIP - All three tools are required for authentic creative output
      * Example thought: "I will FIRST call retrieve_mental_models for email writing frameworks, THEN retrieve_core_beliefs for my philosophy on direct response copy, FINALLY retrieve_transcripts for proven email examples"

    - **If `intent_type` is `conversational_exchange`:** [TOOLS OPTIONAL] The user is making simple small talk.
      * This is the ONLY case where you may skip tools
      * Examples: "hi", "hello", "thanks", "ok", "got it"
      * If there is ANY substantive question or request, treat it as a different intent type and use tools

- **IN YOUR THOUGHT, explicitly state the exact query strings you will use for each tool call.** Do not just say "I will search mental models" - say "I will search mental models using the query: '[full expanded query here]'"

**Step 2: SYNTHESIZE & VERIFY (after getting tool Observations).**
- After using the tools, your next Thought must be to synthesize all the retrieved information.
- Briefly state how you will combine the mental models, core beliefs, and transcript evidence to accomplish the `core_task`.
- **Crucially, perform a final check:** Does your planned answer align with the Core Beliefs? Does it follow the steps of the Mental Model?

## CRITICAL: YOUR FINAL RESPONSE OUTPUT FORMAT ##

**MANDATORY FORMATTING REQUIREMENT:**

When you are ready to provide your final response to the user, you MUST format it EXACTLY as shown below:

```
Final Answer: [Your complete response here]
```

**STRICT RULES:**
1. Every final response MUST start with the EXACT text "Final Answer:" (capital F, capital A, with colon)
2. The prefix "Final Answer:" must appear on its own line or at the start of your response
3. After the colon, provide your complete answer in {persona_name}'s voice
4. This is a SYSTEM REQUIREMENT for proper UI rendering - do NOT skip this prefix
5. Do NOT use variations like "My final answer is" or "Here's my answer" - use EXACTLY "Final Answer:"

**Example Format:**
```
Final Answer: Alright, listen up. Here's the deal with building an audience...
```

**Content Requirements (after the "Final Answer:" prefix):**
- Your final answer's writing style **MUST STRICTLY ADHERE** to the rules in the <persona_linguistic_profile>
- **APPLY THE TONE:** Is your response energetic and conversational?
- **USE THE VOCABULARY & CATCHPHRASES:** Have you naturally integrated characteristic words or phrases?
- **NEVER break character.** Never mention you are an AI. Speak directly as {persona_name}

Remember: You are {persona_name}. Think, speak, and reason exactly as they would. But ALWAYS start your final response with "Final Answer:" - this is non-negotiable."""

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

## LANGUAGE HANDLING ##
CRITICAL: Detect the primary language of the input and produce your ENTIRE output EXCLUSIVELY in that single language.

**STRICT RULES:**
1. **SINGLE LANGUAGE ONLY**: Use ONLY the detected input language. NO mixing of languages.
2. **NO TRANSLATIONS**: Do not provide translations, explanations, or parenthetical notes in other languages.
3. **NO ROMANIZATION**: If input is Chinese, do NOT include Pinyin romanization. If input is English, do NOT include IPA or phonetics.
4. **ASSUME FLUENT READER**: The reader is a native/fluent speaker of the input language and does not need assistance from other languages.
5. **JSON FIELDS**: JSON keys remain in English, but all field VALUES must be in the detected input language.

**Examples:**
- ✅ CORRECT for Chinese input: "恢复秩序和繁荣"
- ❌ WRONG for Chinese input: "恢复秩序 (huīfù zhìxù - restore order)"
- ✅ CORRECT for English input: "Restore order and prosperity"
- ❌ WRONG for English input: "Restore order (恢复秩序)"

Your output language must match the input language, NOT the language this prompt is written in.

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
            invalid_tool_call_detected = False
            all_messages = []  # Track all messages for policy validation
            
            for step in agent_executor.stream(
                {"messages": messages}, 
                config=config, 
                stream_mode="values"
            ):
                # Get the latest message
                if step["messages"]:
                    all_messages = step["messages"]  # Keep track of full message history
                    latest_message = step["messages"][-1]
                    
                    # Validate tool calls before proceeding
                    if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                        if not self._validate_tool_calls(latest_message):
                            invalid_tool_call_detected = True
                            self.logger.error(f"Invalid tool call detected. Stopping agent execution.")
                            self.logger.error(f"Latest message: {latest_message}")
                            break
                        
                        # Log valid tool calls
                        for tool_call in latest_message.tool_calls:
                            self.logger.info(f"Tool called: {tool_call['name']}")
                    
                    # Capture the final AI response
                    if latest_message.type == 'ai' and latest_message.content:
                        response_content = latest_message.content
            
            # Handle invalid tool calls
            if invalid_tool_call_detected:
                error_response = f"I apologize, but I encountered a technical issue with tool selection. Let me try to answer your question directly based on my knowledge."
                self.logger.warning("Returning error response due to invalid tool call")
                return error_response
            
            # Validate tool usage policy after execution
            intent_type = query_analysis.get('intent_type', '')
            if not self._check_tool_usage_policy(all_messages, intent_type):
                self.logger.warning(f"Tool usage policy violated for intent_type='{intent_type}'")
                # Log the violation but still return the response
                # In production, you might want to retry or force tool usage
            
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
            
            # Use astream_events for real token streaming (LangGraph bug workaround)
            self.logger.info("DEBUG: Starting astream_events for token streaming")
            token_count = 0
            invalid_tool_detected = False
            
            async for event in agent_executor.astream_events(
                {"messages": messages},
                config=config,
                version="v2"  # Required for astream_events
            ):
                # Log all events for debugging
                event_type = event.get("event", "unknown")
                
                # Validate tool calls
                if event_type == "on_tool_start":
                    tool_name = event["data"].get("name", "unknown_tool")
                    valid_tools = {'retrieve_mental_models', 'retrieve_core_beliefs', 'retrieve_transcripts'}
                    
                    if tool_name not in valid_tools:
                        self.logger.error(f"INVALID TOOL CALL in streaming: '{tool_name}'")
                        invalid_tool_detected = True
                        yield f"\n\n[Error: Invalid tool call detected. Please retry your request.]"
                        break
                    
                    self.logger.info(f"Tool called: {tool_name}")
                
                # Filter for LLM token streaming events
                if event_type == "on_chat_model_stream" and not invalid_tool_detected:
                    chunk = event["data"]["chunk"]
                    
                    # Extract content from different chunk formats
                    content = ""
                    if hasattr(chunk, 'content') and chunk.content:
                        content = chunk.content
                    elif isinstance(chunk, dict) and 'content' in chunk:
                        content = chunk['content']
                    elif hasattr(chunk, 'choices') and chunk.choices:
                        # Handle OpenAI-style chunks
                        choice = chunk.choices[0]
                        if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                            content = choice.delta.content or ""
                    
                    if content:
                        token_count += 1
                        self.logger.info(f"DEBUG: Streaming token #{token_count}: '{content[:20]}{'...' if len(content) > 20 else ''}'")
                        yield content
                
                # Log other significant events
                elif event_type in ["on_chat_model_start", "on_chat_model_end"]:
                    self.logger.info(f"DEBUG: {event_type}")
            
            self.logger.info(f"DEBUG: Token streaming completed, total tokens yielded: {token_count}")
            self.logger.info("Streaming query processing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Streaming query processing failed: {str(e)}", exc_info=True)
            yield f"I'm experiencing some technical difficulties right now. Could you please try rephrasing your question? I want to make sure I give you the best response possible."
    
    async def process_query_structured_stream(
        self, 
        user_query: str, 
        session_id: Optional[str] = None
    ) -> AsyncIterator[StreamEvent]:
        """
        Process a user query with structured streaming events for UI organization.
        
        Uses LangGraph's stream_mode="messages" with node metadata to properly
        separate thinking/reasoning from final answers.
        
        Args:
            user_query: The user's input query
            session_id: Optional session ID for conversation continuity
            
        Yields:
            StreamEvent objects with categorized content
        """
        self.logger.info(f"Processing query (structured streaming): {user_query[:100]}...")
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Analyze query and load persona data
        query_analysis = self._analyze_query(user_query)
        persona_data = self._load_persona_data()
        agent_executor = self._create_agent(query_analysis, persona_data)
        
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
            
            # Track state using node transitions
            current_node = None
            emitted_tools = set()
            tools_called_list = []  # Track which tools were called for validation
            first_final_answer = True
            
            # Stream using messages mode with metadata
            async for chunk, metadata in agent_executor.astream(
                {"messages": messages},
                config=config,
                stream_mode="messages"
            ):
                langgraph_node = metadata.get("langgraph_node", "")
                
                # Agent node - check if this is thinking or final answer
                if langgraph_node == "agent":
                    # Check if message has tool calls (indicating thinking/planning phase)
                    has_tool_calls = hasattr(chunk, 'tool_calls') and chunk.tool_calls
                    
                    if has_tool_calls:
                        # Extract and emit tool call events for each unique tool
                        for tool_call in chunk.tool_calls:
                            tool_name = tool_call.get('name', 'unknown_tool')
                            tool_id = tool_call.get('id', '')
                            
                            # Track tools for policy validation
                            if tool_name in {'retrieve_mental_models', 'retrieve_core_beliefs', 'retrieve_transcripts'}:
                                tools_called_list.append(tool_name)
                            
                            # Only emit once per unique tool call
                            if tool_id and tool_id not in emitted_tools:
                                emitted_tools.add(tool_id)
                                yield StreamEvent(
                                    event_type="tool_call",
                                    content=f"Using {tool_name}",
                                    metadata={"tool_name": tool_name, "tool_id": tool_id}
                                )
                        
                        # Emit thinking content if present
                        if hasattr(chunk, 'content') and chunk.content:
                            yield StreamEvent(
                                event_type="thinking",
                                content=chunk.content if isinstance(chunk.content, str) else "",
                                metadata={"phase": "reasoning"}
                            )
                    else:
                        # No tool calls = final answer
                        if hasattr(chunk, 'content'):
                            content = chunk.content if isinstance(chunk.content, str) else ""
                            if content:
                                # Signal to clear tools on first final answer chunk
                                yield StreamEvent(
                                    event_type="final_answer",
                                    content=content,
                                    metadata={"phase": "response", "clear_tools": first_final_answer}
                                )
                                first_final_answer = False
                
                current_node = langgraph_node
            
            # Validate tool usage policy after streaming completes
            intent_type = query_analysis.get('intent_type', '')
            if intent_type != 'conversational_exchange' and not tools_called_list:
                self.logger.error(f"TOOL USAGE POLICY VIOLATION in streaming: No tools called for intent_type='{intent_type}'")
            elif tools_called_list:
                self.logger.info(f"Tool usage policy satisfied in streaming: {len(tools_called_list)} tool(s) called - {tools_called_list}")
            
            self.logger.info("Structured streaming completed successfully")
            
        except Exception as e:
            self.logger.error(f"Structured streaming query processing failed: {str(e)}", exc_info=True)
            yield StreamEvent(
                event_type="final_answer",
                content=f"I'm experiencing some technical difficulties right now. Could you please try rephrasing your question? I want to make sure I give you the best response possible.",
                metadata={"error": True}
            )
    
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