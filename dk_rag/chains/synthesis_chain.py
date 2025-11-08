"""
LangChain LCEL chains for response synthesis
Replaces the manual synthesis engine with composable chains
"""

from typing import Dict, Any, List
from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_litellm import ChatLiteLLM

from ..config.settings import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


def format_linguistic_style(persona_data: Dict) -> str:
    """Format linguistic style for prompt context"""
    style_data = persona_data.get('linguistic_style', {})
    
    if not style_data:
        return "Professional, direct communication style."
    
    formatted = []
    
    # Add tone information
    if 'tone' in style_data:
        formatted.append(f"Tone: {style_data['tone']}")
    
    # Add catchphrases (limit to 5)
    if 'catchphrases' in style_data and style_data['catchphrases']:
        phrases = ', '.join(style_data['catchphrases'][:5])
        formatted.append(f"Common phrases: {phrases}")
    
    # Add vocabulary preferences
    if 'vocabulary' in style_data:
        formatted.append(f"Vocabulary style: {style_data['vocabulary']}")
    
    # Add communication patterns
    comm_patterns = persona_data.get('communication_patterns', {})
    if comm_patterns and 'formality' in comm_patterns:
        formatted.append(f"Formality level: {comm_patterns['formality']}")
    
    return '\n'.join(formatted) if formatted else "Professional, direct communication style."


def format_mental_models(mental_models: List[Dict]) -> str:
    """Format mental models for prompt context"""
    if not mental_models:
        return "No relevant mental models found."
    
    formatted = []
    # Use first k models (already filtered by config)
    for i, model in enumerate(mental_models, 1):
        content = model.get('content', str(model))
        # Truncate long content
        content_preview = content[:500] + "..." if len(content) > 500 else content
        formatted.append(f"{i}. {content_preview}")
    
    return '\n\n'.join(formatted)


def format_core_beliefs(core_beliefs: List[Dict]) -> str:
    """Format core beliefs for prompt context"""
    if not core_beliefs:
        return "No relevant core beliefs found."
    
    formatted = []
    # Use all beliefs (already filtered by config k value)
    for i, belief in enumerate(core_beliefs, 1):
        content = belief.get('content', str(belief))
        # Truncate long content  
        content_preview = content[:300] + "..." if len(content) > 300 else content
        formatted.append(f"{i}. {content_preview}")
    
    return '\n\n'.join(formatted)


def format_transcripts(transcripts: List[Dict]) -> str:
    """Format transcript chunks for prompt context"""
    if not transcripts:
        return "No relevant transcript content found."
    
    formatted = []
    # Use all transcript chunks (already filtered by config k value)
    for i, chunk in enumerate(transcripts, 1):
        content = chunk.get('content', str(chunk))
        source = chunk.get('metadata', {}).get('source', 'Unknown')
        # Truncate long content
        content_preview = content[:400] + "..." if len(content) > 400 else content
        formatted.append(f"[Chunk {i} from {source}]\n{content_preview}")
    
    return '\n\n'.join(formatted)


def create_synthesis_chain(persona_id: str, settings: Settings):
    """
    Create a LangChain LCEL chain for response synthesis.
    
    This replaces the manual synthesis engine with a composable chain.
    
    Args:
        persona_id: The persona identifier
        settings: Application settings
        
    Returns:
        Configured synthesis chain
    """
    logger.info(f"Creating synthesis chain for persona: {persona_id}")
    
    # Initialize LLM
    llm = ChatLiteLLM(
        model=settings.agent.synthesis.llm_model,
        temperature=settings.agent.synthesis.temperature,
        max_tokens=settings.agent.synthesis.max_tokens
    )
    
    # Create the synthesis prompt template
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a virtual AI persona of {persona_name}. Your goal is to respond to the user in a way that is identical to the real {persona_name} in tone, style, knowledge, and problem-solving approach.

### Language Handling ###
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

### Constitutional Rules ###
- You MUST adopt the tone and style described in the <linguistic_style> context
- You MUST use appropriate catchphrases and vocabulary where natural
- You MUST NOT break character or mention that you are an AI
- You MUST apply relevant mental models from the <mental_models> context
- You MUST ensure your reasoning aligns with the <core_beliefs> context
- You MUST ground your response in facts from the <factual_context>

### Context Block ###

<linguistic_style>
{linguistic_style}
</linguistic_style>

<mental_models>
{mental_models}
</mental_models>

<core_beliefs>
{core_beliefs}
</core_beliefs>

<factual_context>
{transcripts}
</factual_context>

### User Request ###
Core Task: {core_task}
Original Query: {user_query}

### Response Generation ###
Think through this step-by-step:
1. What is the user really asking for?
2. Which mental model best applies?
3. How do my core beliefs guide this response?
4. What specific facts support my answer?
5. How should I structure this authentically?

Then provide your final in-character response:"""),
        ("human", "{user_query}")
    ])
    
    # Create the synthesis chain
    synthesis_chain = (
        # Format all context data
        RunnablePassthrough.assign(
            persona_name=lambda x: persona_id.replace('_', ' ').title(),
            linguistic_style=lambda x: format_linguistic_style(x.get('persona_data', {})),
            mental_models=lambda x: format_mental_models(x.get('mental_models', [])),
            core_beliefs=lambda x: format_core_beliefs(x.get('core_beliefs', [])),
            transcripts=lambda x: format_transcripts(x.get('transcripts', [])),
            core_task=lambda x: x.get('query_analysis', {}).get('core_task', x.get('user_query', '')),
        )
        | synthesis_prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info("Synthesis chain created successfully")
    return synthesis_chain


def create_simple_response_chain(persona_id: str, settings: Settings):
    """
    Create a simpler response chain for cases where full synthesis isn't needed.
    
    Args:
        persona_id: The persona identifier
        settings: Application settings
        
    Returns:
        Configured simple response chain
    """
    
    # Initialize LLM
    llm = ChatLiteLLM(
        model=settings.agent.synthesis.llm_model,
        temperature=settings.agent.synthesis.temperature,
        max_tokens=settings.agent.synthesis.max_tokens
    )
    
    # Create simple prompt
    simple_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are {persona_id.replace('_', ' ').title()}. Respond authentically as this persona would, maintaining their characteristic tone and style."),
        ("human", "{user_query}")
    ])
    
    # Create simple chain
    simple_chain = simple_prompt | llm | StrOutputParser()
    
    return simple_chain


def create_context_aggregation_chain():
    """
    Create a chain that aggregates all retrieval context into a structured format.
    
    This can be used as a preprocessing step before synthesis.
    
    Returns:
        Context aggregation chain
    """
    
    def aggregate_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate all context into a structured format"""
        
        aggregated = {
            "user_query": inputs.get("user_query", ""),
            "query_analysis": inputs.get("query_analysis", {}),
            "persona_data": inputs.get("persona_data", {}),
            "mental_models": inputs.get("mental_models", []),
            "core_beliefs": inputs.get("core_beliefs", []),
            "transcripts": inputs.get("transcripts", []),
            "context_summary": {
                "mental_models_count": len(inputs.get("mental_models", [])),
                "core_beliefs_count": len(inputs.get("core_beliefs", [])),
                "transcript_chunks_count": len(inputs.get("transcripts", [])),
                "has_persona_data": bool(inputs.get("persona_data", {}))
            }
        }
        
        logger.info(f"Aggregated context: {aggregated['context_summary']}")
        return aggregated
    
    return RunnableLambda(aggregate_context)


# Chain composition helpers
def create_full_synthesis_pipeline(persona_id: str, settings: Settings):
    """
    Create a complete synthesis pipeline with context aggregation and synthesis.
    
    Args:
        persona_id: The persona identifier
        settings: Application settings
        
    Returns:
        Complete synthesis pipeline
    """
    
    context_chain = create_context_aggregation_chain()
    synthesis_chain = create_synthesis_chain(persona_id, settings)
    
    # Compose the full pipeline
    full_pipeline = context_chain | synthesis_chain
    
    return full_pipeline


# Export the main creation functions
__all__ = [
    "create_synthesis_chain",
    "create_simple_response_chain", 
    "create_context_aggregation_chain",
    "create_full_synthesis_pipeline"
]