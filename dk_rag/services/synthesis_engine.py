"""Response Synthesis with Chain-of-Thought reasoning"""

import json
import re
from datetime import datetime
from typing import Dict, Any

from litellm import completion

from ..config.settings import Settings
from ..utils.logging import get_logger


class SynthesisEngine:
    """Synthesizes final responses using Chain-of-Thought reasoning"""
    
    def __init__(self, persona_id: str, settings: Settings):
        self.persona_id = persona_id
        self.settings = settings
        self.logger = get_logger(__name__)
        
        # LLM configuration from settings
        self.model_name = settings.agent.synthesis.llm_model
        self.temperature = settings.agent.synthesis.temperature
        self.max_tokens = settings.agent.synthesis.max_tokens
        self.use_chain_of_thought = settings.agent.synthesis.use_chain_of_thought
        self.include_scratchpad = settings.agent.synthesis.include_scratchpad
        
    def synthesize(
        self,
        user_query: str,
        query_analysis: Dict[str, Any],
        retrieval_results: Dict[str, Any]
    ) -> str:
        """
        Synthesize response using master prompt with Chain-of-Thought
        """
        self.logger.info("Starting response synthesis")
        
        # Build the master synthesis prompt
        prompt = self._build_synthesis_prompt(
            user_query,
            query_analysis,
            retrieval_results
        )
        
        # Call LLM for synthesis
        self.logger.info("Calling LLM for synthesis")
        response = self._call_llm(prompt)
        
        # Parse response to extract answer (remove scratchpad)
        final_answer = self._extract_final_answer(response)
        
        # Log the complete interaction if enabled
        if self.settings.agent.logging.enabled:
            self._log_synthesis_interaction(prompt, response, final_answer)
        
        self.logger.info("Synthesis completed")
        
        return final_answer
    
    def _build_synthesis_prompt(
        self,
        user_query: str,
        query_analysis: Dict,
        retrieval_results: Dict
    ) -> str:
        """Build the master synthesis prompt with constitutional rules"""
        
        template = """You are a virtual AI persona of {persona_name}. Your goal is to respond to the user in a way that is identical to the real {persona_name} in tone, style, knowledge, and problem-solving approach.

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

<user_task>
Core Task: {core_task}
Original Query: {user_query}
</user_task>

### Response Generation ###
{scratchpad_instruction}

{answer_instruction}"""

        # Build scratchpad instruction
        scratchpad_instruction = ""
        answer_instruction = "[Your final in-character response goes here]"
        
        if self.use_chain_of_thought and self.include_scratchpad:
            scratchpad_instruction = """First, think through your response step-by-step in a private scratchpad. Then write the final answer.

<scratchpad>
1. **User's Core Need**: What is the user really asking for?
2. **Relevant Mental Model**: Which framework from the context best applies?
3. **Belief Alignment**: How do the core beliefs guide my response?
4. **Factual Support**: What specific facts from transcripts support my answer?
5. **Response Structure**: How will I structure this in character?
6. **Tone Check**: Is this authentic to the persona's style?
</scratchpad>

<answer>
[Your final in-character response goes here]
</answer>"""
            answer_instruction = ""
        
        # Fill in the template
        filled_prompt = template.format(
            persona_name=self.persona_id.replace('_', ' ').title(),
            linguistic_style=self._format_linguistic_style(retrieval_results['persona_data']),
            mental_models=self._format_mental_models(retrieval_results['mental_models']),
            core_beliefs=self._format_core_beliefs(retrieval_results['core_beliefs']),
            transcripts=self._format_transcripts(retrieval_results['transcripts']),
            core_task=query_analysis.get('core_task', 'Respond to user query'),
            user_query=user_query,
            scratchpad_instruction=scratchpad_instruction,
            answer_instruction=answer_instruction
        )
        
        return filled_prompt
    
    def _format_linguistic_style(self, persona_data: Dict) -> str:
        """Format linguistic style for prompt"""
        style_data = persona_data.get('linguistic_style', {})
        
        if not style_data:
            return "No specific linguistic style data available."
        
        formatted = []
        
        # Add tone information
        if 'tone' in style_data:
            formatted.append(f"Tone: {style_data['tone']}")
        
        # Add catchphrases
        if 'catchphrases' in style_data and style_data['catchphrases']:
            phrases = ', '.join(style_data['catchphrases'][:5])  # Limit to 5
            formatted.append(f"Common phrases: {phrases}")
        
        # Add vocabulary preferences
        if 'vocabulary' in style_data:
            formatted.append(f"Vocabulary style: {style_data['vocabulary']}")
        
        # Add communication patterns
        comm_patterns = persona_data.get('communication_patterns', {})
        if comm_patterns:
            if 'formality' in comm_patterns:
                formatted.append(f"Formality level: {comm_patterns['formality']}")
        
        return '\n'.join(formatted) if formatted else "Professional, direct communication style."
    
    def _format_mental_models(self, mental_models: list) -> str:
        """Format mental models for prompt"""
        if not mental_models:
            return "No relevant mental models found."
        
        formatted = []
        for i, model in enumerate(mental_models[:3], 1):  # Top 3
            content = model.get('content', str(model))
            formatted.append(f"{i}. {content[:500]}...")
        
        return '\n\n'.join(formatted)
    
    def _format_core_beliefs(self, core_beliefs: list) -> str:
        """Format core beliefs for prompt"""
        if not core_beliefs:
            return "No relevant core beliefs found."
        
        formatted = []
        for i, belief in enumerate(core_beliefs[:5], 1):  # Top 5
            content = belief.get('content', str(belief))
            formatted.append(f"{i}. {content[:300]}...")
        
        return '\n\n'.join(formatted)
    
    def _format_transcripts(self, transcripts: list) -> str:
        """Format transcript chunks for prompt"""
        if not transcripts:
            return "No relevant transcript content found."
        
        formatted = []
        for i, chunk in enumerate(transcripts[:5], 1):  # Top 5
            content = chunk.get('content', str(chunk))
            source = chunk.get('source', 'Unknown')
            formatted.append(f"[Chunk {i} from {source}]\n{content[:400]}...")
        
        return '\n\n'.join(formatted)
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM for synthesis"""
        try:
            response = completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"LLM synthesis call failed: {str(e)}")
            raise
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from response, removing scratchpad"""
        
        if not self.include_scratchpad:
            return response.strip()
        
        # Look for answer tags
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Fallback: look for content after scratchpad
        scratchpad_end = response.find('</scratchpad>')
        if scratchpad_end != -1:
            remaining = response[scratchpad_end + len('</scratchpad>'):].strip()
            # Remove any remaining tags
            remaining = re.sub(r'<[^>]+>', '', remaining).strip()
            if remaining:
                return remaining
        
        # Final fallback: return entire response
        self.logger.warning("Could not extract answer from response, returning full response")
        return response.strip()
    
    def _log_synthesis_interaction(self, prompt: str, response: str, final_answer: str):
        """Log synthesis interaction for debugging"""
        
        # Use base tool logging mechanism
        from ..tools.base_tool import BasePersonaTool
        
        # Create a temporary tool instance for logging
        class SynthesisLogger(BasePersonaTool):
            name = "synthesis"
            def execute(self, query, metadata=None):
                pass
        
        logger_tool = SynthesisLogger(self.persona_id, self.settings)
        
        extracted_data = {
            "synthesis_model": self.model_name,
            "temperature": self.temperature,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "final_answer_length": len(final_answer),
            "timestamp": datetime.now().isoformat()
        }
        
        logger_tool.log_llm_interaction(
            prompt=prompt,
            response=response,
            extracted=extracted_data,
            component_name="synthesis"
        )