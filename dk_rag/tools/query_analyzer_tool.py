"""Query Analyzer Tool - Extracts core task and generates RAG query"""

import json
from typing import Dict, Any, Optional

from litellm import completion

from .base_tool import BasePersonaTool
from ..config.settings import Settings
from ..utils.logging import get_logger


class QueryAnalyzerTool(BasePersonaTool):
    """Analyzes user queries to extract core tasks and generate RAG queries"""
    
    name: str = "query_analyzer"
    description: str = "Extract core task and generate optimized RAG query from user input"
    
    def __init__(self, persona_id: str, settings: Settings):
        super().__init__(persona_id, settings)
        object.__setattr__(self, 'model_name', settings.agent.query_analysis.llm_model)
        object.__setattr__(self, 'temperature', settings.agent.query_analysis.temperature)
        object.__setattr__(self, 'max_tokens', settings.agent.query_analysis.max_tokens)
    
    def execute(self, query: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze query to extract:
        - core_task: The main task the user wants to accomplish
        - rag_query: Optimized query for RAG retrieval
        - provided_context: Any context provided by the user
        - intent_type: Classification of user intent
        """
        self.logger.info("Analyzing query to extract core task")
        
        # Build and call LLM
        prompt = self.build_analysis_prompt(query)
        response = self.call_llm(prompt)
        
        # Extract structured output
        extracted = self.parse_analysis_response(response)
        
        # Log the interaction if enabled
        if self.settings.agent.logging.enabled:
            self.log_llm_interaction(prompt, response, extracted, "query_analysis")
        
        self.logger.info(f"Extracted core task: {extracted.get('core_task', '')[:100]}...")
        
        return extracted
    
    def build_analysis_prompt(self, query: str) -> str:
        """Build prompt for query analysis"""
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
        
        return prompt
    
    def call_llm(self, prompt: str) -> str:
        """Call LLM for analysis"""
        try:
            response = completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            raise
    
    def parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract structured data"""
        try:
            # Try to parse as JSON
            extracted = json.loads(response)
            
            # Validate required fields
            required_fields = ['core_task', 'rag_query', 'intent_type']
            for field in required_fields:
                if field not in extracted:
                    extracted[field] = ""
            
            # Add provided_context if missing
            if 'provided_context' not in extracted:
                extracted['provided_context'] = ""
            
            return extracted
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {str(e)}")
            self.logger.debug(f"Raw response: {response}")
            
            # Return default structure on parse failure
            return {
                "core_task": "Failed to parse query",
                "rag_query": "",
                "provided_context": "",
                "intent_type": "unknown"
            }