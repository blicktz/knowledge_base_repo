"""Main orchestrator for the persona agent system"""

from typing import List, Dict, Any

from ..config.settings import Settings
from ..tools import (
    QueryAnalyzerTool,
    PersonaDataTool,
    MentalModelsRetrieverTool,
    CoreBeliefsRetrieverTool,
    TranscriptRetrieverTool
)
from ..services.synthesis_engine import SynthesisEngine
from ..utils.logging import get_logger
from .agent_config import AgentConfig


class PersonaAgent:
    """Main orchestrator for the persona agent system"""
    
    def __init__(self, persona_id: str, settings: Settings):
        self.persona_id = persona_id
        self.settings = settings
        self.config = AgentConfig.from_settings(persona_id, settings)
        self.logger = get_logger(__name__)
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize synthesis engine
        self.synthesis_engine = SynthesisEngine(persona_id, settings)
        
        self.logger.info(f"PersonaAgent initialized for: {persona_id}")
    
    def _initialize_tools(self) -> List:
        """Initialize all agent tools"""
        self.logger.info("Initializing agent tools")
        
        tools = [
            QueryAnalyzerTool(self.persona_id, self.settings),      # Index 0
            PersonaDataTool(self.persona_id, self.settings),        # Index 1
            MentalModelsRetrieverTool(self.persona_id, self.settings),    # Index 2
            CoreBeliefsRetrieverTool(self.persona_id, self.settings),     # Index 3
            TranscriptRetrieverTool(self.persona_id, self.settings)       # Index 4
        ]
        
        self.logger.info(f"Initialized {len(tools)} tools")
        return tools
    
    def process_query(self, user_query: str) -> str:
        """
        Main entry point for processing user queries
        
        Workflow:
        1. Analyze query to extract core task
        2. Retrieve persona data (parallel)
        3. Retrieve mental models (parallel) 
        4. Retrieve core beliefs (parallel)
        5. Retrieve transcripts (parallel)
        6. Synthesize response with Chain-of-Thought
        """
        self.logger.info(f"Processing query: {user_query[:100]}...")
        
        try:
            # Step 1: Query Analysis
            self.logger.info("Step 1: Analyzing query")
            query_analysis = self.tools[0].execute(user_query)
            
            # Extract RAG query for retrieval
            rag_query = query_analysis.get('rag_query', user_query)
            metadata = {'rag_query': rag_query}
            
            # Step 2-5: Knowledge Retrieval
            self.logger.info("Step 2-5: Executing knowledge retrieval")
            retrieval_results = self._execute_retrieval_tools(rag_query, metadata)
            
            # Step 6: Synthesis
            self.logger.info("Step 6: Synthesizing response")
            response = self.synthesis_engine.synthesize(
                user_query=user_query,
                query_analysis=query_analysis,
                retrieval_results=retrieval_results
            )
            
            self.logger.info("Query processing completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}", exc_info=True)
            if self.config.fail_fast:
                raise
            else:
                # Fallback response (to be implemented later)
                return self._generate_fallback_response(user_query, str(e))
    
    def _execute_retrieval_tools(self, rag_query: str, metadata: Dict) -> Dict[str, Any]:
        """Execute all retrieval tools (sequential for now)"""
        
        retrieval_results = {}
        
        try:
            # Persona data (tool index 1)
            self.logger.info("Retrieving persona data")
            retrieval_results['persona_data'] = self.tools[1].execute(rag_query, metadata)
            
            # Mental models (tool index 2) 
            self.logger.info("Retrieving mental models")
            retrieval_results['mental_models'] = self.tools[2].execute(rag_query, metadata)
            
            # Core beliefs (tool index 3)
            self.logger.info("Retrieving core beliefs")
            retrieval_results['core_beliefs'] = self.tools[3].execute(rag_query, metadata)
            
            # Transcripts (tool index 4)
            self.logger.info("Retrieving transcripts")
            retrieval_results['transcripts'] = self.tools[4].execute(rag_query, metadata)
            
            return retrieval_results
            
        except Exception as e:
            self.logger.error(f"Retrieval execution failed: {str(e)}")
            raise
    
    def _generate_fallback_response(self, query: str, error: str) -> str:
        """Generate fallback response on error (placeholder)"""
        self.logger.warning(f"Generating fallback response due to error: {error}")
        
        return f"I apologize, but I encountered an issue processing your query: '{query[:100]}...'. Please try rephrasing your request or contact support if the issue persists."
    
    def get_tool_status(self) -> Dict[str, bool]:
        """Get status of all tools for debugging"""
        status = {}
        
        for tool in self.tools:
            try:
                # Basic health check - could be expanded
                status[tool.name] = True
            except Exception as e:
                self.logger.warning(f"Tool {tool.name} health check failed: {str(e)}")
                status[tool.name] = False
        
        return status