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
        Main entry point for processing user queries with comprehensive error handling
        
        Workflow:
        1. Analyze query to extract core task
        2. Retrieve persona data (parallel)
        3. Retrieve mental models (parallel) 
        4. Retrieve core beliefs (parallel)
        5. Retrieve transcripts (parallel)
        6. Synthesize response with Chain-of-Thought
        """
        self.logger.info(f"Processing query: {user_query[:100]}...")
        
        # Validate input
        if not user_query or not user_query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            # Step 1: Query Analysis
            self.logger.info("Step 1: Analyzing query")
            query_analysis = None
            try:
                query_analysis = self.tools[0].execute(user_query)
                self.logger.info("✅ Query analysis completed successfully")
            except Exception as e:
                self.logger.error(f"❌ Query analysis failed: {str(e)}")
                if self.config.fail_fast:
                    raise
                # Fallback: use original query
                query_analysis = {
                    'core_task': user_query,
                    'rag_query': user_query,
                    'provided_context': '',
                    'intent_type': 'general'
                }
                self.logger.info("🔄 Using fallback query analysis")
            
            # Extract RAG query for retrieval
            rag_query = query_analysis.get('rag_query', user_query)
            metadata = {'rag_query': rag_query}
            
            # Step 2-5: Knowledge Retrieval
            self.logger.info("Step 2-5: Executing knowledge retrieval")
            retrieval_results = self._execute_retrieval_tools(rag_query, metadata)
            
            # Validate we have at least some data
            total_results = sum(len(results) if isinstance(results, list) else 1 
                              for results in retrieval_results.values() 
                              if results is not None)
            
            if total_results == 0:
                self.logger.warning("⚠️ No retrieval results obtained - generating basic response")
            else:
                self.logger.info(f"✅ Retrieved {total_results} total knowledge items")
            
            # Step 6: Synthesis
            self.logger.info("Step 6: Synthesizing response")
            try:
                response = self.synthesis_engine.synthesize(
                    user_query=user_query,
                    query_analysis=query_analysis,
                    retrieval_results=retrieval_results
                )
                
                # Validate response
                if not response or len(response.strip()) < 10:
                    self.logger.warning("⚠️ Generated response is too short, using fallback")
                    return self._generate_fallback_response(user_query, "Generated response was insufficient")
                
                self.logger.info("✅ Query processing completed successfully")
                return response
                
            except Exception as e:
                self.logger.error(f"❌ Synthesis failed: {str(e)}")
                if self.config.fail_fast:
                    raise
                return self._generate_fallback_response(user_query, f"Synthesis error: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"❌ Query processing failed: {str(e)}", exc_info=True)
            if self.config.fail_fast:
                raise
            else:
                return self._generate_fallback_response(user_query, str(e))
    
    def _execute_retrieval_tools(self, rag_query: str, metadata: Dict) -> Dict[str, Any]:
        """Execute all retrieval tools with individual error handling"""
        
        retrieval_results = {}
        
        # Tool execution with individual error handling
        tools_config = [
            (1, 'persona_data', 'Retrieving persona data'),
            (2, 'mental_models', 'Retrieving mental models'), 
            (3, 'core_beliefs', 'Retrieving core beliefs'),
            (4, 'transcripts', 'Retrieving transcripts')
        ]
        
        for tool_index, result_key, log_message in tools_config:
            try:
                self.logger.info(log_message)
                result = self.tools[tool_index].execute(rag_query, metadata)
                retrieval_results[result_key] = result
                
                # Log success with result count
                if isinstance(result, list):
                    self.logger.info(f"✅ {result_key}: Retrieved {len(result)} items")
                else:
                    self.logger.info(f"✅ {result_key}: Retrieved successfully")
                    
            except Exception as e:
                self.logger.error(f"❌ {result_key} retrieval failed: {str(e)}")
                
                # Check if we should fail fast or continue
                if self.config.fail_fast:
                    self.logger.error(f"Fail-fast enabled, stopping execution")
                    raise
                else:
                    # Continue with empty results for this tool
                    retrieval_results[result_key] = []
                    self.logger.info(f"🔄 Continuing with empty {result_key} results")
        
        # Log overall results summary
        success_count = sum(1 for v in retrieval_results.values() if v is not None and v != [])
        total_tools = len(tools_config)
        
        if success_count == 0:
            self.logger.warning(f"⚠️ All retrieval tools failed ({success_count}/{total_tools})")
        elif success_count < total_tools:
            self.logger.warning(f"⚠️ Partial retrieval success ({success_count}/{total_tools})")
        else:
            self.logger.info(f"✅ All retrieval tools succeeded ({success_count}/{total_tools})")
            
        return retrieval_results
    
    def _generate_fallback_response(self, query: str, error: str) -> str:
        """Generate improved fallback response with persona context"""
        self.logger.warning(f"Generating fallback response due to error: {error}")
        
        # Try to provide a more helpful fallback response
        fallback_responses = [
            f"I'm experiencing some technical difficulties right now, but I understand you're asking about: '{query[:100]}...'. ",
            f"While I can't access all my knowledge at the moment due to: {error[:50]}..., I'd be happy to help if you could try rephrasing your question. ",
            f"Let me try to provide a general response to your question about: '{query[:50]}...', though I may not have access to all my usual resources right now."
        ]
        
        # Use the first fallback as default
        base_response = fallback_responses[0]
        
        # Add helpful guidance
        guidance = (
            "You might try:\n"
            "• Rephrasing your question more specifically\n" 
            "• Asking about a different topic\n"
            "• Trying again in a moment as the issue may be temporary"
        )
        
        return f"{base_response}\n\n{guidance}"
    
    def get_tool_status(self) -> Dict[str, bool]:
        """Get comprehensive status of all tools for debugging"""
        status = {}
        
        for i, tool in enumerate(self.tools):
            try:
                # Basic health check - verify tool is accessible and has required attributes
                tool_name = getattr(tool, 'name', f'tool_{i}')
                
                # Check if tool has required methods
                has_execute = hasattr(tool, 'execute') and callable(getattr(tool, 'execute'))
                
                # For retrieval tools, check if they have their core dependencies
                if hasattr(tool, 'knowledge_indexer'):
                    # Check knowledge indexer tools
                    has_indexer = tool.knowledge_indexer is not None
                    status[tool_name] = has_execute and has_indexer
                elif hasattr(tool, 'pipeline'):
                    # Check pipeline-based tools
                    has_pipeline = tool.pipeline is not None
                    status[tool_name] = has_execute and has_pipeline
                else:
                    # Basic tools just need execute method
                    status[tool_name] = has_execute
                
                if status[tool_name]:
                    self.logger.debug(f"✅ Tool {tool_name}: healthy")
                else:
                    self.logger.warning(f"⚠️ Tool {tool_name}: unhealthy")
                    
            except Exception as e:
                tool_name = getattr(tool, 'name', f'tool_{i}')
                self.logger.warning(f"❌ Tool {tool_name} health check failed: {str(e)}")
                status[tool_name] = False
        
        return status