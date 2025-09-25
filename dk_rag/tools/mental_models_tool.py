"""Mental Models Retriever Tool - Retrieves relevant mental models using RAG pipeline"""

from typing import List, Dict, Any, Optional

from .base_tool import BasePersonaTool
from ..config.settings import Settings
from ..core.knowledge_indexer import KnowledgeIndexer
from ..core.persona_manager import PersonaManager
from ..utils.logging import get_logger


class MentalModelsRetrieverTool(BasePersonaTool):
    """Retrieves relevant mental models using RAG pipeline"""
    
    name: str = "mental_models_retriever"
    description: str = "Retrieve relevant problem-solving frameworks and methodologies"
    
    def __init__(self, persona_id: str, settings: Settings):
        super().__init__(persona_id, settings)
        object.__setattr__(self, 'knowledge_indexer', self._initialize_knowledge_indexer())
        
    def _initialize_knowledge_indexer(self):
        """Initialize knowledge indexer for mental models search"""
        self.logger.info("Initializing knowledge indexer for mental models")
        
        try:
            # Initialize persona manager and knowledge indexer
            persona_manager = PersonaManager(self.settings)
            knowledge_indexer = KnowledgeIndexer(self.settings, persona_manager, self.persona_id)
            
            self.logger.info("Mental models knowledge indexer initialized successfully")
            return knowledge_indexer
            
        except Exception as e:
            self.logger.error(f"Failed to initialize mental models knowledge indexer: {str(e)}")
            raise
    
    def execute(self, query: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant mental models using the working test pattern
        """
        self.logger.info("Retrieving relevant mental models")
        
        # Use the RAG query from metadata if available
        rag_query = metadata.get('rag_query', query) if metadata else query
        
        # Get k value from settings with fallback
        k = getattr(getattr(getattr(self.settings, 'agent', None), 'tools', None), 'mental_models', {}).get('k', 3)
        use_reranking = getattr(getattr(getattr(self.settings, 'agent', None), 'tools', None), 'mental_models', {}).get('use_reranking', True)
        log_retrievals = getattr(getattr(getattr(self.settings, 'agent', None), 'tools', None), 'mental_models', {}).get('log_retrievals', True)
        
        self.logger.info(f"Retrieving {k} mental models with query: {rag_query[:100]}...")
        
        try:
            # Use the working pattern from test_knowledge_interactive.py
            results = self.knowledge_indexer.search_mental_models(
                query=rag_query,
                persona_id=self.persona_id,
                k=k,
                use_reranking=use_reranking,
                return_scores=True
            )
            
            self.logger.info(f"Retrieved {len(results)} mental models")
            
            # Convert to serializable format
            serialized_results = []
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    # Handle (result, score) tuples
                    item, score = result
                    if hasattr(item, 'to_dict'):
                        result_dict = item.to_dict()
                        result_dict['score'] = score
                        serialized_results.append(result_dict)
                    else:
                        serialized_results.append({
                            'content': item.page_content if hasattr(item, 'page_content') else str(item),
                            'metadata': item.metadata if hasattr(item, 'metadata') else {},
                            'score': score
                        })
                else:
                    # Handle single results
                    if hasattr(result, 'to_dict'):
                        serialized_results.append(result.to_dict())
                    else:
                        serialized_results.append({
                            'content': result.page_content if hasattr(result, 'page_content') else str(result),
                            'metadata': result.metadata if hasattr(result, 'metadata') else {},
                            'score': getattr(result, 'score', None)
                        })
            
            # Log retrieval if enabled
            if log_retrievals:
                self.log_retrieval_results(rag_query, serialized_results)
            
            return serialized_results
            
        except Exception as e:
            self.logger.error(f"Mental models retrieval failed: {str(e)}")
            # Check fail_fast setting with fallback
            fail_fast = getattr(getattr(getattr(self.settings, 'agent', None), 'error_handling', None), 'throw_on_error', True)
            if fail_fast:
                raise
            else:
                self.logger.warning("Returning empty results due to failure")
                return []
    
    def log_retrieval_results(self, query: str, results: List[Dict]):
        """Log retrieval results for debugging"""
        # Check logging settings with fallbacks
        logging_enabled = getattr(getattr(self.settings, 'agent', None), 'logging', {}).get('enabled', True)
        
        if not logging_enabled:
            return
        
        # Log summary
        self.logger.debug(f"Mental models retrieved for query: {query[:100]}")
        for i, result in enumerate(results):
            content_preview = result.get('content', '')[:200]
            score = result.get('score', 'N/A')
            self.logger.debug(f"  [{i+1}] Score: {score}, Content: {content_preview}...")
        
        # Save detailed results if needed
        self.log_llm_interaction(
            prompt=query,
            response="",  # No LLM response for retrieval
            extracted={"query": query, "results": results},
            component_name="mental_models"
        )