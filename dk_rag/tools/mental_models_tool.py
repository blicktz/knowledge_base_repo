"""Mental Models Retriever Tool - Retrieves relevant mental models using RAG pipeline"""

from typing import List, Dict, Any, Optional

from .base_tool import BasePersonaTool
from ..config.settings import Settings
from ..data.storage.mental_models_store import MentalModelsStore
from ..core.retrieval.knowledge_aware import MentalModelsPipeline
from ..core.retrieval.reranker import CrossEncoderReranker
from ..utils.logging import get_logger


class MentalModelsRetrieverTool(BasePersonaTool):
    """Retrieves relevant mental models using RAG pipeline"""
    
    name: str = "mental_models_retriever"
    description: str = "Retrieve relevant problem-solving frameworks and methodologies"
    
    def __init__(self, persona_id: str, settings: Settings):
        super().__init__(persona_id, settings)
        object.__setattr__(self, 'pipeline', self._initialize_pipeline())
        
    def _initialize_pipeline(self):
        """Initialize mental models RAG pipeline"""
        self.logger.info("Initializing mental models pipeline")
        
        try:
            # Initialize store and reranker
            store = MentalModelsStore(self.settings, self.persona_id)
            reranker = CrossEncoderReranker(
                model_name=self.settings.retrieval.reranking.model,
                use_cohere=self.settings.retrieval.reranking.use_cohere,
                device=self.settings.retrieval.reranking.device,
                batch_size=self.settings.retrieval.reranking.batch_size
            )
            
            # Create pipeline
            pipeline = MentalModelsPipeline(
                vector_store=store,
                reranker=reranker,
                persona_id=self.persona_id
            )
            
            self.logger.info("Mental models pipeline initialized successfully")
            return pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to initialize mental models pipeline: {str(e)}")
            raise
    
    def execute(self, query: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top 3 most relevant mental models
        """
        self.logger.info("Retrieving relevant mental models")
        
        # Use the RAG query from metadata if available
        rag_query = metadata.get('rag_query', query) if metadata else query
        
        # Get k value from settings
        k = self.settings.agent.tools.mental_models.k
        use_reranking = self.settings.agent.tools.mental_models.use_reranking
        
        self.logger.info(f"Retrieving {k} mental models with query: {rag_query[:100]}...")
        
        try:
            # Retrieve using pipeline
            results = self.pipeline.retrieve(
                query=rag_query,
                k=k,
                use_reranking=use_reranking
            )
            
            self.logger.info(f"Retrieved {len(results)} mental models")
            
            # Convert to serializable format
            serialized_results = []
            for result in results:
                if hasattr(result, 'to_dict'):
                    serialized_results.append(result.to_dict())
                else:
                    # Handle Document objects
                    serialized_results.append({
                        'content': result.page_content if hasattr(result, 'page_content') else str(result),
                        'metadata': result.metadata if hasattr(result, 'metadata') else {},
                        'score': getattr(result, 'score', None)
                    })
            
            # Log retrieval if enabled
            if self.settings.agent.tools.mental_models.log_retrievals:
                self.log_retrieval_results(rag_query, serialized_results)
            
            return serialized_results
            
        except Exception as e:
            self.logger.error(f"Mental models retrieval failed: {str(e)}")
            raise
    
    def log_retrieval_results(self, query: str, results: List[Dict]):
        """Log retrieval results for debugging"""
        if not self.settings.agent.logging.enabled:
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