"""Transcript Retriever Tool - Retrieves relevant transcript chunks using Phase 2 advanced pipeline"""

from typing import List, Dict, Any, Optional

from .base_tool import BasePersonaTool
from ..config.settings import Settings
from ..core.knowledge_indexer import KnowledgeIndexer
from ..core.persona_manager import PersonaManager
from ..utils.logging import get_logger


class TranscriptRetrieverTool(BasePersonaTool):
    """Retrieves relevant transcript chunks using Phase 2 advanced pipeline"""
    
    name: str = "transcript_retriever"
    description: str = "Retrieve relevant transcript chunks using advanced RAG"
    
    def __init__(self, persona_id: str, settings: Settings):
        super().__init__(persona_id, settings)
        object.__setattr__(self, 'pipeline', self._initialize_pipeline())
        
    def _initialize_pipeline(self):
        """Initialize Phase 2 advanced retrieval pipeline"""
        self.logger.info("Initializing transcript retrieval pipeline")
        
        try:
            # Initialize knowledge indexer and pipeline
            persona_manager = PersonaManager(self.settings)
            knowledge_indexer = KnowledgeIndexer(self.settings, persona_manager, self.persona_id)
            
            # Get the advanced retrieval pipeline from Phase 2
            pipeline = knowledge_indexer.get_advanced_retrieval_pipeline(self.persona_id)
            
            self.logger.info("Transcript retrieval pipeline initialized successfully")
            return pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to initialize transcript retrieval pipeline: {str(e)}")
            raise
    
    def execute(self, query: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top 5 most relevant transcript chunks using Phase 2 pipeline
        """
        self.logger.info("Retrieving relevant transcript chunks")
        
        # Use the RAG query from metadata if available
        rag_query = metadata.get('rag_query', query) if metadata else query
        
        # Get k value from settings
        k = self.settings.agent.tools.transcripts.k
        retrieval_k = self.settings.agent.tools.transcripts.retrieval_k
        
        self.logger.info(f"Retrieving {k} transcript chunks with query: {rag_query[:100]}...")
        
        try:
            # Use Phase 2 advanced pipeline
            results = self.pipeline.retrieve(
                query=rag_query,
                k=k,
                retrieval_k=retrieval_k
            )
            
            self.logger.info(f"Retrieved {len(results)} transcript chunks")
            
            # Convert to serializable format
            serialized_results = []
            for doc in results:
                chunk_dict = {
                    'content': doc.page_content if hasattr(doc, 'page_content') else str(doc),
                    'metadata': doc.metadata if hasattr(doc, 'metadata') else {},
                    'score': getattr(doc, 'score', None)
                }
                
                # Add transcript-specific metadata if available
                if hasattr(doc, 'metadata') and doc.metadata:
                    # Extract useful metadata
                    metadata_dict = chunk_dict['metadata']
                    if 'source' in metadata_dict:
                        chunk_dict['source'] = metadata_dict['source']
                    if 'timestamp' in metadata_dict:
                        chunk_dict['timestamp'] = metadata_dict['timestamp']
                    if 'chunk_id' in metadata_dict:
                        chunk_dict['chunk_id'] = metadata_dict['chunk_id']
                
                serialized_results.append(chunk_dict)
            
            # Log retrieval if enabled
            if self.settings.agent.tools.transcripts.log_retrievals:
                self.log_retrieval_results(rag_query, serialized_results)
            
            return serialized_results
            
        except Exception as e:
            self.logger.error(f"Transcript retrieval failed: {str(e)}")
            raise
    
    def log_retrieval_results(self, query: str, results: List[Dict]):
        """Log retrieval results for debugging"""
        if not self.settings.agent.logging.enabled:
            return
        
        # Log summary
        self.logger.debug(f"Transcripts retrieved for query: {query[:100]}")
        for i, result in enumerate(results):
            content_preview = result.get('content', '')[:200]
            score = result.get('score', 'N/A')
            source = result.get('source', 'Unknown')
            self.logger.debug(f"  [{i+1}] Score: {score}, Source: {source}, Content: {content_preview}...")
        
        # Save detailed results if needed
        self.log_llm_interaction(
            prompt=query,
            response="",  # No LLM response for retrieval
            extracted={"query": query, "results": results},
            component_name="transcripts"
        )