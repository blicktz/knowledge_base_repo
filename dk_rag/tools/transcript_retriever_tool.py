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
        Retrieve relevant transcript chunks using Phase 2 advanced pipeline with working test pattern
        """
        self.logger.info("Retrieving relevant transcript chunks")
        
        # Use the RAG query from metadata if available
        rag_query = metadata.get('rag_query', query) if metadata else query
        
        # Get settings with fallbacks
        k = getattr(getattr(getattr(self.settings, 'agent', None), 'tools', None), 'transcripts', {}).get('k', 5)
        retrieval_k = getattr(getattr(getattr(self.settings, 'agent', None), 'tools', None), 'transcripts', {}).get('retrieval_k', 25)
        use_phase2_pipeline = getattr(getattr(getattr(self.settings, 'agent', None), 'tools', None), 'transcripts', {}).get('use_phase2_pipeline', True)
        log_retrievals = getattr(getattr(getattr(self.settings, 'agent', None), 'tools', None), 'transcripts', {}).get('log_retrievals', True)
        
        self.logger.info(f"Retrieving {k} transcript chunks with query: {rag_query[:100]}...")
        
        try:
            # Use the working pattern from test_phase2_interactive.py
            if use_phase2_pipeline and self.pipeline:
                results = self.pipeline.retrieve(
                    query=rag_query,
                    k=k,
                    use_hyde=True,
                    use_hybrid=True,
                    use_reranking=True,
                    return_scores=True
                )
            else:
                # Fallback to basic retrieval if Phase 2 not available
                self.logger.warning("Phase 2 pipeline not available, falling back to basic retrieval")
                # This would need to be implemented as a fallback
                results = []
            
            self.logger.info(f"Retrieved {len(results)} transcript chunks")
            
            # Convert to serializable format
            serialized_results = []
            for item in results:
                if isinstance(item, tuple) and len(item) == 2:
                    # Handle (doc, score) tuples
                    doc, score = item
                    chunk_dict = {
                        'content': doc.page_content if hasattr(doc, 'page_content') else str(doc),
                        'metadata': doc.metadata if hasattr(doc, 'metadata') else {},
                        'score': score
                    }
                else:
                    # Handle single documents
                    doc = item
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
            if log_retrievals:
                self.log_retrieval_results(rag_query, serialized_results)
            
            return serialized_results
            
        except Exception as e:
            self.logger.error(f"Transcript retrieval failed: {str(e)}")
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