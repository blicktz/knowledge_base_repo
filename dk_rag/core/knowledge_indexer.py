"""
Knowledge indexer for building and managing the vector knowledge base
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from typing import List, Dict, Any, Optional
from pathlib import Path

from ..data.storage.vector_store import VectorStore
from ..data.processing.transcript_loader import TranscriptLoader
from ..data.processing.chunk_processor import ChunkProcessor
from ..core.persona_extractor import PersonaExtractor
from ..core.statistical_analyzer import StatisticalAnalyzer
from ..core.persona_manager import PersonaManager
from ..config.settings import Settings
from ..utils.logging import get_logger
from ..utils.validation import validate_documents

# Phase 2 imports
from ..data.storage.bm25_store import BM25Store
from ..data.storage.retrieval_cache import RetrievalCache
from ..core.retrieval.hyde_retriever import HyDERetriever
from ..core.retrieval.hybrid_retriever import HybridRetriever
from ..core.retrieval.reranker import CrossEncoderReranker
from ..core.retrieval.advanced_pipeline import AdvancedRetrievalPipeline
from ..config.retrieval_config import Phase2RetrievalConfig


class KnowledgeIndexer:
    """
    Orchestrates the complete knowledge indexing and persona extraction pipeline
    with multi-tenant support for isolated persona management
    """
    
    def __init__(self, settings: Settings, persona_manager: PersonaManager, persona_id: Optional[str] = None):
        """Initialize the knowledge indexer with shared persona manager and optional persona context"""
        self.settings = settings
        self.logger = get_logger(__name__)
        self.persona_id = persona_id
        
        # Use provided persona manager instance
        self.persona_manager = persona_manager
        
        # Initialize components - persona-specific if persona_id provided
        if persona_id:
            # Ensure persona is registered
            if not self.persona_manager.persona_exists(persona_id):
                raise ValueError(f"Persona '{persona_id}' not registered. Register it first.")
            
            # Use persona-specific components
            self.vector_store = self.persona_manager.get_persona_vector_store(persona_id)
        else:
            # Legacy mode - use global components
            self.vector_store = VectorStore(settings)
        
        self.transcript_loader = TranscriptLoader(settings)
        self.chunk_processor = ChunkProcessor(settings)
        self.persona_extractor = PersonaExtractor(settings, persona_id)
        self.statistical_analyzer = StatisticalAnalyzer(settings, persona_id)
        
        # Phase 2 components (initialized on first use)
        self.retrieval_config: Optional[Phase2RetrievalConfig] = None
        self.bm25_store: Optional[BM25Store] = None
        self.retrieval_cache: Optional[RetrievalCache] = None
        self.hyde_retriever: Optional[HyDERetriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.reranker: Optional[CrossEncoderReranker] = None
        self.advanced_pipeline: Optional[AdvancedRetrievalPipeline] = None
        
        # Initialize Phase 2 if enabled in settings
        if hasattr(settings, 'retrieval') and getattr(settings.retrieval, 'enabled', False):
            self.setup_phase2_retrieval()
    
    def build_knowledge_base(self, 
                           documents_dir: str,
                           persona_id: Optional[str] = None,
                           rebuild: bool = False,
                           file_pattern: str = "*.md") -> Dict[str, Any]:
        """
        Build complete knowledge base from documents for a specific persona
        
        Args:
            documents_dir: Directory containing documents
            persona_id: Persona identifier for tenant isolation
            rebuild: Whether to rebuild from scratch
            file_pattern: Pattern for files to include
            
        Returns:
            Dictionary with indexing results
        """
        # Use instance persona_id if not provided
        if persona_id is None:
            persona_id = self.persona_id
        
        if not persona_id:
            raise ValueError("persona_id required for building knowledge base")
        
        self.logger.info(f"Building knowledge base for persona '{persona_id}' from: {documents_dir}")
        
        # Get persona-specific vector store
        vector_store = self.persona_manager.get_persona_vector_store(persona_id)
        
        # Clear existing data if rebuilding
        if rebuild:
            self.logger.info(f"Clearing existing vector store for persona '{persona_id}'...")
            vector_store.clear_collection()
        
        # Load documents
        self.logger.info("Loading documents...")
        documents = self.transcript_loader.load_documents(
            documents_dir, 
            file_pattern=file_pattern
        )
        
        # Validate documents
        validation_issues = validate_documents(documents)
        if validation_issues:
            self.logger.warning(f"Document validation issues: {validation_issues}")
        
        # Deduplicate documents
        documents = self.transcript_loader.deduplicate_documents(documents)
        
        # Get document summary
        summary = self.transcript_loader.get_document_summary(documents)
        self.logger.info(f"Loaded {summary['total_documents']} documents with {summary['total_words']:,} words")
        
        # Chunk documents before adding to vector store
        self.logger.info("Chunking documents...")
        chunks = self.chunk_processor.chunk_documents(documents)
        
        # Get chunk statistics
        chunk_stats = self.chunk_processor.get_chunk_stats(chunks)
        self.logger.info(f"Created {chunk_stats['total_chunks']} chunks from {chunk_stats['unique_parent_documents']} documents")
        self.logger.info(f"Average words per chunk: {chunk_stats['avg_words_per_chunk']:.1f}")
        
        # Add documents to vector store with persona metadata
        self.logger.info(f"Indexing {len(chunks)} chunks in vector store for persona '{persona_id}'...")
        # Add persona_id to all chunks metadata
        for chunk in chunks:
            chunk['persona_id'] = persona_id
        chunks_added = vector_store.add_documents(chunks)
        
        # Get collection statistics
        collection_stats = vector_store.get_collection_stats()
        
        # Update persona stats in registry
        self.persona_manager.update_persona_stats(persona_id, {
            'documents': len(documents),
            'chunks': len(chunks),
            'total_words': summary['total_words']
        })
        
        results = {
            'documents_loaded': len(documents),
            'total_words': summary['total_words'],
            'chunks_created': len(chunks),
            'chunks_indexed': len(chunks_added) if isinstance(chunks_added, list) else chunks_added,
            'collection_stats': collection_stats,
            'document_summary': summary,
            'chunk_stats': chunk_stats
        }
        
        self.logger.info(f"Knowledge base built successfully: {len(chunks)} chunks created and indexed")
        return results
    
    def extract_and_save_persona(self,
                                documents_dir: str,
                                persona_name: str,
                                file_pattern: str = "*.md",
                                use_cached_analysis: bool = True,
                                force_reanalyze: bool = False) -> str:
        """
        Extract persona from documents and save artifacts
        
        Args:
            documents_dir: Directory containing documents
            persona_name: Name for the persona
            file_pattern: Pattern for files to include
            use_cached_analysis: Whether to use cached statistical analysis if available
            force_reanalyze: Force fresh statistical analysis even if cache exists
            
        Returns:
            Path to saved persona artifact
        """
        # Register or get persona ID
        persona_id = self.persona_manager.get_or_create_persona(persona_name)
        
        self.logger.info(f"Extracting persona '{persona_name}' (ID: {persona_id}) from {documents_dir}")
        
        # Use persona manager for artifact operations
        
        # Load documents
        documents = self.transcript_loader.load_documents(
            documents_dir,
            file_pattern=file_pattern
        )
        
        # Validate
        validation_issues = validate_documents(documents)
        if validation_issues:
            self.logger.warning(f"Document validation issues: {validation_issues}")
            if self.settings.validation.strict_mode:
                raise ValueError(f"Validation failed: {validation_issues}")
        
        # Extract persona with cache support
        if use_cached_analysis and not force_reanalyze:
            if self.statistical_analyzer.has_cached_analysis(documents):
                self.logger.info("Using cached analysis for persona extraction...")
            else:
                self.logger.info("No valid cache found, performing fresh analysis...")
        elif force_reanalyze:
            self.logger.info("Force reanalyze enabled, performing fresh analysis...")
        else:
            self.logger.info("Cache disabled, performing fresh analysis...")
        
        persona = self.persona_extractor.extract_persona_sync(
            documents, 
            use_cached_analysis=use_cached_analysis, 
            force_reanalyze=force_reanalyze
        )
        
        # Validate persona
        from ..utils.validation import validate_persona, auto_fix_persona
        
        persona_issues = validate_persona(persona, self.settings)
        if persona_issues:
            self.logger.warning(f"Persona validation issues: {persona_issues}")
            
            # Attempt auto-fix
            if self.settings.validation.auto_fix:
                self.logger.info("Attempting to auto-fix persona issues...")
                persona = auto_fix_persona(persona, self.settings)
                
                # Re-validate
                persona_issues = validate_persona(persona, self.settings)
                if persona_issues and self.settings.validation.strict_mode:
                    raise ValueError(f"Persona validation failed after auto-fix: {persona_issues}")
        
        # Calculate quality scores
        from ..utils.validation import validate_extraction_quality
        quality_scores = validate_extraction_quality(persona)
        self.logger.info(f"Extraction quality scores: {quality_scores}")
        
        # Update metadata with quality scores
        persona.extraction_metadata.quality_scores = quality_scores
        
        # Save persona artifact using persona-specific artifact manager
        self.logger.info(f"Saving persona artifact for '{persona_id}'...")
        artifact_path = self.persona_manager.save_persona_constitution(
            persona,
            persona_name,
            metadata={
                'source_dir': documents_dir,
                'file_pattern': file_pattern,
                'quality_scores': quality_scores
            }
        )
        
        # Log summary
        summary = persona.get_summary()
        self.logger.info(f"Persona extraction complete: {summary}")
        
        return artifact_path
    
    def extract_statistical_analysis_only(self,
                                         documents_dir: str,
                                         persona_name: str,
                                         file_pattern: str = "*.md",
                                         force_reanalyze: bool = False) -> str:
        """
        Phase 1-a: Extract statistical analysis only using spaCy/NLTK
        
        Args:
            documents_dir: Directory containing documents
            persona_name: Name for the persona
            file_pattern: Pattern for files to include
            force_reanalyze: Force fresh analysis even if cache exists
            
        Returns:
            Path to cached statistical analysis
        """
        # Register or get persona ID
        persona_id = self.persona_manager.get_or_create_persona(persona_name)
        
        self.logger.info(f"Phase 1-a: Statistical analysis for '{persona_name}' (ID: {persona_id})")
        
        # Load documents
        documents = self.transcript_loader.load_documents(
            documents_dir,
            file_pattern=file_pattern
        )
        
        # Validate
        from ..utils.validation import validate_documents
        validation_issues = validate_documents(documents)
        if validation_issues:
            self.logger.warning(f"Document validation issues: {validation_issues}")
            if self.settings.validation.strict_mode:
                raise ValueError(f"Validation failed: {validation_issues}")
        
        # Perform statistical analysis only
        self.logger.info("Performing statistical analysis...")
        statistical_report = self.statistical_analyzer.analyze_content(
            documents, 
            use_cache=not force_reanalyze, 
            force_reanalyze=force_reanalyze
        )
        
        # Log analysis summary
        self.logger.info(f"Statistical analysis complete: {len(statistical_report.top_keywords)} keywords, "
                        f"{len(statistical_report.top_collocations)} collocations")
        
        # Return path to cached analysis
        cache_info = self.statistical_analyzer.get_cache_info()
        cache_path = cache_info.get('cache_dir', 'statistical analysis cached')
        
        return cache_path
    
    def extract_llm_analysis_only(self,
                                documents_dir: str,
                                persona_name: str,
                                file_pattern: str = "*.md",
                                use_cached_stats: bool = True) -> str:
        """
        Phase 1-b: Extract persona using LLM map-reduce processing only
        
        Args:
            documents_dir: Directory containing documents
            persona_name: Name for the persona
            file_pattern: Pattern for files to include
            use_cached_stats: Whether to use cached statistical analysis
            
        Returns:
            Path to saved persona artifact
        """
        # Register or get persona ID
        persona_id = self.persona_manager.get_or_create_persona(persona_name)
        
        self.logger.info(f"Phase 1-b: LLM processing for '{persona_name}' (ID: {persona_id})")
        
        # Use persona manager for artifact operations
        
        # Load documents
        documents = self.transcript_loader.load_documents(
            documents_dir,
            file_pattern=file_pattern
        )
        
        # Validate
        from ..utils.validation import validate_documents
        validation_issues = validate_documents(documents)
        if validation_issues:
            self.logger.warning(f"Document validation issues: {validation_issues}")
            if self.settings.validation.strict_mode:
                raise ValueError(f"Validation failed: {validation_issues}")
        
        # Check if statistical analysis is available
        if use_cached_stats:
            if not self.statistical_analyzer.has_cached_analysis(documents):
                raise ValueError(
                    "No cached statistical analysis found. "
                    "Please run Phase 1-a (extract-persona-stats) first, "
                    "or set use_cached_stats=False to perform fresh analysis."
                )
            self.logger.info("Using cached statistical analysis from Phase 1-a")
        else:
            self.logger.info("Performing fresh statistical analysis...")
        
        # Extract persona using LLM processing
        persona = self.persona_extractor.extract_persona_sync(
            documents, 
            use_cached_analysis=use_cached_stats, 
            force_reanalyze=False
        )
        
        # Validate persona
        from ..utils.validation import validate_persona, auto_fix_persona
        
        persona_issues = validate_persona(persona, self.settings)
        if persona_issues:
            self.logger.warning(f"Persona validation issues: {persona_issues}")
            
            # Attempt auto-fix
            if self.settings.validation.auto_fix:
                self.logger.info("Attempting to auto-fix persona issues...")
                persona = auto_fix_persona(persona, self.settings)
                
                # Re-validate
                persona_issues = validate_persona(persona, self.settings)
                if persona_issues and self.settings.validation.strict_mode:
                    raise ValueError(f"Persona validation failed after auto-fix: {persona_issues}")
        
        # Calculate quality scores
        from ..utils.validation import validate_extraction_quality
        quality_scores = validate_extraction_quality(persona)
        self.logger.info(f"Extraction quality scores: {quality_scores}")
        
        # Update metadata with quality scores
        persona.extraction_metadata.quality_scores = quality_scores
        
        # Save persona artifact using persona-specific artifact manager
        self.logger.info(f"Saving persona artifact for '{persona_id}'...")
        artifact_path = self.persona_manager.save_persona_constitution(
            persona,
            persona_name,
            metadata={
                'source_dir': documents_dir,
                'file_pattern': file_pattern,
                'quality_scores': quality_scores,
                'processing_stage': 'Phase 1-b: LLM processing'
            }
        )
        
        # Log summary
        summary = persona.get_summary()
        self.logger.info(f"LLM processing complete: {summary}")
        
        return artifact_path
    
    def update_knowledge_base(self,
                            new_documents_dir: str,
                            persona_id: Optional[str] = None,
                            file_pattern: str = "*.md") -> Dict[str, Any]:
        """
        Update existing knowledge base with new documents for a specific persona
        
        Args:
            new_documents_dir: Directory with new documents
            persona_id: Persona identifier for tenant isolation
            file_pattern: Pattern for files to include
            
        Returns:
            Dictionary with update results
        """
        # Use instance persona_id if not provided
        if persona_id is None:
            persona_id = self.persona_id
        
        if not persona_id:
            raise ValueError("persona_id required for updating knowledge base")
        
        self.logger.info(f"Updating knowledge base for persona '{persona_id}' with documents from: {new_documents_dir}")
        
        # Get persona-specific vector store
        vector_store = self.persona_manager.get_persona_vector_store(persona_id)
        
        # Get current stats
        before_stats = vector_store.get_collection_stats()
        
        # Load new documents
        new_documents = self.transcript_loader.load_documents(
            new_documents_dir,
            file_pattern=file_pattern
        )
        
        # Check for duplicates against existing content
        # This is a simplified check - in production you'd want more sophisticated deduplication
        new_documents = self.transcript_loader.deduplicate_documents(new_documents)
        
        # Chunk new documents
        self.logger.info("Chunking new documents...")
        new_chunks = self.chunk_processor.chunk_documents(new_documents)
        chunk_stats = self.chunk_processor.get_chunk_stats(new_chunks)
        self.logger.info(f"Created {chunk_stats['total_chunks']} new chunks")
        
        # Add to vector store with persona metadata
        for chunk in new_chunks:
            chunk['persona_id'] = persona_id
        chunks_added = vector_store.add_documents(new_chunks)
        
        # Get updated stats
        after_stats = vector_store.get_collection_stats()
        
        # Update persona stats
        self.persona_manager.update_persona_stats(persona_id, {
            'chunks': after_stats.get('total_chunks', 0)
        })
        
        results = {
            'new_documents': len(new_documents),
            'chunks_added': chunks_added,
            'total_chunks_before': before_stats.get('total_chunks', 0),
            'total_chunks_after': after_stats.get('total_chunks', 0),
            'collection_stats': after_stats
        }
        
        self.logger.info(f"Knowledge base updated: {chunks_added} new chunks added")
        return results
    
    def search_knowledge(self, 
                        query: str,
                        persona_id: Optional[str] = None,
                        n_results: int = 10,
                        source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for a specific persona
        
        Args:
            query: Search query
            persona_id: Persona identifier for tenant isolation
            n_results: Number of results
            source_filter: Optional source filter
            
        Returns:
            List of search results
        """
        # Use instance persona_id if not provided
        if persona_id is None:
            persona_id = self.persona_id
        
        if not persona_id:
            raise ValueError("persona_id required for searching knowledge")
        
        self.logger.debug(f"Searching in persona '{persona_id}' for: '{query}'")
        
        # Get persona-specific vector store
        vector_store = self.persona_manager.get_persona_vector_store(persona_id)
        
        if source_filter:
            results = vector_store.search_by_source(query, source_filter, n_results)
        else:
            results = vector_store.search(query, n_results)
        
        self.logger.debug(f"Found {len(results)} results")
        return results
    
    def analyze_knowledge_base(self, persona_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze the current knowledge base for a specific persona
        
        Args:
            persona_id: Persona identifier for tenant isolation
            
        Returns:
            Analysis results
        """
        # Use instance persona_id if not provided
        if persona_id is None:
            persona_id = self.persona_id
        
        if not persona_id:
            raise ValueError("persona_id required for analyzing knowledge base")
        
        self.logger.info(f"Analyzing knowledge base for persona '{persona_id}'...")
        
        # Get persona-specific vector store
        vector_store = self.persona_manager.get_persona_vector_store(persona_id)
        
        # Get collection statistics
        collection_stats = vector_store.get_collection_stats()
        
        # Sample some chunks for analysis
        sample_results = vector_store.search(
            "insights knowledge wisdom learning",
            n_results=100
        )
        
        if sample_results:
            # Combine sample content for statistical analysis
            sample_docs = [
                {'content': r['content'], 'source': r['metadata'].get('source', 'unknown')}
                for r in sample_results
            ]
            
            # Run statistical analysis
            statistical_report = self.statistical_analyzer.analyze_content(sample_docs)
            
            analysis = {
                'collection_stats': collection_stats,
                'sample_analysis': {
                    'total_words': statistical_report.total_words,
                    'total_sentences': statistical_report.total_sentences,
                    'top_keywords': list(statistical_report.top_keywords.keys())[:20],
                    'top_entities': list(statistical_report.top_entities.keys())[:10],
                    'readability': statistical_report.readability_metrics,
                    'sentiment': statistical_report.sentiment_analysis
                }
            }
        else:
            analysis = {
                'collection_stats': collection_stats,
                'sample_analysis': None
            }
        
        self.logger.info("Knowledge base analysis complete")
        return analysis
    
    def export_knowledge_base(self, export_dir: str) -> Dict[str, str]:
        """
        Export knowledge base and personas to a directory
        
        Args:
            export_dir: Directory to export to
            
        Returns:
            Dictionary of export paths
        """
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        
        exports = {}
        
        # Export collection statistics
        stats_path = export_path / "knowledge_base_stats.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(self.vector_store.get_collection_stats(), f, indent=2)
        exports['statistics'] = str(stats_path)
        
        # Export personas
        personas = self.artifact_manager.list_artifacts()
        for persona_info in personas:
            persona_name = persona_info['name']
            
            # Export in multiple formats
            json_path = self.artifact_manager.export_artifact(
                persona_name,
                str(export_path / f"persona_{persona_name}.json"),
                format='json'
            )
            exports[f'persona_{persona_name}_json'] = json_path
            
            md_path = self.artifact_manager.export_artifact(
                persona_name,
                str(export_path / f"persona_{persona_name}.md"),
                format='markdown'
            )
            exports[f'persona_{persona_name}_md'] = md_path
        
        self.logger.info(f"Exported knowledge base to: {export_dir}")
        return exports
    
    # Phase 2 Advanced Retrieval Methods
    
    def setup_phase2_retrieval(self):
        """
        Initialize Phase 2 advanced retrieval components.
        """
        try:
            self.logger.info("Setting up Phase 2 advanced retrieval...")
            
            # Load retrieval configuration
            if hasattr(self.settings, 'retrieval'):
                self.retrieval_config = self.settings.retrieval
            else:
                self.retrieval_config = Phase2RetrievalConfig.from_env()
            
            # Setup storage paths
            base_storage = self.retrieval_config.storage.base_storage_dir
            bm25_path = self.retrieval_config.storage.get_bm25_index_path()
            cache_dir = self.retrieval_config.storage.get_cache_dir()
            
            # Initialize cache
            if self.retrieval_config.caching.enabled:
                self.retrieval_cache = RetrievalCache(
                    str(cache_dir),
                    cache_size=self.retrieval_config.caching.hyde_cache_size,
                    ttl_hours=self.retrieval_config.caching.cache_ttl_hours,
                    enable_compression=self.retrieval_config.caching.enable_compression
                )
                self.logger.debug("Retrieval cache initialized")
            
            # Initialize BM25 store
            if self.retrieval_config.hybrid_search.enabled:
                self.bm25_store = BM25Store(
                    str(bm25_path),
                    k1=self.retrieval_config.hybrid_search.bm25_k1,
                    b=self.retrieval_config.hybrid_search.bm25_b
                )
                self.logger.debug("BM25 store initialized")
            
            self.logger.info("Phase 2 retrieval setup complete")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Phase 2 retrieval: {e}")
            # Disable Phase 2 if setup fails
            if self.retrieval_config:
                self.retrieval_config.enabled = False
    
    def build_phase2_indexes(self, persona_id: Optional[str] = None, rebuild: bool = False):
        """
        Build Phase 2 indexes (BM25) for advanced retrieval.
        
        Args:
            persona_id: Persona identifier
            rebuild: Whether to rebuild existing indexes
        """
        if not self.retrieval_config or not self.retrieval_config.enabled:
            self.logger.warning("Phase 2 not enabled, skipping index building")
            return
        
        persona_id = persona_id or self.persona_id
        if not persona_id:
            raise ValueError("persona_id required for building Phase 2 indexes")
        
        self.logger.info(f"Building Phase 2 indexes for persona '{persona_id}'...")
        
        # Get vector store for persona
        vector_store = self.persona_manager.get_persona_vector_store(persona_id)
        
        # Get all documents from vector store
        try:
            # This is a simplified approach - in practice, you might need to 
            # store document texts separately or retrieve them differently
            collection_stats = vector_store.get_collection_stats()
            
            if collection_stats.get('document_count', 0) == 0:
                self.logger.warning("No documents found in vector store")
                return
            
            # For now, we'll build from the documents directory used during indexing
            # In a production system, you'd store this metadata
            self.logger.info("Phase 2 indexes would be built here")
            self.logger.info("Note: Full implementation requires document text storage")
            
        except Exception as e:
            self.logger.error(f"Failed to build Phase 2 indexes: {e}")
    
    def get_advanced_retrieval_pipeline(self, persona_id: Optional[str] = None) -> Optional[AdvancedRetrievalPipeline]:
        """
        Get the advanced retrieval pipeline for a persona.
        
        Args:
            persona_id: Persona identifier
            
        Returns:
            AdvancedRetrievalPipeline instance or None if not available
        """
        if not self.retrieval_config or not self.retrieval_config.enabled:
            return None
        
        if self.advanced_pipeline is not None:
            return self.advanced_pipeline
        
        try:
            persona_id = persona_id or self.persona_id
            if not persona_id:
                self.logger.warning("No persona_id provided for advanced retrieval")
                return None
            
            # Get persona-specific vector store
            vector_store = self.persona_manager.get_persona_vector_store(persona_id)
            
            # Get embeddings model (assuming it's available from vector store)
            embeddings = vector_store.embedding_function if hasattr(vector_store, 'embedding_function') else None
            
            # Get LLM from persona extractor
            llm = self.persona_extractor.llm if hasattr(self.persona_extractor, 'llm') else None
            
            if not llm or not embeddings:
                self.logger.error("LLM or embeddings not available for HyDE")
                return None
            
            # Initialize HyDE retriever
            if self.retrieval_config.hyde.enabled:
                cache_dir = str(self.retrieval_config.storage.get_cache_dir())
                self.hyde_retriever = HyDERetriever(
                    llm=llm,
                    embeddings=embeddings,
                    vector_store=vector_store,
                    settings=self.settings,
                    cache_dir=cache_dir
                )
            
            # Initialize hybrid retriever
            if self.retrieval_config.hybrid_search.enabled and self.bm25_store:
                self.hybrid_retriever = HybridRetriever(
                    bm25_store=self.bm25_store,
                    vector_store=vector_store,
                    bm25_weight=self.retrieval_config.hybrid_search.bm25_weight,
                    vector_weight=self.retrieval_config.hybrid_search.vector_weight
                )
            
            # Initialize reranker
            if self.retrieval_config.reranking.enabled:
                cache_dir = str(self.retrieval_config.storage.get_cache_dir())
                self.reranker = CrossEncoderReranker(
                    model_name=self.retrieval_config.reranking.model,
                    use_cohere=self.retrieval_config.reranking.use_cohere,
                    cohere_api_key=self.retrieval_config.reranking.cohere_api_key,
                    device=self.retrieval_config.reranking.device,
                    batch_size=self.retrieval_config.reranking.batch_size,
                    cache_dir=cache_dir
                )
            
            # Create advanced pipeline
            if self.hyde_retriever and self.hybrid_retriever and self.reranker:
                cache_dir = str(self.retrieval_config.storage.get_cache_dir())
                self.advanced_pipeline = AdvancedRetrievalPipeline(
                    hyde_retriever=self.hyde_retriever,
                    hybrid_retriever=self.hybrid_retriever,
                    reranker=self.reranker,
                    cache_dir=cache_dir,
                    enable_hyde=self.retrieval_config.hyde.enabled,
                    enable_hybrid=self.retrieval_config.hybrid_search.enabled,
                    enable_reranking=self.retrieval_config.reranking.enabled
                )
                
                self.logger.info("Advanced retrieval pipeline initialized")
                return self.advanced_pipeline
            else:
                self.logger.warning("Could not initialize all pipeline components")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to initialize advanced pipeline: {e}")
            return None
    
    def advanced_search(
        self,
        query: str,
        persona_id: Optional[str] = None,
        k: int = 5,
        use_phase2: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform advanced search using Phase 2 pipeline.
        
        Args:
            query: Search query
            persona_id: Persona identifier
            k: Number of results to return
            use_phase2: Whether to use Phase 2 advanced retrieval
            
        Returns:
            List of search results with metadata
        """
        persona_id = persona_id or self.persona_id
        if not persona_id:
            raise ValueError("persona_id required for search")
        
        self.logger.info(f"Advanced search for query: {query[:100]}...")
        
        # Try Phase 2 pipeline first
        if use_phase2 and self.retrieval_config and self.retrieval_config.enabled:
            pipeline = self.get_advanced_retrieval_pipeline(persona_id)
            if pipeline:
                try:
                    documents = pipeline.retrieve(query, k=k)
                    
                    # Convert to result format
                    results = []
                    for doc in documents:
                        result = {
                            'content': doc.page_content,
                            'metadata': doc.metadata,
                            'retrieval_method': 'phase2_advanced'
                        }
                        results.append(result)
                    
                    self.logger.info(f"Advanced search returned {len(results)} results")
                    return results
                    
                except Exception as e:
                    self.logger.error(f"Phase 2 search failed: {e}")
                    # Fall through to basic search
        
        # Fallback to basic vector search
        self.logger.info("Using fallback basic search")
        vector_store = self.persona_manager.get_persona_vector_store(persona_id)
        
        try:
            documents = vector_store.similarity_search(query, k=k)
            results = []
            for doc in documents:
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'retrieval_method': 'basic_vector'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Basic search also failed: {e}")
            return []
    
    def get_phase2_statistics(self) -> Dict[str, Any]:
        """
        Get Phase 2 retrieval statistics.
        
        Returns:
            Dictionary with Phase 2 statistics
        """
        stats = {
            "phase2_enabled": self.retrieval_config.enabled if self.retrieval_config else False,
            "components_initialized": {}
        }
        
        if self.retrieval_config and self.retrieval_config.enabled:
            stats["configuration"] = self.retrieval_config.to_dict()
            
            # Component initialization status
            stats["components_initialized"] = {
                "bm25_store": self.bm25_store is not None,
                "retrieval_cache": self.retrieval_cache is not None,
                "hyde_retriever": self.hyde_retriever is not None,
                "hybrid_retriever": self.hybrid_retriever is not None,
                "reranker": self.reranker is not None,
                "advanced_pipeline": self.advanced_pipeline is not None
            }
            
            # Get component statistics
            if self.bm25_store:
                stats["bm25_statistics"] = self.bm25_store.get_statistics()
            
            if self.retrieval_cache:
                stats["cache_statistics"] = self.retrieval_cache.get_cache_statistics()
            
            if self.advanced_pipeline:
                stats["pipeline_statistics"] = self.advanced_pipeline.get_statistics()
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'vector_store'):
            self.vector_store.close()
        if hasattr(self, 'persona_manager'):
            self.persona_manager.cleanup()
        
        # Cleanup Phase 2 components
        if self.retrieval_cache:
            self.retrieval_cache.cleanup_expired()
        
        self.logger.info("Knowledge indexer cleanup complete")