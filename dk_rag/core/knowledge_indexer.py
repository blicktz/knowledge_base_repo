"""
Knowledge indexer for building and managing the vector knowledge base
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from typing import List, Dict, Any, Optional
from pathlib import Path

from ..data.storage.vector_store import VectorStore
from ..data.storage.artifacts import ArtifactManager
from ..data.processing.transcript_loader import TranscriptLoader
from ..core.persona_extractor import PersonaExtractor
from ..core.statistical_analyzer import StatisticalAnalyzer
from ..core.persona_manager import PersonaManager
from ..config.settings import Settings
from ..utils.logging import get_logger
from ..utils.validation import validate_documents


class KnowledgeIndexer:
    """
    Orchestrates the complete knowledge indexing and persona extraction pipeline
    with multi-tenant support for isolated persona management
    """
    
    def __init__(self, settings: Settings, persona_id: Optional[str] = None):
        """Initialize the knowledge indexer with optional persona context"""
        self.settings = settings
        self.logger = get_logger(__name__)
        self.persona_id = persona_id
        
        # Initialize persona manager
        self.persona_manager = PersonaManager(settings)
        
        # Initialize components - persona-specific if persona_id provided
        if persona_id:
            # Ensure persona is registered
            if not self.persona_manager.persona_exists(persona_id):
                raise ValueError(f"Persona '{persona_id}' not registered. Register it first.")
            
            # Use persona-specific components
            self.vector_store = self.persona_manager.get_persona_vector_store(persona_id)
            self.artifact_manager = self.persona_manager.get_persona_artifact_manager(persona_id)
        else:
            # Legacy mode - use global components
            self.vector_store = VectorStore(settings)
            self.artifact_manager = ArtifactManager(settings)
        
        self.transcript_loader = TranscriptLoader(settings)
        self.persona_extractor = PersonaExtractor(settings, persona_id)
        self.statistical_analyzer = StatisticalAnalyzer(settings, persona_id)
    
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
        
        # Add documents to vector store with persona metadata
        self.logger.info(f"Indexing documents in vector store for persona '{persona_id}'...")
        # Add persona_id to all documents metadata
        for doc in documents:
            doc['persona_id'] = persona_id
        chunks_added = vector_store.add_documents(documents)
        
        # Get collection statistics
        collection_stats = vector_store.get_collection_stats()
        
        # Update persona stats in registry
        self.persona_manager.update_persona_stats(persona_id, {
            'documents': len(documents),
            'chunks': chunks_added,
            'total_words': summary['total_words']
        })
        
        results = {
            'documents_loaded': len(documents),
            'total_words': summary['total_words'],
            'chunks_created': chunks_added,
            'collection_stats': collection_stats,
            'document_summary': summary
        }
        
        self.logger.info(f"Knowledge base built successfully: {chunks_added} chunks indexed")
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
        
        # Get persona-specific artifact manager
        artifact_manager = self.persona_manager.get_persona_artifact_manager(persona_id)
        
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
        artifact_path = artifact_manager.save_persona_constitution(
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
        
        # Add to vector store with persona metadata
        for doc in new_documents:
            doc['persona_id'] = persona_id
        chunks_added = vector_store.add_documents(new_documents)
        
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
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'vector_store'):
            self.vector_store.close()
        if hasattr(self, 'persona_manager'):
            self.persona_manager.cleanup()
        self.logger.info("Knowledge indexer cleanup complete")