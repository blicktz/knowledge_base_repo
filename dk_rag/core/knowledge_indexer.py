"""
Knowledge indexer for building and managing the vector knowledge base
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..data.storage.vector_store import VectorStore
from ..data.storage.artifacts import ArtifactManager
from ..data.processing.transcript_loader import TranscriptLoader
from ..core.persona_extractor import PersonaExtractor
from ..core.statistical_analyzer import StatisticalAnalyzer
from ..config.settings import Settings
from ..utils.logging import get_logger
from ..utils.validation import validate_documents


class KnowledgeIndexer:
    """
    Orchestrates the complete knowledge indexing and persona extraction pipeline
    """
    
    def __init__(self, settings: Settings):
        """Initialize the knowledge indexer"""
        self.settings = settings
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.vector_store = VectorStore(settings)
        self.artifact_manager = ArtifactManager(settings)
        self.transcript_loader = TranscriptLoader(settings)
        self.persona_extractor = PersonaExtractor(settings)
        self.statistical_analyzer = StatisticalAnalyzer(settings)
    
    def build_knowledge_base(self, 
                           documents_dir: str,
                           rebuild: bool = False,
                           file_pattern: str = "*.md") -> Dict[str, Any]:
        """
        Build complete knowledge base from documents
        
        Args:
            documents_dir: Directory containing documents
            rebuild: Whether to rebuild from scratch
            file_pattern: Pattern for files to include
            
        Returns:
            Dictionary with indexing results
        """
        self.logger.info(f"Building knowledge base from: {documents_dir}")
        
        # Clear existing data if rebuilding
        if rebuild:
            self.logger.info("Clearing existing vector store...")
            self.vector_store.clear_collection()
        
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
        
        # Add documents to vector store
        self.logger.info("Indexing documents in vector store...")
        chunks_added = self.vector_store.add_documents(documents)
        
        # Get collection statistics
        collection_stats = self.vector_store.get_collection_stats()
        
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
                                file_pattern: str = "*.md") -> str:
        """
        Extract persona from documents and save artifacts
        
        Args:
            documents_dir: Directory containing documents
            persona_name: Name for the persona
            file_pattern: Pattern for files to include
            
        Returns:
            Path to saved persona artifact
        """
        self.logger.info(f"Extracting persona '{persona_name}' from {documents_dir}")
        
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
        
        # Extract persona
        self.logger.info("Extracting persona constitution...")
        persona = self.persona_extractor.extract_persona_sync(documents)
        
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
        
        # Save persona artifact
        self.logger.info("Saving persona artifact...")
        artifact_path = self.artifact_manager.save_persona_constitution(
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
                            file_pattern: str = "*.md") -> Dict[str, Any]:
        """
        Update existing knowledge base with new documents
        
        Args:
            new_documents_dir: Directory with new documents
            file_pattern: Pattern for files to include
            
        Returns:
            Dictionary with update results
        """
        self.logger.info(f"Updating knowledge base with documents from: {new_documents_dir}")
        
        # Get current stats
        before_stats = self.vector_store.get_collection_stats()
        
        # Load new documents
        new_documents = self.transcript_loader.load_documents(
            new_documents_dir,
            file_pattern=file_pattern
        )
        
        # Check for duplicates against existing content
        # This is a simplified check - in production you'd want more sophisticated deduplication
        new_documents = self.transcript_loader.deduplicate_documents(new_documents)
        
        # Add to vector store
        chunks_added = self.vector_store.add_documents(new_documents)
        
        # Get updated stats
        after_stats = self.vector_store.get_collection_stats()
        
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
                        n_results: int = 10,
                        source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search the knowledge base
        
        Args:
            query: Search query
            n_results: Number of results
            source_filter: Optional source filter
            
        Returns:
            List of search results
        """
        self.logger.debug(f"Searching for: '{query}'")
        
        if source_filter:
            results = self.vector_store.search_by_source(query, source_filter, n_results)
        else:
            results = self.vector_store.search(query, n_results)
        
        self.logger.debug(f"Found {len(results)} results")
        return results
    
    def analyze_knowledge_base(self) -> Dict[str, Any]:
        """
        Analyze the current knowledge base
        
        Returns:
            Analysis results
        """
        self.logger.info("Analyzing knowledge base...")
        
        # Get collection statistics
        collection_stats = self.vector_store.get_collection_stats()
        
        # Sample some chunks for analysis
        sample_results = self.vector_store.search(
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
        self.vector_store.close()
        self.logger.info("Knowledge indexer cleanup complete")