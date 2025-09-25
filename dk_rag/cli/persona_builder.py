"""
Command-line interface for building and managing virtual influencer personas
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import json

from ..core.knowledge_indexer import KnowledgeIndexer
from ..core.persona_manager import PersonaManager
from ..config.settings import Settings, get_settings
from ..utils.logging import setup_logger, get_component_logger
from ..utils.validation import validate_config_file


class PersonaBuilderCLI:
    """
    CLI for the Virtual Influencer Persona Agent
    """
    
    def __init__(self):
        """Initialize the CLI"""
        self.settings = None
        self.logger = None
        self.knowledge_indexer = None
        self.persona_manager = None
    
    def setup(self, config_path: Optional[str] = None, debug: bool = False):
        """Setup CLI components"""
        # Setup logging with component-specific prefix
        log_level = "DEBUG" if debug else "INFO"
        # First setup base logger
        setup_logger("persona_cli", level=log_level)
        # Then use component logger
        self.logger = get_component_logger("CLI", "PersonaBuilder")
        
        # Load settings
        if config_path:
            self.logger.info(f"Loading configuration from: {config_path}")
            validation_issues = validate_config_file(config_path)
            if validation_issues:
                self.logger.error(f"Configuration validation failed: {validation_issues}")
                sys.exit(1)
            self.settings = Settings.from_file(config_path)
        else:
            self.logger.info("Using default configuration")
            self.settings = Settings.from_default_config()
        
        # Enable debug mode if requested
        if debug:
            self.settings.development.debug_mode = True
        
        # Initialize components
        self.persona_manager = PersonaManager(self.settings)
        # Note: knowledge_indexer will be initialized per-command with persona context
        self.knowledge_indexer = None
    
    def build_knowledge_base(self, args):
        """Build or rebuild the knowledge base"""
        self.logger.info("=" * 60)
        self.logger.info("BUILDING KNOWLEDGE BASE")
        self.logger.info("=" * 60)
        
        # Register or get persona
        persona_id = self.persona_manager.get_or_create_persona(args.persona_id)
        self.logger.info(f"Using persona: {persona_id}")
        
        # Initialize persona-specific knowledge indexer
        self.knowledge_indexer = KnowledgeIndexer(self.settings, self.persona_manager, persona_id)
        
        results = self.knowledge_indexer.build_knowledge_base(
            documents_dir=args.documents_dir,
            rebuild=args.rebuild,
            file_pattern=args.pattern
        )
        
        # Display results
        print("\nKnowledge Base Built Successfully!")
        print("-" * 40)
        print(f"Documents loaded: {results['documents_loaded']}")
        print(f"Total words: {results['total_words']:,}")
        print(f"Chunks created: {results['chunks_created']}")
        print(f"Total chunks in database: {results['collection_stats'].get('total_chunks', 0)}")
        
        # Display Phase 2 results
        if 'phase2_results' in results and results['phase2_results']:
            phase2 = results['phase2_results']
            print("\nPhase 2 Advanced Retrieval:")
            print("-" * 30)
            if phase2.get('bm25_built'):
                print(f"✅ BM25 Index: Built successfully ({phase2.get('bm25_documents', 0)} documents)")
                print("   Advanced retrieval ready (HyDE + BM25 + Vector + Reranking)")
            else:
                print("❌ BM25 Index: Build failed")
                if 'bm25_error' in phase2:
                    print(f"   Error: {phase2['bm25_error']}")
        
        if args.verbose:
            print("\nDocument Summary:")
            for key, value in results['document_summary'].items():
                print(f"  {key}: {value}")
    
    def extract_persona(self, args):
        """Extract persona from documents"""
        self.logger.info("=" * 60)
        self.logger.info("EXTRACTING PERSONA")
        self.logger.info("=" * 60)
        
        try:
            # Register or get persona
            persona_id = self.persona_manager.get_or_create_persona(args.name)
            self.logger.info(f"Using persona: {persona_id}")
            
            # Initialize persona-specific knowledge indexer
            self.knowledge_indexer = KnowledgeIndexer(self.settings, self.persona_manager, persona_id)
            
            artifact_path = self.knowledge_indexer.extract_and_save_persona(
                documents_dir=args.documents_dir,
                persona_name=args.name,
                file_pattern=args.pattern,
                use_cached_analysis=not getattr(args, 'skip_cache', False),
                force_reanalyze=getattr(args, 'force_reanalyze', False)
            )
            
            print("\nPersona Extracted Successfully!")
            print("-" * 40)
            print(f"Saved to: {artifact_path}")
            
            # Load and display summary
            persona = self.persona_manager.load_persona_constitution(persona_name=args.name)
            summary = persona.get_summary()
            
            print("\nPersona Summary:")
            print(f"  Mental Models: {summary['total_mental_models']}")
            print(f"  Core Beliefs: {summary['total_core_beliefs']}")
            print(f"  Catchphrases: {summary['total_catchphrases']}")
            print(f"  Vocabulary Terms: {summary['total_vocabulary_terms']}")
            print(f"  Processing Time: {summary['processing_time']}")
            
            if args.verbose:
                if hasattr(persona.extraction_metadata, 'quality_scores') and persona.extraction_metadata.quality_scores:
                    print("\nQuality Scores:")
                    for key, value in persona.extraction_metadata.quality_scores.items():
                        print(f"  {key}: {value:.2f}")
                
                print("\nTop Catchphrases:")
                for phrase in persona.linguistic_style.catchphrases[:5]:
                    print(f"  - \"{phrase}\"")
                
                print("\nMental Models:")
                for model in persona.mental_models[:3]:
                    print(f"  - {model.name}")
            
        except Exception as e:
            self.logger.error(f"Persona extraction failed: {e}")
            sys.exit(1)
    
    def extract_persona_stats(self, args):
        """Extract persona - statistical analysis only (Phase 1-a)"""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 1-A: STATISTICAL ANALYSIS")
        self.logger.info("=" * 60)
        
        try:
            # Override batch_size if provided via CLI
            if hasattr(args, 'batch_size') and args.batch_size is not None:
                self.settings.map_reduce_extraction.batch_size = args.batch_size
                self.logger.info(f"Overriding batch_size from CLI: {args.batch_size}")
            
            # Register or get persona
            persona_id = self.persona_manager.get_or_create_persona(args.name)
            self.logger.info(f"Using persona: {persona_id}")
            
            # Initialize persona-specific knowledge indexer
            self.knowledge_indexer = KnowledgeIndexer(self.settings, self.persona_manager, persona_id)
            
            # Perform statistical analysis only
            self.knowledge_indexer.extract_statistical_analysis_only(
                documents_dir=args.documents_dir,
                persona_name=args.name,
                file_pattern=args.pattern,
                force_reanalyze=getattr(args, 'force_reanalyze', False)
            )
            
            print("\nPhase 1-a: Statistical Analysis Complete!")
            print("-" * 40)
            print("Statistical analysis cached for LLM processing stage.")
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {e}")
            sys.exit(1)
    
    def extract_persona_llm(self, args):
        """Extract persona - LLM processing only (Phase 1-b)"""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 1-B: LLM MAP-REDUCE PROCESSING")
        self.logger.info("=" * 60)
        
        try:
            # Override batch_size if provided via CLI
            if hasattr(args, 'batch_size') and args.batch_size is not None:
                self.settings.map_reduce_extraction.batch_size = args.batch_size
                self.logger.info(f"Overriding batch_size from CLI: {args.batch_size}")
            
            # Register or get persona
            persona_id = self.persona_manager.get_or_create_persona(args.name)
            self.logger.info(f"Using persona: {persona_id}")
            
            # Initialize persona-specific knowledge indexer
            self.knowledge_indexer = KnowledgeIndexer(self.settings, self.persona_manager, persona_id)
            
            # Perform LLM processing only
            artifact_path = self.knowledge_indexer.extract_llm_analysis_only(
                documents_dir=args.documents_dir,
                persona_name=args.name,
                file_pattern=args.pattern,
                use_cached_stats=getattr(args, 'use_cached_stats', True)
            )
            
            print("\nPhase 1-b: LLM Processing Complete!")
            print("-" * 40)
            print(f"Saved to: {artifact_path}")
            
            # Load and display summary
            persona = self.persona_manager.load_persona_constitution(persona_name=args.name)
            summary = persona.get_summary()
            
            print("\nPersona Summary:")
            print(f"  Mental Models: {summary['total_mental_models']}")
            print(f"  Core Beliefs: {summary['total_core_beliefs']}")
            print(f"  Catchphrases: {summary['total_catchphrases']}")
            print(f"  Vocabulary Terms: {summary['total_vocabulary_terms']}")
            print(f"  Processing Time: {summary['processing_time']}")
            
            if args.verbose:
                if hasattr(persona.extraction_metadata, 'quality_scores') and persona.extraction_metadata.quality_scores:
                    print("\nQuality Scores:")
                    for key, value in persona.extraction_metadata.quality_scores.items():
                        print(f"  {key}: {value:.2f}")
                
                print("\nTop Catchphrases:")
                for phrase in persona.linguistic_style.catchphrases[:5]:
                    print(f"  - \"{phrase}\"")
                
                print("\nMental Models:")
                for model in persona.mental_models[:3]:
                    print(f"  - {model.name}")
            
        except Exception as e:
            self.logger.error(f"LLM processing failed: {e}")
            sys.exit(1)
    
    def list_personas(self, args):
        """List available personas"""
        personas = self.persona_manager.list_personas()
        
        if not personas:
            print("No personas found")
            return
        
        print("\nAvailable Personas:")
        print("-" * 60)
        
        for persona_info in personas:
            print(f"Name: {persona_info['name']}")
            print(f"  ID: {persona_info['id']}")
            print(f"  Created: {persona_info['created_at']}")
            if persona_info['stats'].get('stats_updated_at'):
                print(f"  Last Updated: {persona_info['stats']['stats_updated_at']}")
            print(f"  Documents: {persona_info['stats']['documents']}")
            print(f"  Chunks: {persona_info['stats']['chunks']}")
            if args.verbose and persona_info.get('metadata'):
                print(f"  Metadata: {persona_info['metadata']}")
            print()
    
    def analyze_knowledge(self, args):
        """Analyze the knowledge base for a specific persona"""
        if not args.persona_id:
            print("Error: --persona-id is required for analysis")
            sys.exit(1)
        
        # Check if persona exists
        persona_id = self.persona_manager._sanitize_persona_id(args.persona_id)
        if not self.persona_manager.persona_exists(persona_id):
            print(f"Error: Persona '{args.persona_id}' not found")
            sys.exit(1)
        
        self.logger.info(f"Analyzing knowledge base for persona '{persona_id}'...")
        
        # Initialize persona-specific knowledge indexer
        self.knowledge_indexer = KnowledgeIndexer(self.settings, self.persona_manager, persona_id)
        
        analysis = self.knowledge_indexer.analyze_knowledge_base(persona_id=persona_id)
        
        print("\nKnowledge Base Analysis:")
        print("=" * 60)
        
        # Collection statistics
        stats = analysis['collection_stats']
        print(f"Total Chunks: {stats.get('total_chunks', 0)}")
        print(f"Unique Sources: {stats.get('unique_sources', 0)}")
        
        if stats.get('sources'):
            print("\nSources:")
            for source in stats['sources'][:10]:
                print(f"  - {source}")
        
        # Sample analysis
        if analysis.get('sample_analysis'):
            sample = analysis['sample_analysis']
            print("\nContent Analysis (sample):")
            print(f"  Total Words: {sample['total_words']:,}")
            print(f"  Total Sentences: {sample['total_sentences']:,}")
            
            if sample.get('readability'):
                print(f"\nReadability Metrics:")
                for key, value in sample['readability'].items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
            
            if sample.get('top_keywords'):
                print(f"\nTop Keywords:")
                for keyword in sample['top_keywords'][:10]:
                    print(f"  - {keyword}")
            
            if sample.get('sentiment'):
                print(f"\nSentiment Analysis:")
                sentiment = sample['sentiment']
                print(f"  Overall: {sentiment.get('overall_sentiment', 0):.2f}")
                print(f"  Positive Ratio: {sentiment.get('positive_ratio', 0):.2%}")
                print(f"  Negative Ratio: {sentiment.get('negative_ratio', 0):.2%}")
    
    def search(self, args):
        """Search the knowledge base for a specific persona"""
        if not args.persona_id:
            print("Error: --persona-id is required for searching")
            sys.exit(1)
        
        # Check if persona exists
        persona_id = self.persona_manager._sanitize_persona_id(args.persona_id)
        if not self.persona_manager.persona_exists(persona_id):
            print(f"Error: Persona '{args.persona_id}' not found")
            sys.exit(1)
        
        # Initialize persona-specific knowledge indexer
        self.knowledge_indexer = KnowledgeIndexer(self.settings, self.persona_manager, persona_id)
        
        results = self.knowledge_indexer.search_knowledge(
            query=args.query,
            persona_id=persona_id,
            n_results=args.n_results,
            source_filter=args.source
        )
        
        if not results:
            print("No results found")
            return
        
        print(f"\nSearch Results for: '{args.query}'")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [Score: {result['similarity_score']:.3f}]")
            print(f"   Source: {result['metadata'].get('source', 'unknown')}")
            
            # Display content preview
            content = result['content']
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"   Content: {content}")
    
    def export(self, args):
        """Export knowledge base and personas"""
        self.logger.info(f"Exporting to: {args.output_dir}")
        
        exports = self.knowledge_indexer.export_knowledge_base(args.output_dir)
        
        print(f"\nExported {len(exports)} files:")
        for name, path in exports.items():
            print(f"  - {name}: {path}")
    
    def validate(self, args):
        """Validate configuration and setup"""
        print("Validating configuration...")
        
        issues = self.settings.validate_configuration()
        
        if not issues:
            print("✓ Configuration is valid")
        else:
            print("✗ Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
        
        # Test components
        print("\nTesting components:")
        
        try:
            # Test persona manager
            personas = self.persona_manager.list_personas()
            print(f"✓ Persona manager: {len(personas)} personas registered")
        except Exception as e:
            print(f"✗ Persona manager error: {e}")
        
        try:
            # Test vector store (if we have any personas)
            personas = self.persona_manager.list_personas()
            if personas:
                first_persona = personas[0]['id']
                test_knowledge_indexer = KnowledgeIndexer(self.settings, self.persona_manager, first_persona)
                stats = test_knowledge_indexer.vector_store.get_collection_stats()
                print(f"✓ Vector store (persona: {first_persona}): {stats.get('total_chunks', 0)} chunks")
            else:
                print("ℹ No personas registered yet")
        except Exception as e:
            print(f"✗ Vector store error: {e}")
        
        print("\n✓ Validation complete")
    
    def manage_cache(self, args):
        """Manage analysis cache"""
        if not args.cache_action:
            print("Error: Cache action required (info, clear)")
            return
        
        if args.cache_action == "info":
            self._show_cache_info(args)
        elif args.cache_action == "clear":
            self._clear_cache(args)
    
    def _show_cache_info(self, args):
        """Show cache information"""
        if args.persona_id:
            # Show info for specific persona
            persona_id = self.persona_manager._sanitize_persona_id(args.persona_id)
            if not self.persona_manager.persona_exists(persona_id):
                print(f"Error: Persona '{args.persona_id}' not found")
                return
            
            # Initialize analyzer with persona context
            from ..core.statistical_analyzer import StatisticalAnalyzer
            analyzer = StatisticalAnalyzer(self.settings, persona_id)
            cache_info = analyzer.get_cache_info()
            
            print(f"\nCache Information for Persona: {args.persona_id}")
            print("=" * 50)
            self._print_cache_info(cache_info)
        else:
            # Show info for all personas
            personas = self.persona_manager.list_personas()
            print(f"\nCache Information for All Personas ({len(personas)} total)")
            print("=" * 60)
            
            for persona in personas:
                persona_id = persona['id']
                from ..core.statistical_analyzer import StatisticalAnalyzer
                analyzer = StatisticalAnalyzer(self.settings, persona_id)
                cache_info = analyzer.get_cache_info()
                
                print(f"\nPersona: {persona['name']} ({persona_id})")
                print("-" * 40)
                self._print_cache_info(cache_info)
    
    def _print_cache_info(self, cache_info):
        """Print formatted cache information"""
        status = cache_info.get('status', 'unknown')
        print(f"Status: {status}")
        
        if status == "available":
            latest = cache_info.get('latest', {})
            files = cache_info.get('files', [])
            
            print(f"Latest Analysis:")
            print(f"  Timestamp: {latest.get('timestamp', 'unknown')}")
            print(f"  Documents: {latest.get('document_count', 0)}")
            print(f"  Total Words: {latest.get('total_words', 0):,}")
            
            if files:
                total_size = sum(f.get('size', 0) for f in files)
                print(f"Cache Files: {len(files)} files ({total_size / 1024:.1f} KB)")
                for file_info in files:
                    size_kb = file_info.get('size', 0) / 1024
                    print(f"  - {file_info.get('file', 'unknown')} ({size_kb:.1f} KB)")
            
            cache_dir = cache_info.get('cache_dir')
            if cache_dir:
                print(f"Cache Directory: {cache_dir}")
        elif status == "no_cache":
            print("No cached analysis found")
        elif status == "error":
            print(f"Error: {cache_info.get('error', 'Unknown error')}")
    
    def _clear_cache(self, args):
        """Clear cache files"""
        if args.persona_id:
            # Clear cache for specific persona
            persona_id = self.persona_manager._sanitize_persona_id(args.persona_id)
            if not self.persona_manager.persona_exists(persona_id):
                print(f"Error: Persona '{args.persona_id}' not found")
                return
            
            from ..core.statistical_analyzer import StatisticalAnalyzer
            analyzer = StatisticalAnalyzer(self.settings, persona_id)
            analyzer.clear_analysis_cache(args.older_than_days)
            
            age_desc = f" older than {args.older_than_days} days" if args.older_than_days else ""
            print(f"Cleared cache{age_desc} for persona: {args.persona_id}")
        else:
            # Clear cache for all personas
            personas = self.persona_manager.list_personas()
            cleared_count = 0
            
            for persona in personas:
                persona_id = persona['id']
                from ..core.statistical_analyzer import StatisticalAnalyzer
                analyzer = StatisticalAnalyzer(self.settings, persona_id)
                analyzer.clear_analysis_cache(args.older_than_days)
                cleared_count += 1
            
            age_desc = f" older than {args.older_than_days} days" if args.older_than_days else ""
            print(f"Cleared cache{age_desc} for {cleared_count} personas")
    
    def run(self):
        """Run the CLI"""
        parser = argparse.ArgumentParser(
            description="Virtual Influencer Persona Agent - Build and manage influencer personas",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Build knowledge base from documents
  python -m dk_rag.cli.persona_builder build-kb --persona-id dan_kennedy --documents-dir /path/to/docs
  
  # Extract persona from documents (uses cached analysis if available)
  python -m dk_rag.cli.persona_builder extract-persona --documents-dir /path/to/docs --name "Dan Kennedy"
  
  # Extract persona with fresh analysis (ignores cache)
  python -m dk_rag.cli.persona_builder extract-persona --documents-dir /path/to/docs --name "Dan Kennedy" --force-reanalyze
  
  # List available personas
  python -m dk_rag.cli.persona_builder list-personas
  
  # Search knowledge base
  python -m dk_rag.cli.persona_builder search "mental models for productivity" --persona-id dan_kennedy
  
  # Analyze knowledge base
  python -m dk_rag.cli.persona_builder analyze --persona-id dan_kennedy
            """
        )
        
        # Global arguments
        parser.add_argument(
            "--config",
            help="Path to configuration file",
            default=None
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode"
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output"
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Build knowledge base
        build_parser = subparsers.add_parser("build-kb", help="Build knowledge base from documents")
        build_parser.add_argument("--persona-id", required=True, help="Persona identifier (e.g., 'dan_kennedy')")
        build_parser.add_argument("--documents-dir", required=True, help="Directory containing documents")
        build_parser.add_argument("--pattern", default="*.md", help="File pattern to match")
        build_parser.add_argument("--rebuild", action="store_true", help="Rebuild from scratch")
        
        # Extract persona
        extract_parser = subparsers.add_parser("extract-persona", help="Extract persona from documents")
        extract_parser.add_argument("--documents-dir", required=True, help="Directory containing documents")
        extract_parser.add_argument("--name", required=True, help="Name for the persona")
        extract_parser.add_argument("--pattern", default="*.md", help="File pattern to match")
        extract_parser.add_argument("--force-reanalyze", action="store_true", 
                                   help="Force fresh statistical analysis even if cache exists")
        extract_parser.add_argument("--skip-cache", action="store_true", 
                                   help="Skip using cached analysis altogether")
        
        # Extract persona - statistical analysis only (Phase 1-a)
        extract_stats_parser = subparsers.add_parser("extract-persona-stats", help="Phase 1-a: Statistical analysis only (spaCy/NLTK)")
        extract_stats_parser.add_argument("--documents-dir", required=True, help="Directory containing documents")
        extract_stats_parser.add_argument("--name", required=True, help="Name for the persona")
        extract_stats_parser.add_argument("--pattern", default="*.md", help="File pattern to match")
        extract_stats_parser.add_argument("--batch-size", type=int, help="Number of documents per batch (overrides config)")
        extract_stats_parser.add_argument("--force-reanalyze", action="store_true", 
                                         help="Force fresh statistical analysis even if cache exists")
        
        # Extract persona - LLM processing only (Phase 1-b)
        extract_llm_parser = subparsers.add_parser("extract-persona-llm", help="Phase 1-b: LLM map-reduce processing only")
        extract_llm_parser.add_argument("--documents-dir", required=True, help="Directory containing documents")
        extract_llm_parser.add_argument("--name", required=True, help="Name for the persona")
        extract_llm_parser.add_argument("--pattern", default="*.md", help="File pattern to match")
        extract_llm_parser.add_argument("--batch-size", type=int, help="Number of documents per batch (overrides config)")
        extract_llm_parser.add_argument("--use-cached-stats", action="store_true", default=True,
                                       help="Use cached statistical analysis from Phase 1-a")
        
        # List personas
        list_parser = subparsers.add_parser("list-personas", help="List available personas")
        list_parser.add_argument("--name", help="Filter by persona name")
        
        # Analyze knowledge base
        analyze_parser = subparsers.add_parser("analyze", help="Analyze knowledge base")
        analyze_parser.add_argument("--persona-id", required=True, help="Persona identifier")
        
        # Search
        search_parser = subparsers.add_parser("search", help="Search knowledge base")
        search_parser.add_argument("query", help="Search query")
        search_parser.add_argument("--persona-id", required=True, help="Persona identifier")
        search_parser.add_argument("--n-results", type=int, default=10, help="Number of results")
        search_parser.add_argument("--source", help="Filter by source")
        
        # Export
        export_parser = subparsers.add_parser("export", help="Export knowledge base and personas")
        export_parser.add_argument("--output-dir", required=True, help="Output directory")
        export_parser.add_argument("--persona-id", help="Export specific persona only")
        
        # Validate
        validate_parser = subparsers.add_parser("validate", help="Validate configuration and setup")
        
        # Cache management
        cache_parser = subparsers.add_parser("cache", help="Manage analysis cache")
        cache_subparsers = cache_parser.add_subparsers(dest="cache_action", help="Cache management actions")
        
        # Cache info
        cache_info_parser = cache_subparsers.add_parser("info", help="Show cache information")
        cache_info_parser.add_argument("--persona-id", help="Show cache info for specific persona")
        
        # Cache clear
        cache_clear_parser = cache_subparsers.add_parser("clear", help="Clear cache files")
        cache_clear_parser.add_argument("--persona-id", help="Clear cache for specific persona")
        cache_clear_parser.add_argument("--older-than-days", type=int, help="Clear files older than N days")
        
        # Parse arguments
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            sys.exit(1)
        
        # Setup CLI
        self.setup(config_path=args.config, debug=args.debug)
        
        # Execute command
        commands = {
            "build-kb": self.build_knowledge_base,
            "extract-persona": self.extract_persona,
            "extract-persona-stats": self.extract_persona_stats,
            "extract-persona-llm": self.extract_persona_llm,
            "list-personas": self.list_personas,
            "analyze": self.analyze_knowledge,
            "search": self.search,
            "export": self.export,
            "validate": self.validate,
            "cache": self.manage_cache
        }
        
        command_func = commands.get(args.command)
        if command_func:
            try:
                command_func(args)
            except KeyboardInterrupt:
                self.logger.info("\nOperation cancelled by user")
                sys.exit(0)
            except Exception as e:
                self.logger.error(f"Command failed: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                sys.exit(1)
            finally:
                if self.knowledge_indexer:
                    self.knowledge_indexer.cleanup()
                if self.persona_manager:
                    self.persona_manager.cleanup()
        else:
            parser.print_help()
            sys.exit(1)


def main():
    """Main entry point for CLI"""
    cli = PersonaBuilderCLI()
    cli.run()


if __name__ == "__main__":
    main()