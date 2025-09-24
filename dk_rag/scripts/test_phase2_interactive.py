#!/usr/bin/env python3
"""
Phase 2 Advanced Retrieval Interactive Testing Script

This script provides comprehensive testing for Phase 2 advanced retrieval features:
- HyDE (Hypothetical Document Embeddings)
- Hybrid Search (BM25 + Vector)
- Cross-Encoder Reranking

Usage:
    python scripts/test_phase2_interactive.py --persona-id PERSONA_NAME [--config CONFIG_FILE]
    
Examples:
    python scripts/test_phase2_interactive.py --persona-id greg_startup
    python scripts/test_phase2_interactive.py --persona-id dan_kennedy --config ./config/persona_config.yaml
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dk_rag.config.settings import Settings
from dk_rag.core.persona_manager import PersonaManager
from dk_rag.core.knowledge_indexer import KnowledgeIndexer
from dk_rag.data.storage.langchain_vector_store import LangChainVectorStore as VectorStore
from dk_rag.utils.logging import get_logger


class Phase2Tester:
    """Interactive tester for Phase 2 advanced retrieval features"""
    
    def __init__(self, persona_id: str, config_file: str):
        """
        Initialize Phase 2 tester with persona and configuration
        
        Args:
            persona_id: Persona identifier to test with
            config_file: Path to configuration file
        """
        self.persona_id = persona_id
        self.config_file = config_file
        self.logger = get_logger(__name__)
        
        # Color codes for terminal output
        self.COLORS = {
            'HEADER': '\033[95m',
            'BLUE': '\033[94m',
            'CYAN': '\033[96m',
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m',
            'RED': '\033[91m',
            'BOLD': '\033[1m',
            'UNDERLINE': '\033[4m',
            'END': '\033[0m'
        }
        
        # Initialize components
        self._setup_components()
    
    def _setup_components(self):
        """Initialize all required components and verify Phase 2 readiness"""
        try:
            # Load settings
            self.print_header("Loading Configuration")
            self.settings = Settings.from_file(self.config_file)
            print(f"✓ Configuration loaded from: {self.config_file}")
            
            # Initialize persona manager
            self.persona_manager = PersonaManager(self.settings)
            
            # Verify persona exists
            if not self.persona_manager.persona_exists(self.persona_id):
                available = self.persona_manager.list_personas()
                raise ValueError(f"Persona '{self.persona_id}' not found. Available: {[p['name'] for p in available]}")
            
            print(f"✓ Persona verified: {self.persona_id}")
            
            # Initialize knowledge indexer with Phase 2
            self.knowledge_indexer = KnowledgeIndexer(self.settings, self.persona_manager, self.persona_id)
            
            # Verify Phase 2 is enabled
            if not (hasattr(self.settings, 'retrieval') and getattr(self.settings.retrieval, 'enabled', False)):
                raise ValueError("Phase 2 retrieval not enabled in configuration")
            
            print(f"✓ Phase 2 retrieval enabled")
            
            # Get vector store
            self.vector_store = self.persona_manager.get_persona_vector_store(self.persona_id)
            
            # Get Phase 2 pipeline
            self.advanced_pipeline = self.knowledge_indexer.get_advanced_retrieval_pipeline(self.persona_id)
            
            if not self.advanced_pipeline:
                raise ValueError("Phase 2 pipeline not available - ensure knowledge base is built")
            
            print(f"✓ Advanced retrieval pipeline ready")
            
            # Verify knowledge base statistics
            stats = self.vector_store.get_collection_stats()
            print(f"✓ Knowledge base ready: {stats.get('total_chunks', 0)} chunks")
            
            if stats.get('total_chunks', 0) == 0:
                raise ValueError("Knowledge base is empty - run 'make build-kb' first")
            
        except Exception as e:
            self.print_error(f"Setup failed: {e}")
            sys.exit(1)
    
    def print_header(self, text: str):
        """Print a colored header"""
        print(f"\n{self.COLORS['BOLD']}{self.COLORS['BLUE']}{'=' * 60}{self.COLORS['END']}")
        print(f"{self.COLORS['BOLD']}{self.COLORS['BLUE']}{text.center(60)}{self.COLORS['END']}")
        print(f"{self.COLORS['BOLD']}{self.COLORS['BLUE']}{'=' * 60}{self.COLORS['END']}")
    
    def print_subheader(self, text: str):
        """Print a colored subheader"""
        print(f"\n{self.COLORS['BOLD']}{self.COLORS['CYAN']}{'-' * 40}{self.COLORS['END']}")
        print(f"{self.COLORS['BOLD']}{self.COLORS['CYAN']}{text}{self.COLORS['END']}")
        print(f"{self.COLORS['BOLD']}{self.COLORS['CYAN']}{'-' * 40}{self.COLORS['END']}")
    
    def print_success(self, text: str):
        """Print success message"""
        print(f"{self.COLORS['GREEN']}✓ {text}{self.COLORS['END']}")
    
    def print_error(self, text: str):
        """Print error message"""
        print(f"{self.COLORS['RED']}✗ {text}{self.COLORS['END']}")
    
    def print_info(self, text: str):
        """Print info message"""
        print(f"{self.COLORS['YELLOW']}ℹ {text}{self.COLORS['END']}")
    
    def print_results(self, results: List[Union[Dict[str, Any], Any]], title: str = "Results"):
        """Print search results in a formatted way"""
        self.print_subheader(title)
        
        if not results:
            self.print_info("No results found")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n{self.COLORS['BOLD']}{i}. {self.COLORS['END']}", end="")
            
            # Handle (Document, score) tuples from retrieve_with_scores()
            if isinstance(result, tuple) and len(result) == 2:
                doc, score = result
                content = doc.page_content[:200] + "..."
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            # Handle (doc_id, score, doc_text) tuples from BM25 with return_docs=True
            elif isinstance(result, tuple) and len(result) == 3:
                doc_id, score, doc_text = result
                content = doc_text[:200] + "..."
                metadata = {'doc_id': doc_id}
            # Handle Document objects
            elif hasattr(result, 'page_content'):
                content = result.page_content[:200] + "..."
                metadata = result.metadata if hasattr(result, 'metadata') else {}
                score = metadata.get('similarity_score', 'N/A')
            # Handle dictionary results
            else:
                content = result.get('content', result.get('document', ''))[:200] + "..."
                metadata = result.get('metadata', {})
                score = result.get('score', result.get('distance', 'N/A'))
            
            print(f"{self.COLORS['CYAN']}{content}{self.COLORS['END']}")
            print(f"   Score: {score}")
    
    def test_hyde_retrieval(self, query: str):
        """Test HyDE (Hypothetical Document Embeddings) retrieval"""
        self.print_header("Testing HyDE Retrieval")
        
        print(f"{self.COLORS['BOLD']}Query:{self.COLORS['END']} {query}")
        
        try:
            start_time = time.time()
            
            # Get HyDE retriever
            hyde_retriever = self.advanced_pipeline.hyde
            
            if not hyde_retriever:
                self.print_error("HyDE retriever not available")
                return
            
            # Perform HyDE search (this will generate hypothesis internally)
            print(f"\n{self.COLORS['YELLOW']}Searching with HyDE-enhanced query...{self.COLORS['END']}")
            results = hyde_retriever.retrieve_with_scores(query, k=5)
            
            elapsed = time.time() - start_time
            
            self.print_success(f"HyDE search completed in {elapsed:.2f}s")
            self.print_results(results, "HyDE Enhanced Results")
            
        except Exception as e:
            self.print_error(f"HyDE test failed: {e}")
    
    def test_hybrid_search(self, query: str):
        """Test Hybrid Search (BM25 + Vector) with score breakdown"""
        self.print_header("Testing Hybrid Search (BM25 + Vector)")
        
        print(f"{self.COLORS['BOLD']}Query:{self.COLORS['END']} {query}")
        
        try:
            start_time = time.time()
            
            # Get hybrid retriever
            hybrid_retriever = self.advanced_pipeline.hybrid
            
            if not hybrid_retriever:
                self.print_error("Hybrid retriever not available")
                return
            
            # Perform hybrid search
            print(f"\n{self.COLORS['YELLOW']}Performing BM25 + Vector hybrid search...{self.COLORS['END']}")
            results = hybrid_retriever.search_with_scores(query, k=5)
            
            elapsed = time.time() - start_time
            
            # Also get individual BM25 and vector results for comparison
            print(f"\n{self.COLORS['YELLOW']}Getting BM25-only results...{self.COLORS['END']}")
            bm25_results = hybrid_retriever.bm25_store.search(query, k=5, return_docs=True)
            
            print(f"\n{self.COLORS['YELLOW']}Getting Vector-only results...{self.COLORS['END']}")
            vector_results = hybrid_retriever.vector_store.similarity_search_with_score(query, k=5)
            
            self.print_success(f"Hybrid search completed in {elapsed:.2f}s")
            
            # Display comparison
            self.print_results(bm25_results, "BM25-Only Results")
            self.print_results(vector_results, "Vector-Only Results") 
            self.print_results(results, "Hybrid Fusion Results")
            
            # Show fusion weights
            print(f"\n{self.COLORS['BOLD']}Fusion Configuration:{self.COLORS['END']}")
            print(f"BM25 Weight: {hybrid_retriever.bm25_weight}")
            print(f"Vector Weight: {hybrid_retriever.vector_weight}")
            
        except Exception as e:
            self.print_error(f"Hybrid search test failed: {e}")
    
    def test_reranking(self, query: str):
        """Test Cross-Encoder Reranking with before/after comparison"""
        self.print_header("Testing Cross-Encoder Reranking")
        
        print(f"{self.COLORS['BOLD']}Query:{self.COLORS['END']} {query}")
        
        try:
            start_time = time.time()
            
            # Get reranker
            reranker = self.advanced_pipeline.reranker
            
            if not reranker:
                self.print_error("Reranker not available")
                return
            
            # First get initial results (before reranking)
            print(f"\n{self.COLORS['YELLOW']}Getting initial retrieval candidates...{self.COLORS['END']}")
            
            # Use hybrid search to get candidates with scores (get more candidates for better reranking)
            hybrid_results_with_scores = self.advanced_pipeline.hybrid.search_with_scores(query, k=25)
            
            print(f"Retrieved {len(hybrid_results_with_scores)} candidates for reranking")
            
            # Extract documents for reranker (reranker expects List[Document], not List[Tuple[Document, float]])
            hybrid_docs = [doc for doc, score in hybrid_results_with_scores]
            
            # Perform reranking
            print(f"\n{self.COLORS['YELLOW']}Reranking with cross-encoder...{self.COLORS['END']}")
            
            reranked_results = reranker.rerank(query, hybrid_docs, top_k=5, return_scores=True)
            
            elapsed = time.time() - start_time
            
            self.print_success(f"Reranking completed in {elapsed:.2f}s")
            
            # Show before/after comparison
            self.print_results(hybrid_results_with_scores[:5], "Before Reranking (Top 5)")
            self.print_results(reranked_results, "After Reranking")
            
            # Show reranking model info
            print(f"\n{self.COLORS['BOLD']}Reranking Model:{self.COLORS['END']}")
            print(f"Model: {reranker.model_name}")
            print(f"Device: {reranker.device}")
            
        except Exception as e:
            self.print_error(f"Reranking test failed: {e}")
    
    def interactive_search(self):
        """Interactive search interface using full Phase 2 pipeline"""
        self.print_header("Interactive Search - Full Phase 2 Pipeline")
        
        print(f"{self.COLORS['BOLD']}Full Pipeline Includes:{self.COLORS['END']}")
        print("• HyDE query expansion")
        print("• Hybrid search (BM25 + Vector)")  
        print("• Cross-encoder reranking")
        print("\nEnter your search queries (type 'quit' to exit)")
        
        while True:
            try:
                # Get user input
                query = input(f"\n{self.COLORS['BOLD']}Search> {self.COLORS['END']}").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                # Perform full pipeline search
                start_time = time.time()
                
                print(f"\n{self.COLORS['YELLOW']}Processing with full Phase 2 pipeline...{self.COLORS['END']}")
                
                results = self.advanced_pipeline.retrieve(
                    query=query,
                    k=5,
                    use_hyde=True,
                    use_hybrid=True,
                    use_reranking=True,
                    return_scores=True
                )
                
                elapsed = time.time() - start_time
                
                self.print_success(f"Search completed in {elapsed:.2f}s")
                self.print_results(results, "Phase 2 Pipeline Results")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.print_error(f"Search failed: {e}")
        
        print(f"\n{self.COLORS['GREEN']}Interactive search session ended{self.COLORS['END']}")
    
    def compare_phase1_vs_phase2(self, query: str):
        """Compare Phase 1 (vector-only) vs Phase 2 (full pipeline) results"""
        self.print_header("Phase 1 vs Phase 2 Comparison")
        
        print(f"{self.COLORS['BOLD']}Query:{self.COLORS['END']} {query}")
        
        try:
            # Phase 1: Vector-only search
            print(f"\n{self.COLORS['YELLOW']}Phase 1: Vector-only search...{self.COLORS['END']}")
            start_time = time.time()
            
            phase1_results = self.vector_store.similarity_search_with_score(query, k=5)
            phase1_time = time.time() - start_time
            
            # Phase 2: Full pipeline
            print(f"\n{self.COLORS['YELLOW']}Phase 2: Full advanced pipeline...{self.COLORS['END']}")
            start_time = time.time()
            
            phase2_results = self.advanced_pipeline.retrieve(
                query=query,
                k=5,
                use_hyde=True,
                use_hybrid=True,
                use_reranking=True,
                return_scores=True
            )
            phase2_time = time.time() - start_time
            
            # Display comparison
            print(f"\n{self.COLORS['BOLD']}Performance Comparison:{self.COLORS['END']}")
            print(f"Phase 1 Time: {phase1_time:.2f}s")
            print(f"Phase 2 Time: {phase2_time:.2f}s")
            print(f"Overhead: {(phase2_time - phase1_time):.2f}s ({((phase2_time/phase1_time - 1) * 100):.1f}%)")
            
            self.print_results(phase1_results, "Phase 1 Results (Vector Only)")
            self.print_results(phase2_results, "Phase 2 Results (Full Pipeline)")
            
        except Exception as e:
            self.print_error(f"Comparison failed: {e}")
    
    def run_main_menu(self):
        """Run the main interactive menu"""
        self.print_header(f"Phase 2 Testing - Persona: {self.persona_id}")
        
        while True:
            print(f"\n{self.COLORS['BOLD']}Available Tests:{self.COLORS['END']}")
            print("1. Test HyDE Retrieval")
            print("2. Test Hybrid Search (BM25+Vector)")
            print("3. Test Cross-Encoder Reranking")
            print("4. Interactive Search (Full Pipeline)")
            print("5. Compare Phase 1 vs Phase 2")
            print("6. Exit")
            
            try:
                choice = input(f"\n{self.COLORS['BOLD']}Select option (1-6): {self.COLORS['END']}").strip()
                
                if choice == '6':
                    break
                elif choice == '4':
                    self.interactive_search()
                elif choice in ['1', '2', '3', '5']:
                    query = input(f"\n{self.COLORS['BOLD']}Enter search query: {self.COLORS['END']}").strip()
                    if not query:
                        self.print_info("Query cannot be empty")
                        continue
                    
                    if choice == '1':
                        self.test_hyde_retrieval(query)
                    elif choice == '2':
                        self.test_hybrid_search(query)
                    elif choice == '3':
                        self.test_reranking(query)
                    elif choice == '5':
                        self.compare_phase1_vs_phase2(query)
                else:
                    self.print_info("Invalid option. Please select 1-6.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.print_error(f"Error: {e}")
        
        print(f"\n{self.COLORS['GREEN']}Phase 2 testing session ended{self.COLORS['END']}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Interactive testing for Phase 2 advanced retrieval features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/test_phase2_interactive.py --persona-id greg_startup
    python scripts/test_phase2_interactive.py --persona-id dan_kennedy --config ./config/persona_config.yaml
        """
    )
    
    parser.add_argument(
        '--persona-id',
        required=True,
        help='Persona identifier to test with'
    )
    
    parser.add_argument(
        '--config',
        default='./config/persona_config.yaml',
        help='Path to configuration file (default: ./config/persona_config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Verify config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Initialize and run tester
    try:
        tester = Phase2Tester(args.persona_id, str(config_path))
        tester.run_main_menu()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()