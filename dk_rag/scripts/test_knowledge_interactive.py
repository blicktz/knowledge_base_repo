#!/usr/bin/env python3
"""
Mental Models & Core Beliefs Interactive Testing Script

This script provides comprehensive testing for mental models and core beliefs 
knowledge bases with vector search and reranking capabilities:
- Vector similarity search for both knowledge types
- Cross-encoder reranking with before/after comparison  
- Complete search pipeline with performance metrics
- Interactive search sessions with real-time results

Usage:
    python scripts/test_knowledge_interactive.py --persona-id PERSONA_NAME [--config CONFIG_FILE]
    
Examples:
    python scripts/test_knowledge_interactive.py --persona-id greg_startup
    python scripts/test_knowledge_interactive.py --persona-id dan_kennedy --config ./config/persona_config.yaml
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dk_rag.config.settings import Settings
from dk_rag.core.persona_manager import PersonaManager
from dk_rag.core.knowledge_indexer import KnowledgeIndexer
from dk_rag.data.storage.mental_models_store import MentalModelsStore
from dk_rag.data.storage.core_beliefs_store import CoreBeliefsStore
from dk_rag.models.knowledge_results import MentalModelResult, CoreBeliefResult
from dk_rag.utils.logging import get_logger


class KnowledgeTester:
    """Interactive tester for mental models and core beliefs knowledge bases"""
    
    def __init__(self, persona_id: str, config_file: str):
        """
        Initialize knowledge tester with persona and configuration
        
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
        """Initialize all required components and verify knowledge bases"""
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
            
            # Initialize knowledge indexer
            self.knowledge_indexer = KnowledgeIndexer(self.settings, self.persona_manager, self.persona_id)
            
            # Initialize separate knowledge stores
            self.mental_models_store = MentalModelsStore(
                settings=self.settings,
                persona_id=self.persona_id
            )
            
            self.core_beliefs_store = CoreBeliefsStore(
                settings=self.settings,
                persona_id=self.persona_id
            )
            
            # Verify knowledge bases exist and have data
            self._verify_knowledge_bases()
            
            print(f"✓ Knowledge stores initialized successfully")
            
        except Exception as e:
            self.print_error(f"Setup failed: {e}")
            sys.exit(1)
    
    def _verify_knowledge_bases(self):
        """Verify that both knowledge bases exist and contain data"""
        self.print_header("Verifying Knowledge Bases")
        
        # Check mental models
        if not self.mental_models_store.collection_exists():
            raise ValueError(f"Mental models knowledge base not found for persona '{self.persona_id}'. Run: make build-mental-models")
        
        mm_stats = self.mental_models_store.get_collection_stats()
        mm_count = mm_stats.get('current_document_count', 0)
        print(f"✓ Mental Models: {mm_count} documents, {mm_stats.get('directory_size_mb', 'unknown')} MB")
        
        # Check core beliefs  
        if not self.core_beliefs_store.collection_exists():
            raise ValueError(f"Core beliefs knowledge base not found for persona '{self.persona_id}'. Run: make build-core-beliefs")
        
        cb_stats = self.core_beliefs_store.get_collection_stats()
        cb_count = cb_stats.get('current_document_count', 0)
        print(f"✓ Core Beliefs: {cb_count} documents, {cb_stats.get('directory_size_mb', 'unknown')} MB")
        
        if mm_count == 0 and cb_count == 0:
            raise ValueError("Both knowledge bases are empty. Please rebuild them.")
    
    def print_header(self, title: str):
        """Print a formatted header"""
        print(f"\n{self.COLORS['HEADER']}{'='*60}{self.COLORS['END']}")
        print(f"{self.COLORS['HEADER']}{title.center(60)}{self.COLORS['END']}")
        print(f"{self.COLORS['HEADER']}{'='*60}{self.COLORS['END']}")
    
    def print_success(self, message: str):
        """Print a success message"""
        print(f"{self.COLORS['GREEN']}✓ {message}{self.COLORS['END']}")
    
    def print_error(self, message: str):
        """Print an error message"""
        print(f"{self.COLORS['RED']}✗ {message}{self.COLORS['END']}")
    
    def print_results(self, results: List[Any], title: str):
        """Print search results in a formatted way"""
        print(f"\n{self.COLORS['CYAN']}{title}{self.COLORS['END']}")
        print(f"{self.COLORS['CYAN']}{'-'*len(title)}{self.COLORS['END']}")
        
        if not results:
            print("No results found.")
            return
        
        for i, result in enumerate(results, 1):
            # Handle both scored and unscored results
            if isinstance(result, tuple):
                item, score = result
                score_display = f"{score:.3f}" if score is not None else "N/A"
                print(f"\n{self.COLORS['BOLD']}{i}. [Score: {score_display}]{self.COLORS['END']}")
            else:
                item = result
                print(f"\n{self.COLORS['BOLD']}{i}.{self.COLORS['END']}")
            
            # Format based on result type
            if isinstance(item, MentalModelResult):
                print(f"   {self.COLORS['YELLOW']}Name:{self.COLORS['END']} {item.name}")
                print(f"   {self.COLORS['YELLOW']}Description:{self.COLORS['END']} {item.description[:200]}...")
                if item.categories:
                    print(f"   {self.COLORS['YELLOW']}Categories:{self.COLORS['END']} {item.get_categories_string()}")
                print(f"   {self.COLORS['YELLOW']}Confidence:{self.COLORS['END']} {item.confidence_score:.2f}")
                
            elif isinstance(item, CoreBeliefResult):
                print(f"   {self.COLORS['YELLOW']}Statement:{self.COLORS['END']} {item.statement}")
                print(f"   {self.COLORS['YELLOW']}Category:{self.COLORS['END']} {item.category}")
                print(f"   {self.COLORS['YELLOW']}Confidence:{self.COLORS['END']} {item.confidence_score:.2f} ({item.get_confidence_level()})")
                if item.supporting_evidence:
                    print(f"   {self.COLORS['YELLOW']}Evidence:{self.COLORS['END']} {len(item.supporting_evidence)} items")
                    
            else:
                # Handle raw documents
                content = getattr(item, 'page_content', str(item))
                print(f"   {content[:300]}...")
                if hasattr(item, 'metadata'):
                    print(f"   {self.COLORS['YELLOW']}Metadata:{self.COLORS['END']} {item.metadata}")
    
    def test_vector_search(self, query: str, knowledge_type: str = "both"):
        """Test basic vector similarity search"""
        self.print_header(f"Testing Vector Search - {knowledge_type.title()}")
        
        print(f"{self.COLORS['BOLD']}Query:{self.COLORS['END']} {query}")
        
        try:
            if knowledge_type in ["mental_models", "both"]:
                print(f"\n{self.COLORS['YELLOW']}Mental Models Vector Search...{self.COLORS['END']}")
                start_time = time.time()
                
                mm_results = self.mental_models_store.search(
                    query=query,
                    k=5,
                    return_scores=True
                )
                
                mm_time = time.time() - start_time
                print(f"Search time: {mm_time:.3f}s")
                self.print_results(mm_results, "Mental Models Results")
            
            if knowledge_type in ["core_beliefs", "both"]:
                print(f"\n{self.COLORS['YELLOW']}Core Beliefs Vector Search...{self.COLORS['END']}")
                start_time = time.time()
                
                cb_results = self.core_beliefs_store.search(
                    query=query,
                    k=5,
                    return_scores=True
                )
                
                cb_time = time.time() - start_time
                print(f"Search time: {cb_time:.3f}s")
                self.print_results(cb_results, "Core Beliefs Results")
                
        except Exception as e:
            self.print_error(f"Vector search failed: {e}")
    
    def test_reranking(self, query: str, knowledge_type: str = "both"):
        """Test search with and without reranking for comparison"""
        self.print_header(f"Testing Reranking - {knowledge_type.title()}")
        
        print(f"{self.COLORS['BOLD']}Query:{self.COLORS['END']} {query}")
        
        try:
            if knowledge_type in ["mental_models", "both"]:
                print(f"\n{self.COLORS['YELLOW']}Mental Models: Vector vs Vector+Reranking...{self.COLORS['END']}")
                
                # Vector only
                start_time = time.time()
                vector_results = self.knowledge_indexer.search_mental_models(
                    query=query,
                    persona_id=self.persona_id,
                    k=5,
                    use_reranking=False,
                    return_scores=True
                )
                vector_time = time.time() - start_time
                
                # Vector + Reranking
                start_time = time.time()
                reranked_results = self.knowledge_indexer.search_mental_models(
                    query=query,
                    persona_id=self.persona_id,
                    k=5,
                    use_reranking=True,
                    return_scores=True
                )
                reranked_time = time.time() - start_time
                
                print(f"\n{self.COLORS['BOLD']}Performance:{self.COLORS['END']}")
                print(f"Vector only: {vector_time:.3f}s")
                print(f"Vector+Reranking: {reranked_time:.3f}s")
                print(f"Reranking overhead: {(reranked_time - vector_time):.3f}s")
                
                self.print_results(vector_results, "Mental Models - Vector Only")
                self.print_results(reranked_results, "Mental Models - Vector + Reranking")
            
            if knowledge_type in ["core_beliefs", "both"]:
                print(f"\n{self.COLORS['YELLOW']}Core Beliefs: Vector vs Vector+Reranking...{self.COLORS['END']}")
                
                # Vector only
                start_time = time.time()
                vector_results = self.knowledge_indexer.search_core_beliefs(
                    query=query,
                    persona_id=self.persona_id,
                    k=5,
                    use_reranking=False,
                    return_scores=True
                )
                vector_time = time.time() - start_time
                
                # Vector + Reranking
                start_time = time.time()
                reranked_results = self.knowledge_indexer.search_core_beliefs(
                    query=query,
                    persona_id=self.persona_id,
                    k=5,
                    use_reranking=True,
                    return_scores=True
                )
                reranked_time = time.time() - start_time
                
                print(f"\n{self.COLORS['BOLD']}Performance:{self.COLORS['END']}")
                print(f"Vector only: {vector_time:.3f}s")
                print(f"Vector+Reranking: {reranked_time:.3f}s")
                print(f"Reranking overhead: {(reranked_time - vector_time):.3f}s")
                
                self.print_results(vector_results, "Core Beliefs - Vector Only")
                self.print_results(reranked_results, "Core Beliefs - Vector + Reranking")
                
        except Exception as e:
            self.print_error(f"Reranking test failed: {e}")
    
    def interactive_search(self):
        """Interactive search interface for both knowledge types"""
        self.print_header("Interactive Knowledge Search")
        
        print(f"{self.COLORS['BOLD']}Available Knowledge Types:{self.COLORS['END']}")
        print("• Mental Models: Problem-solving frameworks and methodologies")
        print("• Core Beliefs: Foundational principles and values")
        print("• Both: Search across both knowledge types")
        print("\nEnter your search queries (type 'quit' to exit)")
        
        while True:
            try:
                # Get user input
                query = input(f"\n{self.COLORS['BOLD']}Search> {self.COLORS['END']}").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                # Ask for knowledge type
                print(f"\n{self.COLORS['BOLD']}Knowledge Type (m/c/b): [m]ental, [c]ore beliefs, [b]oth:{self.COLORS['END']} ", end="")
                knowledge_choice = input().strip().lower() or 'b'
                
                if knowledge_choice.startswith('m'):
                    knowledge_type = "mental_models"
                elif knowledge_choice.startswith('c'):
                    knowledge_type = "core_beliefs"
                else:
                    knowledge_type = "both"
                
                # Ask for search mode
                print(f"{self.COLORS['BOLD']}Search Mode (v/r): [v]ector only, [r]eranked:{self.COLORS['END']} ", end="")
                mode_choice = input().strip().lower() or 'r'
                use_reranking = not mode_choice.startswith('v')
                
                # Perform search
                start_time = time.time()
                mode_str = "Vector + Reranking" if use_reranking else "Vector Only"
                print(f"\n{self.COLORS['YELLOW']}Searching with {mode_str}...{self.COLORS['END']}")
                
                if knowledge_type == "mental_models":
                    results = self.knowledge_indexer.search_mental_models(
                        query=query,
                        persona_id=self.persona_id,
                        k=5,
                        use_reranking=use_reranking,
                        return_scores=True
                    )
                    
                elif knowledge_type == "core_beliefs":
                    results = self.knowledge_indexer.search_core_beliefs(
                        query=query,
                        persona_id=self.persona_id,
                        k=5,
                        use_reranking=use_reranking,
                        return_scores=True
                    )
                    
                else:  # both
                    print(f"Searching Mental Models...")
                    mm_results = self.knowledge_indexer.search_mental_models(
                        query=query,
                        persona_id=self.persona_id,
                        k=3,
                        use_reranking=use_reranking,
                        return_scores=True
                    )
                    
                    print(f"Searching Core Beliefs...")
                    cb_results = self.knowledge_indexer.search_core_beliefs(
                        query=query,
                        persona_id=self.persona_id,
                        k=3,
                        use_reranking=use_reranking,
                        return_scores=True
                    )
                    
                    elapsed = time.time() - start_time
                    self.print_success(f"Search completed in {elapsed:.2f}s")
                    self.print_results(mm_results, f"Mental Models Results ({mode_str})")
                    self.print_results(cb_results, f"Core Beliefs Results ({mode_str})")
                    continue
                
                elapsed = time.time() - start_time
                self.print_success(f"Search completed in {elapsed:.2f}s")
                self.print_results(results, f"{knowledge_type.replace('_', ' ').title()} Results ({mode_str})")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.print_error(f"Search failed: {e}")
        
        print(f"\n{self.COLORS['GREEN']}Interactive search session ended{self.COLORS['END']}")
    
    def run_batch_tests(self):
        """Run a series of predefined test queries"""
        self.print_header("Running Batch Tests")
        
        test_queries = [
            "How to make better decisions?",
            "What drives long-term success?", 
            "How to handle failure and setbacks?",
            "What are the keys to productivity?",
            "How to build good habits?"
        ]
        
        print(f"Running {len(test_queries)} test queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{self.COLORS['BLUE']}Test {i}/{len(test_queries)}: {query}{self.COLORS['END']}")
            
            try:
                # Test both knowledge types with reranking
                start_time = time.time()
                
                mm_results = self.knowledge_indexer.search_mental_models(
                    query=query,
                    persona_id=self.persona_id,
                    k=2,
                    use_reranking=True
                )
                
                cb_results = self.knowledge_indexer.search_core_beliefs(
                    query=query,
                    persona_id=self.persona_id,
                    k=2,
                    use_reranking=True
                )
                
                elapsed = time.time() - start_time
                
                mm_count = len(mm_results) if mm_results else 0
                cb_count = len(cb_results) if cb_results else 0
                
                print(f"  Results: {mm_count} mental models, {cb_count} core beliefs ({elapsed:.2f}s)")
                
            except Exception as e:
                print(f"  {self.COLORS['RED']}Failed: {e}{self.COLORS['END']}")
        
        self.print_success("Batch tests completed")
    
    def run_main_menu(self):
        """Run the main interactive menu"""
        self.print_header(f"Knowledge Testing - Persona: {self.persona_id}")
        
        while True:
            print(f"\n{self.COLORS['BOLD']}Available Tests:{self.COLORS['END']}")
            print("1. Test Vector Search")
            print("2. Test Reranking (Vector vs Vector+Reranking)")
            print("3. Interactive Search Session")
            print("4. Run Batch Tests")
            print("5. Display Knowledge Base Statistics")
            print("6. Exit")
            
            try:
                choice = input(f"\n{self.COLORS['BOLD']}Select option (1-6): {self.COLORS['END']}").strip()
                
                if choice == '6':
                    break
                elif choice == '1':
                    query = input("Enter search query: ").strip()
                    if query:
                        knowledge_type = input("Knowledge type (m/c/b): ").strip().lower() or 'b'
                        if knowledge_type.startswith('m'):
                            knowledge_type = "mental_models"
                        elif knowledge_type.startswith('c'):
                            knowledge_type = "core_beliefs"
                        else:
                            knowledge_type = "both"
                        self.test_vector_search(query, knowledge_type)
                elif choice == '2':
                    query = input("Enter search query: ").strip()
                    if query:
                        knowledge_type = input("Knowledge type (m/c/b): ").strip().lower() or 'b'
                        if knowledge_type.startswith('m'):
                            knowledge_type = "mental_models"
                        elif knowledge_type.startswith('c'):
                            knowledge_type = "core_beliefs"
                        else:
                            knowledge_type = "both"
                        self.test_reranking(query, knowledge_type)
                elif choice == '3':
                    self.interactive_search()
                elif choice == '4':
                    self.run_batch_tests()
                elif choice == '5':
                    self.display_knowledge_stats()
                else:
                    print("Invalid choice. Please select 1-6.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.print_error(f"Menu error: {e}")
        
        print(f"\n{self.COLORS['GREEN']}Knowledge testing session ended{self.COLORS['END']}")
    
    def display_knowledge_stats(self):
        """Display detailed statistics about the knowledge bases"""
        self.print_header("Knowledge Base Statistics")
        
        try:
            # Mental Models Stats
            mm_stats = self.mental_models_store.get_collection_stats()
            print(f"\n{self.COLORS['CYAN']}Mental Models:{self.COLORS['END']}")
            print(f"  Documents: {mm_stats.get('current_document_count', 'unknown')}")
            print(f"  Directory Size: {mm_stats.get('directory_size_mb', 'unknown')} MB")
            print(f"  Last Indexed: {mm_stats.get('last_indexed', 'unknown')}")
            print(f"  Collection Path: {mm_stats.get('collection_path', 'unknown')}")
            
            # Core Beliefs Stats
            cb_stats = self.core_beliefs_store.get_collection_stats()
            print(f"\n{self.COLORS['CYAN']}Core Beliefs:{self.COLORS['END']}")
            print(f"  Documents: {cb_stats.get('current_document_count', 'unknown')}")
            print(f"  Directory Size: {cb_stats.get('directory_size_mb', 'unknown')} MB")
            print(f"  Last Indexed: {cb_stats.get('last_indexed', 'unknown')}")
            print(f"  Collection Path: {cb_stats.get('collection_path', 'unknown')}")
            
        except Exception as e:
            self.print_error(f"Failed to get statistics: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Interactive testing for mental models and core beliefs knowledge bases"
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
    
    try:
        tester = KnowledgeTester(args.persona_id, args.config)
        tester.run_main_menu()
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()