#!/usr/bin/env python3
"""
Interactive Step-by-Step Chain Testing Script
Allows testing each component of the LangChain persona agent system individually
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Fix HuggingFace tokenizers parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dk_rag.config.settings import Settings
from dk_rag.tools.agent_tools import (
    get_persona_data,
    retrieve_mental_models,
    retrieve_core_beliefs, 
    retrieve_transcripts
)
from dk_rag.chains.synthesis_chain import create_full_synthesis_pipeline
from dk_rag.agent.persona_agent import LangChainPersonaAgent
from dk_rag.utils.logging import get_logger

logger = get_logger(__name__)


class InteractiveChainTester:
    """Interactive tester for LangChain persona agent components"""
    
    def __init__(self):
        self.settings = Settings.from_default_config()
        self.current_persona = None
        self.available_personas = self._discover_personas()
        
    def _discover_personas(self) -> List[str]:
        """Discover available personas from storage directory"""
        personas_dir = Path(self.settings.get_personas_base_dir())
        if not personas_dir.exists():
            return ["greg_startup"]  # Default persona
            
        personas = []
        for persona_dir in personas_dir.iterdir():
            if persona_dir.is_dir() and persona_dir.name != "__pycache__":
                personas.append(persona_dir.name)
        
        return personas if personas else ["greg_startup"]
    
    def display_header(self):
        """Display the main header with configuration info"""
        print("\n" + "="*60)
        print("🧪 LangChain Persona Agent - Interactive Step Tester")
        print("="*60)
        
        print(f"\n📋 Configuration:")
        print(f"   Base Storage: {self.settings.storage.base_storage_dir}")
        print(f"   Query Analysis Model: {self.settings.agent.query_analysis.llm_model} (fast)")
        print(f"   Synthesis Model: {self.settings.agent.synthesis.llm_model} (heavy)")
        print(f"   Retrieval Config: MM={self.settings.agent.tools.mental_models.get('k', 3)}, CB={self.settings.agent.tools.core_beliefs.get('k', 5)}, T={self.settings.agent.tools.transcripts.get('k', 5)}")
        
        print(f"\n👥 Available Personas: {', '.join(self.available_personas)}")
        if self.current_persona:
            print(f"   Selected Persona: {self.current_persona}")
        else:
            print("   No persona selected")
    
    def select_persona(self) -> Optional[str]:
        """Allow user to select a persona"""
        print(f"\n🎭 Persona Selection:")
        for i, persona in enumerate(self.available_personas, 1):
            print(f"   {i}. {persona}")
        print(f"   {len(self.available_personas) + 1}. Keep current ({self.current_persona or 'none'})")
        
        while True:
            try:
                choice = input(f"\nSelect persona (1-{len(self.available_personas) + 1}): ").strip()
                if not choice:
                    return self.current_persona
                    
                choice_num = int(choice)
                if 1 <= choice_num <= len(self.available_personas):
                    return self.available_personas[choice_num - 1]
                elif choice_num == len(self.available_personas) + 1:
                    return self.current_persona
                else:
                    print("❌ Invalid choice. Please try again.")
            except (ValueError, KeyboardInterrupt):
                return self.current_persona
    
    def display_menu(self):
        """Display the main test menu"""
        print(f"\n🧪 Test Options:")
        print("   1. Query Analysis Test")
        print("   2. Persona Data Retrieval Test")
        print("   3. Mental Models Retrieval Test") 
        print("   4. Core Beliefs Retrieval Test")
        print("   5. Transcript Retrieval Test")
        print("   6. Synthesis Chain Test")
        print("   7. End-to-End Agent Test")
        print("   8. Configuration Validation")
        print("   9. Select Different Persona")
        print("   0. Exit")
    
    def get_user_input(self, prompt: str, default: str = None) -> str:
        """Get user input with optional default"""
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            return user_input if user_input else default
        else:
            return input(f"{prompt}: ").strip()
    
    def format_execution_time(self, start_time: float) -> str:
        """Format execution time"""
        elapsed = time.time() - start_time
        return f"{elapsed:.2f}s"
    
    def print_results(self, title: str, results: Any, execution_time: str):
        """Print formatted results"""
        print(f"\n✅ {title}")
        if isinstance(results, dict):
            print(json.dumps(results, indent=2))
        elif isinstance(results, list):
            for i, item in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                if isinstance(item, dict):
                    print(json.dumps(item, indent=2))
                else:
                    print(str(item))
        else:
            print(str(results))
        print(f"\n⏱️  Execution time: {execution_time}")
    
    def test_query_analysis(self):
        """Test the query analyzer tool"""
        print(f"\n🔍 Query Analysis Test")
        print("="*30)
        
        query = self.get_user_input(
            "Enter your query",
            "Write me a sales email for a new SaaS product"
        )
        
        if not query:
            print("❌ No query provided")
            return
        
        print(f"\n⏱️  Running query analysis...")
        start_time = time.time()
        
        try:
            # Query analysis is now internal to the agent, test via agent
            agent = LangChainPersonaAgent(self.current_persona, self.settings)
            results = agent._analyze_query(query)
            execution_time = self.format_execution_time(start_time)
            self.print_results("Query Analysis Results:", results, execution_time)
        except Exception as e:
            print(f"❌ Query analysis failed: {str(e)}")
            logger.exception("Query analysis test failed")
    
    def test_persona_data(self):
        """Test persona data retrieval"""
        if not self.current_persona:
            print("❌ No persona selected. Please select a persona first.")
            return
            
        print(f"\n👤 Persona Data Retrieval Test")
        print("="*35)
        print(f"Persona: {self.current_persona}")
        
        print(f"\n⏱️  Loading persona data...")
        start_time = time.time()
        
        try:
            results = get_persona_data.invoke({"persona_id": self.current_persona, "settings": self.settings})
            execution_time = self.format_execution_time(start_time)
            self.print_results("Persona Data Results:", results, execution_time)
        except Exception as e:
            print(f"❌ Persona data retrieval failed: {str(e)}")
            logger.exception("Persona data test failed")
    
    def test_mental_models(self):
        """Test mental models retrieval"""
        if not self.current_persona:
            print("❌ No persona selected. Please select a persona first.")
            return
            
        print(f"\n🧠 Mental Models Retrieval Test")
        print("="*35)
        print(f"Persona: {self.current_persona}")
        
        query = self.get_user_input(
            "Enter search query for mental models",
            "business growth strategies"
        )
        
        if not query:
            print("❌ No query provided")
            return
        
        print(f"\n⏱️  Retrieving mental models...")
        start_time = time.time()
        
        try:
            results = retrieve_mental_models.invoke({"query": query, "persona_id": self.current_persona, "settings": self.settings})
            execution_time = self.format_execution_time(start_time)
            self.print_results(f"Mental Models Results (k={len(results)}):", results, execution_time)
        except Exception as e:
            print(f"❌ Mental models retrieval failed: {str(e)}")
            logger.exception("Mental models test failed")
    
    def test_core_beliefs(self):
        """Test core beliefs retrieval"""
        if not self.current_persona:
            print("❌ No persona selected. Please select a persona first.")
            return
            
        print(f"\n💭 Core Beliefs Retrieval Test")
        print("="*32)
        print(f"Persona: {self.current_persona}")
        
        query = self.get_user_input(
            "Enter search query for core beliefs",
            "entrepreneurship philosophy"
        )
        
        if not query:
            print("❌ No query provided")
            return
        
        print(f"\n⏱️  Retrieving core beliefs...")
        start_time = time.time()
        
        try:
            results = retrieve_core_beliefs.invoke({"query": query, "persona_id": self.current_persona, "settings": self.settings})
            execution_time = self.format_execution_time(start_time)
            self.print_results(f"Core Beliefs Results (k={len(results)}):", results, execution_time)
        except Exception as e:
            print(f"❌ Core beliefs retrieval failed: {str(e)}")
            logger.exception("Core beliefs test failed")
    
    def test_transcripts(self):
        """Test transcript retrieval"""
        if not self.current_persona:
            print("❌ No persona selected. Please select a persona first.")
            return
            
        print(f"\n📄 Transcript Retrieval Test")
        print("="*30)
        print(f"Persona: {self.current_persona}")
        
        query = self.get_user_input(
            "Enter search query for transcripts",
            "sales process customer acquisition"
        )
        
        if not query:
            print("❌ No query provided")
            return
        
        print(f"\n⏱️  Retrieving transcripts...")
        start_time = time.time()
        
        try:
            results = retrieve_transcripts.invoke({"query": query, "persona_id": self.current_persona, "settings": self.settings})
            execution_time = self.format_execution_time(start_time)
            self.print_results(f"Transcript Results (k={len(results)}):", results, execution_time)
        except Exception as e:
            print(f"❌ Transcript retrieval failed: {str(e)}")
            logger.exception("Transcript test failed")
    
    def test_synthesis_chain(self):
        """Test the synthesis chain with mock data"""
        if not self.current_persona:
            print("❌ No persona selected. Please select a persona first.")
            return
            
        print(f"\n🔗 Synthesis Chain Test")
        print("="*25)
        print(f"Persona: {self.current_persona}")
        
        # Get user inputs
        query = self.get_user_input(
            "Enter original user query",
            "Help me write a compelling sales email"
        )
        
        context = self.get_user_input(
            "Enter additional context (optional)",
            "For a B2B SaaS product targeting small businesses"
        )
        
        if not query:
            print("❌ No query provided")
            return
        
        print(f"\n⏱️  Running synthesis chain...")
        start_time = time.time()
        
        try:
            # Create mock retrieval results
            mock_persona_data = {
                "linguistic_style": {"tone": "professional", "style": "direct"},
                "communication_patterns": {"prefers_bullet_points": True}
            }
            
            mock_mental_models = [
                {"content": "Focus on customer pain points first", "score": 0.85},
                {"content": "Always include social proof", "score": 0.80}
            ]
            
            mock_core_beliefs = [
                {"content": "Authentic communication builds trust", "score": 0.90},
                {"content": "Value proposition must be clear and immediate", "score": 0.85}
            ]
            
            mock_transcripts = [
                {"content": "The best sales emails start with a problem the customer recognizes...", "score": 0.88}
            ]
            
            # Create synthesis pipeline using LCEL function
            synthesis_pipeline = create_full_synthesis_pipeline(self.current_persona, self.settings)
            
            # Prepare input data in the format expected by the chain
            chain_input = {
                "user_query": query,
                "query_analysis": {"core_task": f"User requests: {query}"},
                "persona_data": mock_persona_data,
                "mental_models": mock_mental_models,
                "core_beliefs": mock_core_beliefs,
                "transcripts": mock_transcripts
            }
            
            # Run synthesis using LCEL chain
            results = synthesis_pipeline.invoke(chain_input)
            
            execution_time = self.format_execution_time(start_time)
            self.print_results("Synthesis Chain Results:", results, execution_time)
            
        except Exception as e:
            print(f"❌ Synthesis chain test failed: {str(e)}")
            logger.exception("Synthesis chain test failed")
    
    def test_end_to_end_agent(self):
        """Test the complete ReAct agent"""
        if not self.current_persona:
            print("❌ No persona selected. Please select a persona first.")
            return
            
        print(f"\n🤖 End-to-End Agent Test")
        print("="*26)
        print(f"Persona: {self.current_persona}")
        
        query = self.get_user_input(
            "Enter your query for the agent",
            "Help me create a marketing strategy for my startup"
        )
        
        if not query:
            print("❌ No query provided")
            return
        
        print(f"\n⏱️  Initializing and running agent...")
        start_time = time.time()
        
        try:
            # Initialize agent
            agent = LangChainPersonaAgent(self.current_persona, self.settings)
            
            # Create a conversation session
            session_id = f"test-session-{int(time.time())}"
            
            # Run agent
            results = agent.process_query(query, session_id)
            
            execution_time = self.format_execution_time(start_time)
            print(f"\n✅ Agent Response:")
            print("-" * 50)
            print(results)
            print("-" * 50)
            print(f"\n⏱️  Execution time: {execution_time}")
            
        except Exception as e:
            print(f"❌ End-to-end agent test failed: {str(e)}")
            logger.exception("End-to-end agent test failed")
    
    def test_configuration(self):
        """Test and validate configuration"""
        print(f"\n⚙️  Configuration Validation Test")
        print("="*35)
        
        print(f"\n⏱️  Validating configuration...")
        start_time = time.time()
        
        try:
            # Run configuration validation
            issues = self.settings.validate_configuration()
            
            execution_time = self.format_execution_time(start_time)
            
            if not issues:
                print(f"\n✅ Configuration is valid!")
                print(f"   All settings loaded successfully")
                print(f"   All dependencies are available")
            else:
                print(f"\n⚠️  Configuration issues found:")
                for issue in issues:
                    print(f"   - {issue}")
            
            # Show detailed configuration
            print(f"\n📋 Detailed Configuration:")
            print(f"   Agent Enabled: {self.settings.agent.enabled}")
            print(f"   Query Analysis: {self.settings.agent.query_analysis.llm_model}")
            print(f"   Synthesis: {self.settings.agent.synthesis.llm_model}")
            print(f"   Storage Base: {self.settings.storage.base_storage_dir}")
            print(f"   Available Personas: {len(self.available_personas)}")
            
            print(f"\n⏱️  Validation time: {execution_time}")
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {str(e)}")
            logger.exception("Configuration validation failed")
    
    def run(self):
        """Main interactive loop"""
        # Initial setup
        self.display_header()
        
        # Select initial persona
        if not self.current_persona and self.available_personas:
            print(f"\n🎭 Please select a persona to test with:")
            self.current_persona = self.select_persona()
        
        while True:
            try:
                self.display_header()
                self.display_menu()
                
                choice = input(f"\nEnter your choice (0-9): ").strip()
                
                if choice == "0":
                    print(f"\n👋 Goodbye! Happy testing!")
                    break
                elif choice == "1":
                    self.test_query_analysis()
                elif choice == "2":
                    self.test_persona_data()
                elif choice == "3":
                    self.test_mental_models()
                elif choice == "4":
                    self.test_core_beliefs()
                elif choice == "5":
                    self.test_transcripts()
                elif choice == "6":
                    self.test_synthesis_chain()
                elif choice == "7":
                    self.test_end_to_end_agent()
                elif choice == "8":
                    self.test_configuration()
                elif choice == "9":
                    new_persona = self.select_persona()
                    if new_persona:
                        self.current_persona = new_persona
                        print(f"✅ Switched to persona: {new_persona}")
                    else:
                        print("❌ No persona selected")
                else:
                    print("❌ Invalid choice. Please try again.")
                
                # Wait for user to continue
                input(f"\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Goodbye! Happy testing!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")
                logger.exception("Unexpected error in main loop")
                input(f"\nPress Enter to continue...")


def main():
    """Main entry point"""
    try:
        tester = InteractiveChainTester()
        tester.run()
    except KeyboardInterrupt:
        print(f"\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start interactive tester: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()