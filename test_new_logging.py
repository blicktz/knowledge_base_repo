#!/usr/bin/env python3
"""
Test the new explicit context LLM logging system
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Fix HuggingFace tokenizers parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dk_rag.config.settings import Settings
from dk_rag.utils.llm_factory import create_query_analysis_llm
from dk_rag.agent.persona_agent import LangChainPersonaAgent

def test_explicit_logging():
    """Test that the new explicit logging system works"""
    print("🧪 Testing New Explicit Context LLM Logging")
    print("="*60)
    
    settings = Settings.from_default_config()
    persona_id = "greg_startup"
    
    # Check logging path configuration
    logging_path = settings.get_llm_logging_path(persona_id)
    print(f"✅ LLM logging path: {logging_path}")
    
    # Test 1: Direct LLM creation with explicit context
    print(f"\n🔍 Test 1: Query Analysis LLM Creation")
    try:
        llm = create_query_analysis_llm(persona_id, settings)
        print(f"✅ Created query analysis LLM successfully")
        print(f"   Model: {llm.model}")
        print(f"   Callbacks: {len(llm.callbacks)} callback(s)")
        
        # Test a simple call
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content="Test query: Help me with customer acquisition")])
        print(f"✅ LLM call succeeded: {len(response.content)} chars")
        
    except Exception as e:
        print(f"❌ Query analysis LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Agent with new logging
    print(f"\n🤖 Test 2: Agent with New Logging System")
    try:
        agent = LangChainPersonaAgent(persona_id, settings)
        print(f"✅ Created agent successfully")
        print(f"   Tools: {[tool.name for tool in agent.tools]}")
        
        # Test query analysis (should use explicit context logging)
        query = "Help me get my first 50 customers for my call answering service"
        analysis = agent._analyze_query(query)
        print(f"✅ Query analysis succeeded:")
        print(f"   Core task: {analysis.get('core_task', 'N/A')[:50]}...")
        print(f"   RAG query: {analysis.get('rag_query', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check if log files were created
    print(f"\n📁 Test 3: Checking Log Files")
    try:
        log_dir = Path(logging_path)
        if log_dir.exists():
            log_folders = list(log_dir.glob("*"))
            print(f"✅ Found {len(log_folders)} log folder(s):")
            for folder in sorted(log_folders)[-3:]:  # Show last 3
                print(f"   - {folder.name}")
        else:
            print(f"⚠️  Log directory doesn't exist yet: {log_dir}")
    except Exception as e:
        print(f"❌ Log file check failed: {e}")
    
    print(f"\n🎉 All tests completed!")
    return True

if __name__ == "__main__":
    success = test_explicit_logging()
    sys.exit(0 if success else 1)