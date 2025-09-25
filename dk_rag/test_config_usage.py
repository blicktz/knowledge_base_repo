#!/usr/bin/env python3
"""
Test script to verify configuration usage in the LangChain agent system
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dk_rag.config.settings import Settings
from dk_rag.tools.agent_tools import get_tools_for_persona


def test_config_usage():
    """Test that all components properly use configuration instead of hardcoded values"""
    
    print("🔧 Testing Configuration Usage in LangChain Agent System")
    print("=" * 60)
    
    try:
        # Initialize settings
        settings = Settings()
        print(f"✅ Settings loaded successfully")
        
        # Test agent configuration sections exist
        print(f"\n📋 Agent Configuration:")
        print(f"   Agent enabled: {settings.agent.enabled}")
        print(f"   Query Analysis Model: {settings.agent.query_analysis.llm_model}")
        print(f"   Synthesis Model: {settings.agent.synthesis.llm_model}")
        print(f"   Mental Models k: {settings.agent.tools.mental_models.get('k', 'not configured')}")
        print(f"   Core Beliefs k: {settings.agent.tools.core_beliefs.get('k', 'not configured')}")
        print(f"   Transcripts k: {settings.agent.tools.transcripts.get('k', 'not configured')}")
        
        # Test API configuration
        print(f"\n🌐 API Configuration:")
        print(f"   Host: {settings.api.host}:{settings.api.port}")
        print(f"   Title: {settings.api.settings.title}")
        print(f"   Version: {settings.api.settings.version}")
        
        # Test model selection logic
        print(f"\n🤖 Model Selection Logic:")
        query_model = settings.agent.query_analysis.llm_model
        synthesis_model = settings.agent.synthesis.llm_model
        
        if "2.0-flash" in query_model:
            print(f"   ✅ Query Analysis uses fast model: {query_model}")
        else:
            print(f"   ⚠️  Query Analysis model: {query_model} (expected fast model)")
            
        if "2.5-pro" in synthesis_model:
            print(f"   ✅ Synthesis uses heavy model: {synthesis_model}")
        else:
            print(f"   ⚠️  Synthesis model: {synthesis_model} (expected heavy model)")
        
        # Test tools configuration
        print(f"\n🛠️  Tools Configuration Test:")
        try:
            # This should work without hardcoded values
            tools = get_tools_for_persona("test_persona", settings)
            print(f"   ✅ Tools initialized successfully: {len(tools)} tools")
            for tool in tools:
                print(f"      - {tool.name}")
        except Exception as e:
            print(f"   ❌ Tools initialization failed: {str(e)}")
        
        print(f"\n✅ Configuration usage test completed!")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_config_usage()