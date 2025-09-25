"""Test script for Phase 3 implementation"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dk_rag.config.settings import Settings
from dk_rag.agent.persona_agent import PersonaAgent
from dk_rag.core.persona_manager import PersonaManager
from dk_rag.utils.logging import get_logger

def test_basic_imports():
    """Test that all Phase 3 modules can be imported"""
    print("Testing Phase 3 imports...")
    
    try:
        from dk_rag.tools import (
            QueryAnalyzerTool,
            PersonaDataTool,
            MentalModelsRetrieverTool,
            CoreBeliefsRetrieverTool,
            TranscriptRetrieverTool
        )
        print("✓ All tool imports successful")
        
        from dk_rag.agent import PersonaAgent
        print("✓ Agent import successful")
        
        from dk_rag.services import SynthesisEngine
        print("✓ Synthesis engine import successful")
        
        from dk_rag.api import app
        print("✓ FastAPI app import successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {str(e)}")
        return False


def test_settings_loading():
    """Test that settings can be loaded"""
    print("\\nTesting settings loading...")
    
    try:
        settings = Settings()
        print(f"✓ Settings loaded successfully")
        print(f"✓ Agent enabled: {settings.agent.enabled}")
        print(f"✓ Base storage dir: {settings.base_storage_dir}")
        return True
        
    except Exception as e:
        print(f"✗ Settings loading failed: {str(e)}")
        return False


def test_persona_manager():
    """Test persona manager functionality"""
    print("\\nTesting persona manager...")
    
    try:
        settings = Settings()
        manager = PersonaManager(settings)
        personas = manager.list_personas()
        
        print(f"✓ Found {len(personas)} personas")
        for persona in personas:
            print(f"  - {persona['name']}")
        
        return len(personas) > 0
        
    except Exception as e:
        print(f"✗ Persona manager failed: {str(e)}")
        return False


def test_tool_initialization():
    """Test tool initialization"""
    print("\\nTesting tool initialization...")
    
    try:
        settings = Settings()
        manager = PersonaManager(settings)
        personas = manager.list_personas()
        
        if not personas:
            print("✗ No personas available for testing")
            return False
            
        persona_id = personas[0]['name']
        print(f"Using persona: {persona_id}")
        
        from dk_rag.tools import QueryAnalyzerTool
        tool = QueryAnalyzerTool(persona_id, settings)
        print("✓ QueryAnalyzerTool initialized")
        
        from dk_rag.tools import PersonaDataTool
        tool = PersonaDataTool(persona_id, settings)
        print("✓ PersonaDataTool initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Tool initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_initialization():
    """Test agent initialization"""
    print("\\nTesting agent initialization...")
    
    try:
        settings = Settings()
        manager = PersonaManager(settings)
        personas = manager.list_personas()
        
        if not personas:
            print("✗ No personas available for testing")
            return False
            
        persona_id = personas[0]['name']
        print(f"Using persona: {persona_id}")
        
        agent = PersonaAgent(persona_id, settings)
        print("✓ PersonaAgent initialized")
        
        # Test tool status
        status = agent.get_tool_status()
        print("✓ Tool status retrieved:")
        for tool_name, is_healthy in status.items():
            status_symbol = "✓" if is_healthy else "✗"
            print(f"  {status_symbol} {tool_name}: {'healthy' if is_healthy else 'unhealthy'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=== Phase 3 Implementation Test ===\\n")
    
    tests = [
        test_basic_imports,
        test_settings_loading,
        test_persona_manager,
        test_tool_initialization,
        test_agent_initialization
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {str(e)}")
            failed += 1
    
    print(f"\\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    if failed == 0:
        print("\\n🎉 All tests passed! Phase 3 implementation looks good.")
    else:
        print(f"\\n⚠️  {failed} tests failed. Check the output above for details.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)