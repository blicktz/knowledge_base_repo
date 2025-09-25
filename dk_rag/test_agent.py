#!/usr/bin/env python3
"""
Test script for the new LangChain persona agent system
"""

import asyncio
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from .config.settings import Settings
from .agent.persona_agent import create_persona_agent
from .core.persona_manager import PersonaManager


async def test_langchain_agent():
    """Test the LangChain persona agent"""
    
    print("🦜🔗 Testing LangChain Persona Agent")
    print("=" * 50)
    
    try:
        # Initialize settings
        settings = Settings()
        print(f"✅ Settings loaded")
        
        # Get available personas
        persona_manager = PersonaManager(settings)
        personas = persona_manager.list_personas()
        
        if not personas:
            print("❌ No personas found!")
            return
            
        persona_id = personas[0]['name']
        print(f"✅ Testing with persona: {persona_id}")
        
        # Create LangChain agent
        print("\n🔄 Creating LangChain agent...")
        agent = create_persona_agent(persona_id, settings)
        print(f"✅ Agent created successfully")
        
        # Display agent info
        agent_info = agent.get_agent_info()
        print(f"\n📊 Agent Info:")
        print(f"   Framework: {agent_info['framework']}")
        print(f"   Model: {agent_info['model']}")
        print(f"   Tools: {agent_info['tools']}")
        print(f"   Memory: {agent_info['has_memory']}")
        
        # Test queries
        test_queries = [
            "Hello! Can you tell me about yourself?",
            "What's your approach to solving problems?", 
            "Give me some advice on marketing strategy",
            "Do you remember what we talked about earlier?"
        ]
        
        session_id = "test_session_123"
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔄 Test Query {i}: {query}")
            print("-" * 40)
            
            try:
                response = agent.process_query(query, session_id)
                print(f"✅ Response: {response[:200]}...")
                if len(response) > 200:
                    print(f"   (Response was {len(response)} characters)")
                    
            except Exception as e:
                print(f"❌ Query failed: {str(e)}")
        
        # Test conversation history
        print(f"\n📝 Testing conversation history...")
        try:
            history = agent.get_conversation_history(session_id)
            print(f"✅ Retrieved {len(history)} messages from history")
            
            if history:
                print("   Recent messages:")
                for msg in history[-2:]:  # Show last 2 messages
                    print(f"   - {msg['type']}: {msg['content'][:100]}...")
                    
        except Exception as e:
            print(f"❌ History retrieval failed: {str(e)}")
        
        # Test clearing conversation
        print(f"\n🧹 Testing conversation clearing...")
        try:
            success = agent.clear_conversation(session_id)
            if success:
                print("✅ Conversation cleared successfully")
            else:
                print("❌ Failed to clear conversation")
        except Exception as e:
            print(f"❌ Clear conversation failed: {str(e)}")
        
        print(f"\n✅ LangChain agent testing completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_langchain_agent())