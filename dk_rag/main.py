#!/usr/bin/env python3
"""
LangChain-native FastAPI Main Application Entry Point
Launches the new LangChain persona agent API
"""

import uvicorn
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Main entry point for the LangChain FastAPI application"""
    
    try:
        # Import after path setup
        from .config.settings import Settings
        
        # Load settings to validate configuration
        settings = Settings()
        
        print("🦜🔗 Starting LangChain Persona Agent API")
        print("=" * 50)
        print(f"📍 Base storage directory: {settings.storage.base_storage_dir}")
        print(f"🔧 Agent framework: LangChain ReAct with MemorySaver")
        print(f"🧠 Query Analysis Model: {settings.agent.query_analysis.llm_model} (fast)")
        print(f"🧠 Synthesis Model: {settings.agent.synthesis.llm_model} (heavy)")
        print(f"🎯 Retrieval Config: MM={settings.agent.tools.mental_models.get('k', 3)}, CB={settings.agent.tools.core_beliefs.get('k', 5)}, T={settings.agent.tools.transcripts.get('k', 5)}")
        print(f"🌐 API host: {settings.api.host}:{settings.api.port}")
        print(f"📚 Docs: http://{settings.api.host}:{settings.api.port}/docs")
        print(f"💬 Memory: Conversation context enabled")
        print("=" * 50)
        
        # Run the LangChain FastAPI application
        uvicorn.run(
            "dk_rag.api.persona_api:app",
            host=settings.api.host,
            port=settings.api.port,
            reload=settings.api.reload,
            workers=settings.api.workers,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Failed to start LangChain application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()