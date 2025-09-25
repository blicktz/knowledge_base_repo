#!/usr/bin/env python3
"""
FastAPI Main Application Entry Point
Launches the Persona Agent API with proper configuration
"""

import uvicorn
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point for the FastAPI application"""
    
    # Import after path setup
    from dk_rag.config.settings import Settings
    
    try:
        # Load settings to validate configuration
        settings = Settings()
        
        print(f"🚀 Starting Persona Agent API")
        print(f"📍 Base storage directory: {settings.storage.base_storage_dir}")
        print(f"🔧 Agent enabled: {settings.agent.enabled}")
        print(f"🌐 API host: {settings.api.host}:{settings.api.port}")
        print(f"📚 Docs available at: http://{settings.api.host}:{settings.api.port}{settings.api.settings.docs_url}")
        
        # Run the FastAPI application
        uvicorn.run(
            "dk_rag.api.persona_api:app",
            host=settings.api.host,
            port=settings.api.port,
            reload=settings.api.reload,
            workers=settings.api.workers,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()