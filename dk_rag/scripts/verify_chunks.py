#!/usr/bin/env python
"""
Simple script to verify chunks in the vector database
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dk_rag.config.settings import Settings
from dk_rag.core.persona_manager import PersonaManager
from dk_rag.utils.logging import setup_logger

def verify_chunks(persona_name: str):
    """Verify chunks exist in the database for a given persona."""
    
    logger = setup_logger("verify_chunks", level="INFO")
    
    # Load settings
    settings = Settings.from_default_config()
    
    # Initialize persona manager
    persona_manager = PersonaManager(settings)
    
    # Get persona ID
    persona_id = persona_manager._sanitize_persona_id(persona_name)
    
    if not persona_manager.persona_exists(persona_id):
        print(f"❌ Persona '{persona_name}' not found in registry")
        return False
    
    print(f"✓ Persona '{persona_name}' found in registry as '{persona_id}'")
    
    # Get vector store
    try:
        vector_store = persona_manager.get_persona_vector_store(persona_id)
        
        # Get collection stats
        stats = vector_store.get_collection_stats()
        
        print(f"✓ Vector store connected")
        print(f"  Collection name: {stats.get('collection_name', 'unknown')}")
        print(f"  Total chunks: {stats.get('total_chunks', 0)}")
        print(f"  Document count: {stats.get('document_count', 0)}")
        print(f"  Persona ID: {stats.get('persona_id', 'unknown')}")
        
        # Try a simple search to verify data is accessible
        if stats.get('total_chunks', 0) > 0:
            search_results = vector_store.search("business", n_results=3)
            print(f"✓ Search test: Found {len(search_results)} results for 'business'")
            
            if search_results:
                print("  Sample result:")
                result = search_results[0]
                print(f"    Document preview: {result['document'][:100]}...")
                print(f"    Distance: {result.get('distance', 'N/A')}")
        else:
            print("❌ No chunks found in database")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing vector store: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_chunks.py <persona_name>")
        print("Example: python verify_chunks.py 'greg_startup'")
        sys.exit(1)
    
    persona_name = sys.argv[1]
    success = verify_chunks(persona_name)
    sys.exit(0 if success else 1)