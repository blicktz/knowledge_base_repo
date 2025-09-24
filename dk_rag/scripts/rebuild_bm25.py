#!/usr/bin/env python3
"""
Rebuild BM25 Index Script
Rebuilds only the BM25 index without touching the vector store.
"""

import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dk_rag.config.settings import Settings
from dk_rag.core.persona_manager import PersonaManager
from dk_rag.core.knowledge_indexer import KnowledgeIndexer
from dk_rag.utils.logging import get_logger

def main():
    parser = argparse.ArgumentParser(description="Rebuild BM25 index only (preserves vector store)")
    parser.add_argument("--persona-id", required=True, help="Persona identifier (e.g., 'greg_startup')")
    parser.add_argument("--config", default="./config/persona_config.yaml", help="Configuration file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = get_logger(__name__)
    if args.verbose:
        import logging
        logging.getLogger('dk_rag').setLevel(logging.DEBUG)
    
    try:
        # Load settings
        logger.info(f"Loading configuration from: {args.config}")
        settings = Settings.from_file(args.config)
        
        # Initialize persona manager
        persona_manager = PersonaManager(settings)
        
        # Check if persona exists
        if not persona_manager.persona_exists(args.persona_id):
            logger.error(f"Persona '{args.persona_id}' not found. Available personas:")
            for persona in persona_manager.list_personas():
                logger.info(f"  - {persona['name']} (ID: {persona['id']})")
            sys.exit(1)
        
        # Get persona-specific vector store
        vector_store = persona_manager.get_persona_vector_store(args.persona_id)
        
        # Check if vector store has documents
        stats = vector_store.get_collection_stats()
        document_count = stats.get('total_chunks', 0)
        
        if document_count == 0:
            logger.error("Vector store is empty. Please build the knowledge base first with 'make build-kb'")
            sys.exit(1)
        
        logger.info(f"Found {document_count} documents in vector store for persona '{args.persona_id}'")
        
        # Initialize knowledge indexer with persona manager
        knowledge_indexer = KnowledgeIndexer(settings, persona_manager, persona_id=args.persona_id)
        
        # Rebuild only Phase 2 indexes (BM25)
        logger.info("Rebuilding BM25 index (Phase 2)...")
        knowledge_indexer.build_phase2_indexes(persona_id=args.persona_id, rebuild=True)
        
        logger.info("✅ BM25 index rebuilt successfully!")
        logger.info(f"Vector store preserved with {document_count} documents")
        
    except Exception as e:
        logger.error(f"Failed to rebuild BM25 index: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()