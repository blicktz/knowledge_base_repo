#!/usr/bin/env python3
"""
Test script for the new Map-Reduce persona extraction implementation
Tests the implementation with the existing greg_startup corpus
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to the Python path for dk_rag imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dk_rag.config.settings import Settings
from dk_rag.core.persona_extractor import PersonaExtractor
from dk_rag.utils.logging import setup_logger


async def test_map_reduce_extraction():
    """Test the map-reduce extraction with a subset of documents"""
    
    # Setup logging
    logger = setup_logger("test_map_reduce", level="INFO")
    logger.info("Starting Map-Reduce extraction test")
    
    try:
        # Load settings with map-reduce enabled
        config_path = Path(__file__).parent / "config" / "persona_config.yaml"
        settings = Settings.from_file(str(config_path))
        
        # Ensure map-reduce is enabled
        settings.map_reduce_extraction.enabled = True
        settings.map_reduce_extraction.batch_size = 5  # Smaller batches for testing
        settings.map_reduce_extraction.show_progress = True
        
        logger.info(f"Map-reduce enabled: {settings.map_reduce_extraction.enabled}")
        logger.info(f"Batch size: {settings.map_reduce_extraction.batch_size}")
        logger.info(f"Map phase model: {settings.map_reduce_extraction.map_phase_model}")
        logger.info(f"Reduce phase model: {settings.map_reduce_extraction.reduce_phase_model}")
        
        # Check API key availability
        import os
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            logger.info(f"✅ GEMINI_API_KEY found (length: {len(gemini_key)})")
        else:
            logger.warning("⚠️ GEMINI_API_KEY not found in environment")
        
        # Initialize persona extractor
        extractor = PersonaExtractor(settings, persona_id="greg_startup")
        
        # Load a subset of documents for testing
        content_dir = Path(__file__).parent.parent / "content_repo" / "greg_startup"
        
        if not content_dir.exists():
            logger.error(f"Content directory not found: {content_dir}")
            return False
        
        # Load first 20 documents for testing
        documents = []
        txt_files = list(content_dir.glob("*.txt"))[:20]  # First 20 files
        
        if not txt_files:
            logger.error(f"No .txt files found in {content_dir}")
            return False
        
        logger.info(f"Loading {len(txt_files)} documents for testing")
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        'content': content,
                        'source': txt_file.name
                    })
            except Exception as e:
                logger.warning(f"Failed to load {txt_file}: {e}")
                continue
        
        if not documents:
            logger.error("No documents loaded successfully")
            return False
        
        total_words = sum(len(doc['content'].split()) for doc in documents)
        logger.info(f"Loaded {len(documents)} documents with {total_words:,} words")
        
        # Extract persona using map-reduce
        logger.info("Starting persona extraction with map-reduce")
        persona_constitution = await extractor.extract_persona(
            documents=documents,
            use_cached_analysis=True,
            force_reanalyze=False
        )
        
        # Display results
        logger.info("=== EXTRACTION RESULTS ===")
        logger.info(f"Mental Models: {len(persona_constitution.mental_models)}")
        logger.info(f"Core Beliefs: {len(persona_constitution.core_beliefs)}")
        logger.info(f"Catchphrases: {len(persona_constitution.linguistic_style.catchphrases)}")
        
        # Show first few results
        if persona_constitution.mental_models:
            logger.info("\nFirst Mental Models:")
            for i, model in enumerate(persona_constitution.mental_models[:3]):
                logger.info(f"  {i+1}. {model.name}")
                logger.info(f"     {model.description}")
        
        if persona_constitution.core_beliefs:
            logger.info("\nFirst Core Beliefs:")
            for i, belief in enumerate(persona_constitution.core_beliefs[:3]):
                logger.info(f"  {i+1}. {belief.statement}")
                logger.info(f"     Category: {belief.category}")
        
        # Check if map-reduce was actually used
        if hasattr(extractor, 'map_reduce_extractor') and extractor.map_reduce_extractor:
            stats = extractor.map_reduce_extractor.get_processing_stats()
            logger.info("\n=== MAP-REDUCE STATISTICS ===")
            logger.info(f"Total batches: {stats.get('total_batches', 0)}")
            logger.info(f"Completed batches: {stats.get('completed_batches', 0)}")
            logger.info(f"Cached batches: {stats.get('cached_batches', 0)}")
            logger.info(f"Failed batches: {stats.get('failed_batches', 0)}")
            logger.info(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.1%}")
            logger.info(f"Average batch time: {stats.get('avg_processing_time_per_batch', 0):.1f}s")
        
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test"""
    success = asyncio.run(test_map_reduce_extraction())
    
    if success:
        print("\n✅ Map-Reduce extraction test passed!")
        sys.exit(0)
    else:
        print("\n❌ Map-Reduce extraction test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()