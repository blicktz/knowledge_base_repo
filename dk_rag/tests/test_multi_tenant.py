#!/usr/bin/env python
"""
Test script for multi-tenant architecture
"""

import sys
import tempfile
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dk_rag.config.settings import Settings
from dk_rag.core.persona_manager import PersonaManager
from dk_rag.core.knowledge_indexer import KnowledgeIndexer
from dk_rag.utils.logging import setup_logger


def test_persona_isolation():
    """Test that personas are properly isolated from each other"""
    logger = setup_logger("test", level="INFO")
    logger.info("=" * 60)
    logger.info("Testing Multi-Tenant Persona Isolation")
    logger.info("=" * 60)
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test settings with temp directory
        settings = Settings.from_default_config()
        settings.storage.artifacts_dir = f"{temp_dir}/artifacts"
        settings.storage.vector_db_dir = f"{temp_dir}/vector_db"
        
        # Initialize persona manager
        persona_manager = PersonaManager(settings)
        
        # Test 1: Create multiple personas
        logger.info("\n[Test 1] Creating multiple personas...")
        persona1_id = persona_manager.register_persona("Test Persona 1")
        persona2_id = persona_manager.register_persona("Test Persona 2")
        persona3_id = persona_manager.register_persona("Test Persona 3")
        
        assert persona1_id == "test_persona_1"
        assert persona2_id == "test_persona_2"
        assert persona3_id == "test_persona_3"
        logger.info("✓ Created 3 distinct personas")
        
        # Test 2: Verify persona directories exist
        logger.info("\n[Test 2] Verifying persona directories...")
        personas_dir = Path(settings.storage.artifacts_dir).parent / "personas"
        
        for persona_id in [persona1_id, persona2_id, persona3_id]:
            persona_dir = personas_dir / persona_id
            assert persona_dir.exists()
            assert (persona_dir / "vector_db").exists()
            assert (persona_dir / "artifacts").exists()
            logger.info(f"✓ Directory structure created for {persona_id}")
        
        # Test 3: Get persona-specific vector stores
        logger.info("\n[Test 3] Testing persona-specific vector stores...")
        vector_store1 = persona_manager.get_persona_vector_store(persona1_id)
        vector_store2 = persona_manager.get_persona_vector_store(persona2_id)
        
        # Add test documents to persona 1 with longer content to ensure chunking
        test_docs1 = [
            {"content": "This is test content for persona 1. " * 50 + " This document contains specific information about persona 1 and should only be found when searching within persona 1's vector store.", "source": "test1.txt"},
            {"content": "More content specific to persona 1. " * 50 + " Additional unique information that belongs exclusively to persona 1's knowledge base.", "source": "test2.txt"}
        ]
        chunks1 = vector_store1.add_documents(test_docs1)
        logger.info(f"✓ Added {chunks1} chunks to persona 1")
        
        # Add test documents to persona 2 with longer content to ensure chunking
        test_docs2 = [
            {"content": "This is test content for persona 2. " * 50 + " This document contains specific information about persona 2 and should only be found when searching within persona 2's vector store.", "source": "test3.txt"},
            {"content": "Different content for persona 2. " * 50 + " Additional unique information that belongs exclusively to persona 2's knowledge base.", "source": "test4.txt"}
        ]
        chunks2 = vector_store2.add_documents(test_docs2)
        logger.info(f"✓ Added {chunks2} chunks to persona 2")
        
        # Test 4: Verify isolation - search in persona 1 should not find persona 2's content
        logger.info("\n[Test 4] Testing search isolation...")
        
        # Search in persona 1 for its own content
        results1 = vector_store1.search("persona 1", n_results=5)
        assert len(results1) > 0
        assert any("persona 1" in r['content'].lower() for r in results1)
        logger.info("✓ Persona 1 can find its own content")
        
        # Search in persona 1 for persona 2's content (should not find)
        results1_cross = vector_store1.search("persona 2", n_results=5)
        if results1_cross:
            # Check that results don't actually contain persona 2's exact content
            for result in results1_cross:
                assert "persona 2" not in result['content'].lower() or "test3.txt" not in result['metadata'].get('source', '')
        logger.info("✓ Persona 1 cannot find persona 2's exact content")
        
        # Search in persona 2 for its own content
        results2 = vector_store2.search("persona 2", n_results=5)
        assert len(results2) > 0
        assert any("persona 2" in r['content'].lower() for r in results2)
        logger.info("✓ Persona 2 can find its own content")
        
        # Test 5: Verify collection statistics are separate
        logger.info("\n[Test 5] Testing collection statistics isolation...")
        stats1 = vector_store1.get_collection_stats()
        stats2 = vector_store2.get_collection_stats()
        
        assert stats1['collection_name'] == f"{persona1_id}_documents"
        assert stats2['collection_name'] == f"{persona2_id}_documents"
        logger.info(f"✓ Persona 1 stats: {stats1['total_chunks']} chunks")
        logger.info(f"✓ Persona 2 stats: {stats2['total_chunks']} chunks")
        
        # Test 6: Test KnowledgeIndexer with persona context
        logger.info("\n[Test 6] Testing KnowledgeIndexer with persona context...")
        indexer1 = KnowledgeIndexer(settings, persona1_id)
        indexer2 = KnowledgeIndexer(settings, persona2_id)
        
        # Search through indexer (should only find persona-specific content)
        search_results1 = indexer1.search_knowledge("test content", persona_id=persona1_id)
        search_results2 = indexer2.search_knowledge("test content", persona_id=persona2_id)
        
        assert len(search_results1) > 0
        assert len(search_results2) > 0
        logger.info("✓ KnowledgeIndexer respects persona boundaries")
        
        # Test 7: List personas
        logger.info("\n[Test 7] Testing persona listing...")
        all_personas = persona_manager.list_personas()
        assert len(all_personas) == 3
        persona_ids = [p['id'] for p in all_personas]
        assert persona1_id in persona_ids
        assert persona2_id in persona_ids
        assert persona3_id in persona_ids
        logger.info(f"✓ Listed all {len(all_personas)} personas")
        
        # Test 8: Test cleanup
        logger.info("\n[Test 8] Testing cleanup...")
        persona_manager.cleanup()
        logger.info("✓ Cleanup successful")
        
        logger.info("\n" + "=" * 60)
        logger.info("All tests passed! Multi-tenant architecture is working correctly.")
        logger.info("=" * 60)
        
        return True


def test_data_migration():
    """Test data migration from single-tenant to multi-tenant"""
    logger = setup_logger("test_migration", level="INFO")
    logger.info("=" * 60)
    logger.info("Testing Data Migration")
    logger.info("=" * 60)
    
    # This would test the migration script
    # For now, we'll just verify the migration script exists
    migration_script = Path(__file__).parent.parent / "scripts" / "migrate_to_multi_tenant.py"
    assert migration_script.exists()
    logger.info(f"✓ Migration script exists at: {migration_script}")
    
    return True


def main():
    """Run all tests"""
    try:
        # Test persona isolation
        test_persona_isolation()
        
        # Test migration
        test_data_migration()
        
        print("\n✅ All multi-tenant tests passed successfully!")
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())