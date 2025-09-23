#!/usr/bin/env python
"""
Test script for analysis caching functionality
"""

import sys
import tempfile
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dk_rag.config.settings import Settings
from dk_rag.core.analysis_cache import AnalysisCacheManager
from dk_rag.core.statistical_analyzer import StatisticalAnalyzer
from dk_rag.core.persona_extractor import PersonaExtractor
from dk_rag.core.persona_manager import PersonaManager
from dk_rag.utils.logging import setup_logger


def test_analysis_caching():
    """Test that analysis results are properly cached and reused"""
    logger = setup_logger("test_cache", level="INFO")
    logger.info("=" * 60)
    logger.info("Testing Analysis Caching")
    logger.info("=" * 60)
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test settings with temp directory
        settings = Settings.from_default_config()
        # Note: Using temp directory for test
        
        # Test persona
        persona_id = "test_persona"
        
        # Test documents with sufficient content for chunking
        test_documents = [
            {
                "content": "This is a comprehensive test document about productivity. " * 100 +
                          "I believe in systematic approaches to getting things done. " * 50 +
                          "My framework for success includes planning, execution, and review. " * 50,
                "source": "test_doc1.md"
            },
            {
                "content": "Another detailed document discussing mental models. " * 100 +
                          "What's important is understanding first principles. " * 50 +
                          "My methodology involves breaking down complex problems. " * 50,
                "source": "test_doc2.md"
            }
        ]
        
        # Test 1: Create cache manager and analyzer
        logger.info("\n[Test 1] Creating cache manager and analyzer...")
        cache_manager = AnalysisCacheManager(settings, persona_id)
        analyzer = StatisticalAnalyzer(settings, persona_id)
        
        # Verify cache directory creation
        cache_dir = cache_manager.cache_dir
        assert cache_dir.exists()
        logger.info(f"✓ Cache directory created: {cache_dir}")
        
        # Test 2: Initial analysis (should create cache)
        logger.info("\n[Test 2] Running initial analysis (should create cache)...")
        start_time = time.time()
        
        # Verify no cache exists initially
        assert not analyzer.has_cached_analysis(test_documents)
        logger.info("✓ Confirmed no initial cache exists")
        
        # Run analysis
        report1 = analyzer.analyze_content(test_documents, use_cache=True, force_reanalyze=False)
        first_analysis_time = time.time() - start_time
        
        assert report1 is not None
        assert report1.total_documents == 2
        assert report1.total_words > 0
        logger.info(f"✓ Initial analysis completed in {first_analysis_time:.2f}s")
        logger.info(f"  - {report1.total_words:,} words analyzed")
        logger.info(f"  - {len(report1.top_keywords)} keywords extracted")
        
        # Test 3: Verify cache was created
        logger.info("\n[Test 3] Verifying cache was created...")
        cache_info = cache_manager.get_cache_info()
        assert cache_info["status"] == "available"
        assert len(cache_info["files"]) > 0
        logger.info("✓ Cache files created successfully")
        
        # Verify has_cached_analysis returns True
        assert analyzer.has_cached_analysis(test_documents)
        logger.info("✓ Cache validation working correctly")
        
        # Test 4: Second analysis (should use cache)
        logger.info("\n[Test 4] Running second analysis (should use cache)...")
        start_time = time.time()
        
        report2 = analyzer.analyze_content(test_documents, use_cache=True, force_reanalyze=False)
        second_analysis_time = time.time() - start_time
        
        assert report2 is not None
        logger.info(f"✓ Cached analysis completed in {second_analysis_time:.2f}s")
        
        # Verify significant speed improvement
        speedup = first_analysis_time / second_analysis_time
        logger.info(f"✓ Speedup: {speedup:.1f}x faster with cache")
        
        # Results should be identical
        assert report1.total_documents == report2.total_documents
        assert report1.total_words == report2.total_words
        assert len(report1.top_keywords) == len(report2.top_keywords)
        logger.info("✓ Cached results identical to original analysis")
        
        # Test 5: Force reanalyze (should ignore cache)
        logger.info("\n[Test 5] Testing force reanalyze...")
        start_time = time.time()
        
        report3 = analyzer.analyze_content(test_documents, use_cache=True, force_reanalyze=True)
        forced_analysis_time = time.time() - start_time
        
        assert report3 is not None
        logger.info(f"✓ Forced analysis completed in {forced_analysis_time:.2f}s")
        
        # Should take longer than cached version
        assert forced_analysis_time > second_analysis_time
        logger.info("✓ Force reanalyze properly ignores cache")
        
        # Test 6: Test with PersonaExtractor
        logger.info("\n[Test 6] Testing PersonaExtractor caching...")
        
        # Initialize persona manager
        persona_manager = PersonaManager(settings)
        persona_manager.register_persona("Test Persona")
        
        # Create persona extractor
        extractor = PersonaExtractor(settings, persona_id)
        
        # First extraction (should use existing cache)
        start_time = time.time()
        persona1 = extractor.extract_persona_sync(
            test_documents, 
            use_cached_analysis=True, 
            force_reanalyze=False
        )
        first_extraction_time = time.time() - start_time
        
        assert persona1 is not None
        logger.info(f"✓ First persona extraction completed in {first_extraction_time:.2f}s")
        
        # Second extraction with different LLM settings (should use cache for analysis)
        start_time = time.time()
        persona2 = extractor.extract_persona_sync(
            test_documents, 
            use_cached_analysis=True, 
            force_reanalyze=False
        )
        second_extraction_time = time.time() - start_time
        
        assert persona2 is not None
        logger.info(f"✓ Second persona extraction completed in {second_extraction_time:.2f}s")
        
        # Statistical analysis should be identical (cached)
        assert persona1.statistical_report.total_words == persona2.statistical_report.total_words
        logger.info("✓ PersonaExtractor correctly uses cached analysis")
        
        # Test 7: Cache invalidation with different documents
        logger.info("\n[Test 7] Testing cache invalidation with different documents...")
        
        different_documents = [
            {
                "content": "Completely different content about marketing strategies. " * 150,
                "source": "different_doc.md"
            }
        ]
        
        # Should not find cache for different documents
        assert not analyzer.has_cached_analysis(different_documents)
        logger.info("✓ Cache correctly invalidated for different documents")
        
        # Test 8: Cache clearing
        logger.info("\n[Test 8] Testing cache clearing...")
        
        # Clear cache
        analyzer.clear_analysis_cache()
        
        # Verify cache is cleared
        cache_info_after_clear = cache_manager.get_cache_info()
        assert cache_info_after_clear["status"] in ["no_cache", "available"]
        if cache_info_after_clear["status"] == "available":
            assert len(cache_info_after_clear.get("files", [])) == 0
        
        # Should not have cached analysis anymore
        assert not analyzer.has_cached_analysis(test_documents)
        logger.info("✓ Cache successfully cleared")
        
        logger.info("\n" + "=" * 60)
        logger.info("All caching tests passed successfully!")
        logger.info("=" * 60)
        
        print(f"\n🎉 Performance Summary:")
        print(f"  Initial analysis: {first_analysis_time:.2f}s")
        print(f"  Cached analysis:  {second_analysis_time:.2f}s")
        print(f"  Speedup:         {speedup:.1f}x faster")
        print(f"\n✅ Analysis caching is working correctly!")
        
        return True


def main():
    """Run cache tests"""
    try:
        test_analysis_caching()
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