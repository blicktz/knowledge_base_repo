"""
Integration tests for Advanced Retrieval Pipeline
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from langchain.schema import Document

from ...core.retrieval.advanced_pipeline import AdvancedRetrievalPipeline
from ...core.retrieval.hyde_retriever import HyDERetriever
from ...core.retrieval.hybrid_retriever import HybridRetriever
from ...core.retrieval.reranker import CrossEncoderReranker
from ...prompts.hyde_prompts import HYDE_PROMPTS


# Mock components for testing
class MockLLM:
    """Mock LLM that generates hypotheses"""
    def invoke(self, prompt):
        return "This is a comprehensive hypothesis about productivity frameworks and time management techniques."
    
    def __str__(self):
        return "MockLLM"


class MockEmbeddings:
    """Mock embeddings model"""
    def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4, 0.5]


class MockVectorStore:
    """Mock vector store with sample documents"""
    def __init__(self):
        self.documents = [
            Document(
                page_content="Productivity frameworks help organize work efficiently through structured methodologies.",
                metadata={"doc_id": "doc1", "source": "productivity", "topic": "frameworks"}
            ),
            Document(
                page_content="Time management techniques like time blocking maximize focus and minimize distractions.",
                metadata={"doc_id": "doc2", "source": "time_mgmt", "topic": "techniques"}
            ),
            Document(
                page_content="Goal setting using SMART criteria ensures clear, measurable objectives.",
                metadata={"doc_id": "doc3", "source": "goals", "topic": "smart_goals"}
            ),
            Document(
                page_content="Deep work practices require sustained focus and elimination of interruptions.",
                metadata={"doc_id": "doc4", "source": "focus", "topic": "deep_work"}
            ),
            Document(
                page_content="Workflow automation reduces repetitive tasks and increases efficiency.",
                metadata={"doc_id": "doc5", "source": "automation", "topic": "workflow"}
            )
        ]
    
    def similarity_search(self, query, k=4):
        return self.documents[:k]
    
    def similarity_search_by_vector(self, embedding, k=4):
        return self.documents[:k]
    
    def similarity_search_with_score(self, query, k=4):
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        return [(doc, scores[i]) for i, doc in enumerate(self.documents[:k])]


class MockBM25Store:
    """Mock BM25 store"""
    def search(self, query, k=4):
        scores = [3.5, 2.8, 2.1, 1.4, 0.9]
        doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        return [(doc_ids[i], scores[i]) for i in range(min(k, len(scores)))]
    
    def get_document_by_id(self, doc_id):
        doc_map = {
            "doc1": "Productivity frameworks help organize work efficiently through structured methodologies.",
            "doc2": "Time management techniques like time blocking maximize focus and minimize distractions.",
            "doc3": "Goal setting using SMART criteria ensures clear, measurable objectives.",
            "doc4": "Deep work practices require sustained focus and elimination of interruptions.",
            "doc5": "Workflow automation reduces repetitive tasks and increases efficiency."
        }
        return doc_map.get(doc_id)


class MockReranker:
    """Mock reranker that returns results with scores"""
    def rank(self, query, docs, doc_ids=None):
        class MockResult:
            def __init__(self, doc_id, score):
                self.doc_id = doc_id
                self.score = score
        
        # Return mock results with decreasing scores
        results = []
        for i in range(len(docs)):
            score = max(0.1, 0.95 - i * 0.15)
            results.append(MockResult(i, score))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_components(temp_cache_dir):
    """Create mock pipeline components"""
    # Mock HyDE retriever
    hyde = Mock(spec=HyDERetriever)
    hyde.generate_hypothesis.return_value = "Comprehensive hypothesis about productivity and time management."
    
    # Add vector store for fallback
    mock_vector_store = MockVectorStore()
    hyde.vector_store = mock_vector_store
    
    # Mock hybrid retriever
    hybrid = Mock(spec=HybridRetriever)
    all_mock_docs = MockVectorStore().documents[:5]
    for i, doc in enumerate(all_mock_docs):
        doc.metadata.update({
            "hybrid_score": 0.9 - i * 0.1,
            "bm25_score": 0.8 - i * 0.1,
            "vector_score": 0.7 - i * 0.1,
            "retrieval_method": "hybrid"
        })
    
    def mock_hybrid_search(query, k=5, bm25_k=None, vector_k=None):
        return all_mock_docs[:k]
    hybrid.search.side_effect = mock_hybrid_search
    
    # Mock reranker
    reranker = Mock(spec=CrossEncoderReranker)
    def mock_rerank(query, docs, top_k=5, log_metadata=None):
        # Return top k documents after reranking
        reranked_docs = docs[:top_k]
        for i, doc in enumerate(reranked_docs):
            doc.metadata["rerank_score"] = 0.95 - i * 0.1
            doc.metadata["reranked"] = True
            doc.metadata["pipeline"] = "advanced_retrieval"  # Add expected metadata
        return reranked_docs
    reranker.rerank.side_effect = mock_rerank
    
    return {
        "hyde": hyde,
        "hybrid": hybrid,
        "reranker": reranker,
        "cache_dir": temp_cache_dir
    }


@pytest.fixture
def pipeline(mock_components):
    """Create advanced pipeline for testing"""
    return AdvancedRetrievalPipeline(
        hyde_retriever=mock_components["hyde"],
        hybrid_retriever=mock_components["hybrid"],
        reranker=mock_components["reranker"],
        cache_dir=mock_components["cache_dir"],
        enable_hyde=True,
        enable_hybrid=True,
        enable_reranking=True
    )


class TestAdvancedRetrievalPipeline:
    """Test cases for the advanced retrieval pipeline"""
    
    def test_pipeline_initialization(self, mock_components):
        """Test pipeline initialization"""
        pipeline = AdvancedRetrievalPipeline(
            hyde_retriever=mock_components["hyde"],
            hybrid_retriever=mock_components["hybrid"],
            reranker=mock_components["reranker"],
            cache_dir=mock_components["cache_dir"]
        )
        
        assert pipeline.hyde == mock_components["hyde"]
        assert pipeline.hybrid == mock_components["hybrid"]
        assert pipeline.reranker == mock_components["reranker"]
        assert pipeline.enable_hyde is True
        assert pipeline.enable_hybrid is True
        assert pipeline.enable_reranking is True
        assert pipeline.cache_dir.exists()
    
    def test_component_enabling_disabling(self, pipeline):
        """Test enabling/disabling pipeline components"""
        # Test disabling components
        pipeline.enable_component("hyde", False)
        pipeline.enable_component("hybrid", False)
        pipeline.enable_component("reranking", False)
        
        assert pipeline.enable_hyde is False
        assert pipeline.enable_hybrid is False
        assert pipeline.enable_reranking is False
        
        # Test re-enabling
        pipeline.enable_component("hyde", True)
        assert pipeline.enable_hyde is True
        
        # Test invalid component name
        with pytest.raises(ValueError):
            pipeline.enable_component("invalid_component", True)
    
    def test_full_pipeline_retrieval(self, pipeline):
        """Test complete pipeline execution"""
        query = "What are the best productivity frameworks?"
        k = 3
        
        documents = pipeline.retrieve(query, k=k)
        
        assert len(documents) == k
        assert all(isinstance(doc, Document) for doc in documents)
        
        # Verify pipeline metadata was added
        for doc in documents:
            assert "pipeline" in doc.metadata
            assert doc.metadata["pipeline"] == "advanced"
            assert "hyde_used" in doc.metadata
            assert "hybrid_used" in doc.metadata
            assert "reranked" in doc.metadata
            assert "original_query" in doc.metadata
            assert doc.metadata["original_query"] == query
        
        # Verify component calls
        pipeline.hyde.generate_hypothesis.assert_called_once()
        pipeline.hybrid.search.assert_called_once()
        pipeline.reranker.rerank.assert_called_once()
    
    def test_hyde_disabled(self, pipeline):
        """Test pipeline with HyDE disabled"""
        pipeline.enable_hyde = False
        
        query = "time management techniques"
        documents = pipeline.retrieve(query, k=2)
        
        assert len(documents) == 2
        
        # HyDE should not be called when disabled
        pipeline.hyde.generate_hypothesis.assert_not_called()
        
        # But hybrid and reranking should still work
        pipeline.hybrid.search.assert_called_once()
        pipeline.reranker.rerank.assert_called_once()
    
    def test_hybrid_disabled(self, pipeline):
        """Test pipeline with hybrid search disabled"""
        pipeline.enable_hybrid = False
        
        # Mock vector store for fallback
        vector_store = MockVectorStore()
        pipeline.hyde.vector_store = vector_store
        
        query = "goal setting frameworks"
        documents = pipeline.retrieve(query, k=2)
        
        assert len(documents) == 2
        
        # HyDE should still be called
        pipeline.hyde.generate_hypothesis.assert_called_once()
        
        # Hybrid should not be called when disabled
        pipeline.hybrid.search.assert_not_called()
        
        # Reranking should still work
        pipeline.reranker.rerank.assert_called_once()
    
    def test_reranking_disabled(self, pipeline):
        """Test pipeline with reranking disabled"""
        pipeline.enable_reranking = False
        
        query = "workflow optimization"
        k = 4
        
        documents = pipeline.retrieve(query, k=k)
        
        assert len(documents) <= k
        
        # HyDE and hybrid should be called
        pipeline.hyde.generate_hypothesis.assert_called_once()
        pipeline.hybrid.search.assert_called_once()
        
        # Reranking should not be called when disabled
        pipeline.reranker.rerank.assert_not_called()
    
    def test_all_components_disabled(self, pipeline):
        """Test pipeline with all advanced components disabled"""
        pipeline.enable_hyde = False
        pipeline.enable_hybrid = False
        pipeline.enable_reranking = False
        
        # Mock vector store for basic fallback
        vector_store = MockVectorStore()
        pipeline.hyde.vector_store = vector_store
        
        query = "basic search query"
        documents = pipeline.retrieve(query, k=2)
        
        # Should still return results via fallback
        assert len(documents) <= 2
        
        # None of the advanced components should be called
        pipeline.hyde.generate_hypothesis.assert_not_called()
        pipeline.hybrid.search.assert_not_called()
        pipeline.reranker.rerank.assert_not_called()
    
    def test_custom_retrieval_parameters(self, pipeline):
        """Test pipeline with custom retrieval parameters"""
        query = "productivity and efficiency"
        k = 2
        retrieval_k = 10
        hyde_prompt_type = "detailed_explanation"
        
        documents = pipeline.retrieve(
            query,
            k=k,
            retrieval_k=retrieval_k,
            hyde_prompt_type=hyde_prompt_type
        )
        
        assert len(documents) == k
        
        # Verify HyDE was called with custom prompt template
        pipeline.hyde.generate_hypothesis.assert_called_once()
        call_args = pipeline.hyde.generate_hypothesis.call_args
        # Check that the prompt_template argument matches the expected prompt
        assert 'prompt_template' in str(call_args)
        assert hyde_prompt_type in str(call_args) or "detailed" in str(call_args)
    
    def test_component_override_parameters(self, pipeline):
        """Test pipeline with component-specific overrides"""
        query = "test query"
        documents = pipeline.retrieve(
            query,
            k=3,
            use_hyde=False,  # Override HyDE
            use_hybrid=True,  # Override hybrid
            use_reranking=True  # Override reranking
        )
        
        assert len(documents) == 3
        
        # HyDE should not be called due to override
        pipeline.hyde.generate_hypothesis.assert_not_called()
        
        # But hybrid and reranking should still work
        pipeline.hybrid.search.assert_called_once()
        pipeline.reranker.rerank.assert_called_once()
    
    def test_pipeline_logging(self, pipeline, temp_cache_dir):
        """Test that pipeline execution is logged"""
        query = "test query for logging"
        metadata = {"experiment": "pipeline_test"}
        
        pipeline.retrieve(query, k=2, metadata=metadata)
        
        # Check that pipeline log was created
        log_files = list(Path(temp_cache_dir).glob("pipeline_*.json"))
        assert len(log_files) == 1
        
        # Verify log content
        with open(log_files[0], 'r') as f:
            log_data = json.load(f)
        
        assert log_data["query"] == query
        assert log_data["component"] == "AdvancedRetrievalPipeline"
        assert "timings" in log_data
        assert "document_counts" in log_data
        assert "configuration" in log_data
        assert log_data["metadata"]["experiment"] == "pipeline_test"
    
    def test_error_handling_and_fallback(self, mock_components):
        """Test error handling with fallback to basic search"""
        # Create pipeline with components that raise exceptions
        hyde_error = Mock(spec=HyDERetriever)
        hyde_error.generate_hypothesis.side_effect = Exception("HyDE failed")
        hyde_error.vector_store = MockVectorStore()
        
        hybrid_error = Mock(spec=HybridRetriever)
        hybrid_error.search.side_effect = Exception("Hybrid search failed")
        
        pipeline = AdvancedRetrievalPipeline(
            hyde_retriever=hyde_error,
            hybrid_retriever=hybrid_error,
            reranker=mock_components["reranker"],
            cache_dir=mock_components["cache_dir"]
        )
        
        query = "error test query"
        documents = pipeline.retrieve(query, k=2)
        
        # Should fall back to basic search
        assert len(documents) <= 2
        assert all(isinstance(doc, Document) for doc in documents)
    
    def test_batch_retrieval(self, pipeline):
        """Test batch retrieval for multiple queries"""
        queries = [
            "productivity frameworks",
            "time management techniques",
            "goal setting methods"
        ]
        k = 2
        
        results = pipeline.batch_retrieve(queries, k=k)
        
        assert len(results) == len(queries)
        
        for result in results:
            assert len(result) <= k
            assert all(isinstance(doc, Document) for doc in result)
        
        # Each query should trigger the pipeline
        assert pipeline.hyde.generate_hypothesis.call_count == len(queries)
        assert pipeline.hybrid.search.call_count == len(queries)
        assert pipeline.reranker.rerank.call_count == len(queries)
    
    def test_get_statistics(self, pipeline, temp_cache_dir):
        """Test statistics collection"""
        # Execute some retrievals to generate stats
        queries = ["query1", "query2"]
        for query in queries:
            pipeline.retrieve(query, k=2)
        
        stats = pipeline.get_statistics()
        
        assert "configuration" in stats
        assert stats["configuration"]["hyde_enabled"] is True
        assert stats["configuration"]["hybrid_enabled"] is True
        assert stats["configuration"]["reranking_enabled"] is True
        
        assert "total_executions" in stats
        assert stats["total_executions"] == len(queries)
        
        # Should have timing statistics
        if "average_timings" in stats:
            assert isinstance(stats["average_timings"], dict)


class TestPipelineIntegrationScenarios:
    """Integration test scenarios for real-world usage"""
    
    @patch('dk_rag.core.retrieval.reranker.RERANKERS_AVAILABLE', True)
    @patch('dk_rag.core.retrieval.reranker.Reranker')
    def test_realistic_pipeline_flow(self, mock_reranker_class, temp_cache_dir):
        """Test with more realistic component implementations"""
        from ...tests.retrieval.test_reranking import MockReranker
        mock_reranker_class.return_value = MockReranker("test-model")
        
        # Create more realistic components
        llm = MockLLM()
        embeddings = MockEmbeddings()
        vector_store = MockVectorStore()
        bm25_store = MockBM25Store()
        
        # Create real component instances
        hyde = HyDERetriever(llm, embeddings, vector_store, cache_dir=temp_cache_dir)
        hybrid = HybridRetriever(bm25_store, vector_store)
        reranker = CrossEncoderReranker(cache_dir=temp_cache_dir)
        
        # Create pipeline
        pipeline = AdvancedRetrievalPipeline(hyde, hybrid, reranker, temp_cache_dir)
        
        # Test realistic queries
        queries = [
            "What are the most effective productivity frameworks?",
            "How can I improve my time management skills?",
            "What goal setting techniques work best for entrepreneurs?"
        ]
        
        for query in queries:
            documents = pipeline.retrieve(query, k=3)
            
            assert len(documents) == 3
            assert all(isinstance(doc, Document) for doc in documents)
            
            # Verify complete pipeline metadata
            for doc in documents:
                assert "pipeline" in doc.metadata
                assert "hyde_used" in doc.metadata
                assert "hybrid_used" in doc.metadata
                assert "reranked" in doc.metadata
    
    def test_performance_characteristics(self, pipeline):
        """Test pipeline performance characteristics"""
        import time
        
        query = "productivity and time management best practices"
        
        # Measure retrieval time
        start_time = time.time()
        documents = pipeline.retrieve(query, k=5)
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds for mocks)
        assert elapsed_time < 5.0
        assert len(documents) == 5
        
        # Test batch performance
        queries = [f"query {i}" for i in range(5)]
        
        start_time = time.time()
        batch_results = pipeline.batch_retrieve(queries, k=3)
        batch_elapsed = time.time() - start_time
        
        # Batch should complete in reasonable time
        assert batch_elapsed < 10.0
        assert len(batch_results) == 5
        assert all(len(result) == 3 for result in batch_results)
    
    def test_different_configuration_combinations(self, mock_components, temp_cache_dir):
        """Test various pipeline configuration combinations"""
        configurations = [
            {"hyde": True, "hybrid": True, "reranking": True},   # Full pipeline
            {"hyde": True, "hybrid": True, "reranking": False},  # No reranking
            {"hyde": True, "hybrid": False, "reranking": True},  # No hybrid
            {"hyde": False, "hybrid": True, "reranking": True},  # No HyDE
            {"hyde": False, "hybrid": False, "reranking": False} # Basic only
        ]
        
        for config in configurations:
            pipeline = AdvancedRetrievalPipeline(
                hyde_retriever=mock_components["hyde"],
                hybrid_retriever=mock_components["hybrid"],
                reranker=mock_components["reranker"],
                cache_dir=temp_cache_dir,
                enable_hyde=config["hyde"],
                enable_hybrid=config["hybrid"],
                enable_reranking=config["reranking"]
            )
            
            query = "test configuration"
            documents = pipeline.retrieve(query, k=2)
            
            # All configurations should return results
            assert len(documents) <= 2
            assert all(isinstance(doc, Document) for doc in documents)
            
            # Verify configuration is reflected in metadata
            for doc in documents:
                assert doc.metadata["hyde_used"] == config["hyde"]
                assert doc.metadata["hybrid_used"] == config["hybrid"]
    
    def test_edge_cases_and_error_conditions(self, pipeline):
        """Test edge cases and error conditions"""
        # Test empty query
        empty_results = pipeline.retrieve("", k=3)
        assert isinstance(empty_results, list)
        
        # Test very long query
        long_query = "productivity " * 100
        long_results = pipeline.retrieve(long_query, k=2)
        assert len(long_results) <= 2
        
        # Test k=0
        zero_results = pipeline.retrieve("test", k=0)
        assert len(zero_results) == 0
        
        # Test very large k
        large_k_results = pipeline.retrieve("test", k=1000)
        assert isinstance(large_k_results, list)
        # Should not crash, even if fewer documents available
        
        # Test special characters in query
        special_query = "test@#$%^&*()query"
        special_results = pipeline.retrieve(special_query, k=1)
        assert isinstance(special_results, list)