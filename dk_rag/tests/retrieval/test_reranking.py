"""
Unit tests for Cross-Encoder Reranking
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from langchain.schema import Document

from ...core.retrieval.reranker import CrossEncoderReranker, DualEncoderReranker


class MockReranker:
    """Mock reranker for testing local model functionality"""
    
    def __init__(self, model_name):
        self.model_name = model_name
    
    def rank(self, query, docs, doc_ids=None):
        """Mock ranking that returns results with scores"""
        if doc_ids is None:
            doc_ids = list(range(len(docs)))
        
        # Mock results with decreasing scores
        class MockResult:
            def __init__(self, doc_id, score):
                self.doc_id = doc_id
                self.score = score
        
        # Simulate relevance scoring based on query-doc similarity
        results = []
        for i, (doc_id, doc) in enumerate(zip(doc_ids, docs)):
            # Simple mock scoring: longer docs get higher scores
            score = max(0.1, (len(doc) / 100.0) - i * 0.1)
            results.append(MockResult(doc_id, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results


class MockCohereClient:
    """Mock Cohere client for testing API functionality"""
    
    def __init__(self, api_key):
        self.api_key = api_key
    
    def rerank(self, query, documents, model="rerank-english-v3.0", top_n=5):
        """Mock Cohere rerank API"""
        class MockCohereResult:
            def __init__(self, index, relevance_score):
                self.index = index
                self.relevance_score = relevance_score
        
        class MockCohereResponse:
            def __init__(self, results):
                self.results = results
        
        # Mock scoring based on document length and position
        results = []
        for i, doc in enumerate(documents[:top_n]):
            score = max(0.1, 0.9 - i * 0.1)  # Decreasing scores
            results.append(MockCohereResult(i, score))
        
        return MockCohereResponse(results)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    return [
        Document(
            page_content="Productivity frameworks help organize work and increase efficiency through structured approaches.",
            metadata={"source": "productivity", "doc_id": "doc1"}
        ),
        Document(
            page_content="Time management is crucial for success.",
            metadata={"source": "time", "doc_id": "doc2"}
        ),
        Document(
            page_content="Goal setting frameworks like SMART goals provide clear structure for achievement and accountability.",
            metadata={"source": "goals", "doc_id": "doc3"}
        ),
        Document(
            page_content="Focus and concentration techniques improve deep work capabilities.",
            metadata={"source": "focus", "doc_id": "doc4"}
        ),
        Document(
            page_content="Workflow optimization eliminates bottlenecks and streamlines business processes effectively.",
            metadata={"source": "workflow", "doc_id": "doc5"}
        )
    ]


class TestCrossEncoderReranker:
    """Test cases for CrossEncoderReranker"""
    
    @patch('dk_rag.core.retrieval.reranker.RERANKERS_AVAILABLE', True)
    @patch('dk_rag.core.retrieval.reranker.Reranker')
    def test_local_reranker_initialization(self, mock_reranker_class, temp_cache_dir):
        """Test initialization with local reranker"""
        mock_reranker_class.return_value = MockReranker("test-model")
        
        reranker = CrossEncoderReranker(
            model_name="test-model",
            use_cohere=False,
            cache_dir=temp_cache_dir
        )
        
        assert reranker.model_name == "test-model"
        assert reranker.reranker_type == "local"
        assert not reranker.use_cohere
        assert reranker.cache_dir.exists()
        mock_reranker_class.assert_called_once()
    
    @patch('dk_rag.core.retrieval.reranker.COHERE_AVAILABLE', True)
    @patch('dk_rag.core.retrieval.reranker.cohere.Client')
    def test_cohere_initialization(self, mock_cohere_client, temp_cache_dir):
        """Test initialization with Cohere API"""
        mock_cohere_client.return_value = MockCohereClient("test-key")
        
        reranker = CrossEncoderReranker(
            model_name="rerank-english-v3.0",
            use_cohere=True,
            cohere_api_key="test-key",
            cache_dir=temp_cache_dir
        )
        
        assert reranker.model_name == "rerank-english-v3.0"
        assert reranker.reranker_type == "cohere"
        assert reranker.use_cohere
        mock_cohere_client.assert_called_once_with("test-key")
    
    def test_cohere_initialization_without_key(self, temp_cache_dir):
        """Test that Cohere initialization fails without API key"""
        with pytest.raises(ValueError, match="Cohere API key required"):
            CrossEncoderReranker(
                use_cohere=True,
                cohere_api_key=None,
                cache_dir=temp_cache_dir
            )
    
    @patch('dk_rag.core.retrieval.reranker.RERANKERS_AVAILABLE', False)
    @patch('dk_rag.core.retrieval.reranker.COHERE_AVAILABLE', False)
    def test_initialization_without_packages(self, temp_cache_dir):
        """Test initialization fails when no reranking packages available"""
        with pytest.raises(ImportError, match="Neither rerankers nor cohere package is available"):
            CrossEncoderReranker(cache_dir=temp_cache_dir)
    
    @patch('dk_rag.core.retrieval.reranker.RERANKERS_AVAILABLE', True)
    @patch('dk_rag.core.retrieval.reranker.Reranker')
    def test_local_reranking(self, mock_reranker_class, sample_documents, temp_cache_dir):
        """Test local reranking functionality"""
        mock_reranker_class.return_value = MockReranker("test-model")
        
        reranker = CrossEncoderReranker(
            model_name="test-model",
            use_cohere=False,
            cache_dir=temp_cache_dir
        )
        
        query = "productivity and time management"
        top_k = 3
        
        reranked_docs = reranker.rerank(query, sample_documents, top_k=top_k)
        
        assert len(reranked_docs) == top_k
        assert all(isinstance(doc, Document) for doc in reranked_docs)
        
        # Check metadata was added
        for doc in reranked_docs:
            assert "rerank_score" in doc.metadata
            assert "reranked" in doc.metadata
            assert "reranker_model" in doc.metadata
            assert doc.metadata["reranked"] is True
            assert doc.metadata["reranker_model"] == "test-model"
    
    @patch('dk_rag.core.retrieval.reranker.COHERE_AVAILABLE', True)
    @patch('dk_rag.core.retrieval.reranker.cohere.Client')
    def test_cohere_reranking(self, mock_cohere_client, sample_documents, temp_cache_dir):
        """Test Cohere API reranking"""
        mock_client = MockCohereClient("test-key")
        mock_cohere_client.return_value = mock_client
        
        reranker = CrossEncoderReranker(
            use_cohere=True,
            cohere_api_key="test-key",
            cache_dir=temp_cache_dir
        )
        
        query = "goal setting frameworks"
        top_k = 2
        
        reranked_docs = reranker.rerank(query, sample_documents, top_k=top_k)
        
        assert len(reranked_docs) == top_k
        assert all(isinstance(doc, Document) for doc in reranked_docs)
        
        # Check metadata
        for doc in reranked_docs:
            assert "rerank_score" in doc.metadata
            assert "reranked" in doc.metadata
            assert doc.metadata["reranked"] is True
    
    @patch('dk_rag.core.retrieval.reranker.RERANKERS_AVAILABLE', True)
    @patch('dk_rag.core.retrieval.reranker.Reranker')
    def test_reranking_with_scores(self, mock_reranker_class, sample_documents, temp_cache_dir):
        """Test reranking returning scores"""
        mock_reranker_class.return_value = MockReranker("test-model")
        
        reranker = CrossEncoderReranker(
            model_name="test-model",
            cache_dir=temp_cache_dir
        )
        
        query = "workflow optimization"
        top_k = 3
        
        results = reranker.rerank(query, sample_documents, top_k=top_k, return_scores=True)
        
        assert len(results) == top_k
        assert all(isinstance(item, tuple) for item in results)
        assert all(len(item) == 2 for item in results)
        
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, (int, float))
            assert doc.metadata["rerank_score"] == score
    
    def test_empty_candidates_handling(self, temp_cache_dir):
        """Test handling of empty candidate list"""
        with patch('dk_rag.core.retrieval.reranker.RERANKERS_AVAILABLE', True), \
             patch('dk_rag.core.retrieval.reranker.Reranker', return_value=MockReranker("test")):
            
            reranker = CrossEncoderReranker(cache_dir=temp_cache_dir)
            
            result = reranker.rerank("query", [], top_k=5)
            assert result == []
    
    @patch('dk_rag.core.retrieval.reranker.RERANKERS_AVAILABLE', True)
    @patch('dk_rag.core.retrieval.reranker.Reranker')
    def test_reranking_logging(self, mock_reranker_class, sample_documents, temp_cache_dir):
        """Test that reranking operations are logged"""
        mock_reranker_class.return_value = MockReranker("test-model")
        
        reranker = CrossEncoderReranker(
            model_name="test-model",
            cache_dir=temp_cache_dir
        )
        
        query = "test query for logging"
        metadata = {"experiment": "unit_test"}
        
        reranker.rerank(query, sample_documents[:2], top_k=2, log_metadata=metadata)
        
        # Check log file was created
        log_files = list(Path(temp_cache_dir).glob("rerank_*.json"))
        assert len(log_files) == 1
        
        # Verify log content
        with open(log_files[0], 'r') as f:
            log_data = json.load(f)
        
        assert log_data["query"] == query
        assert log_data["num_candidates"] == 2
        assert log_data["component"] == "CrossEncoderReranker"
        assert log_data["metadata"]["experiment"] == "unit_test"
        assert "scores" in log_data
        assert "sample_candidates" in log_data
    
    @patch('dk_rag.core.retrieval.reranker.RERANKERS_AVAILABLE', True)
    @patch('dk_rag.core.retrieval.reranker.Reranker')
    def test_error_handling_local_reranking(self, mock_reranker_class, sample_documents, temp_cache_dir):
        """Test error handling in local reranking"""
        # Create mock that raises exception
        mock_reranker = Mock()
        mock_reranker.rank.side_effect = Exception("Reranking failed")
        mock_reranker_class.return_value = mock_reranker
        
        reranker = CrossEncoderReranker(cache_dir=temp_cache_dir)
        
        # Should handle error gracefully and return uniform scores
        results = reranker.rerank("query", sample_documents[:2], top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)
        # All docs should have same score (fallback behavior)
        scores = [doc.metadata["rerank_score"] for doc in results]
        assert all(score == 1.0 for score in scores)
    
    @patch('dk_rag.core.retrieval.reranker.RERANKERS_AVAILABLE', True)
    @patch('dk_rag.core.retrieval.reranker.Reranker')
    def test_batch_reranking(self, mock_reranker_class, sample_documents, temp_cache_dir):
        """Test batch reranking multiple queries"""
        mock_reranker_class.return_value = MockReranker("test-model")
        
        reranker = CrossEncoderReranker(cache_dir=temp_cache_dir)
        
        queries = ["productivity", "time management", "goal setting"]
        candidates_list = [sample_documents[:3], sample_documents[1:4], sample_documents[2:5]]
        
        results = reranker.batch_rerank(queries, candidates_list, top_k=2)
        
        assert len(results) == len(queries)
        
        for result in results:
            assert len(result) <= 2
            assert all(isinstance(doc, Document) for doc in result)
    
    @patch('dk_rag.core.retrieval.reranker.RERANKERS_AVAILABLE', True)
    @patch('dk_rag.core.retrieval.reranker.Reranker')
    def test_get_statistics(self, mock_reranker_class, temp_cache_dir):
        """Test statistics collection"""
        mock_reranker_class.return_value = MockReranker("test-model")
        
        reranker = CrossEncoderReranker(
            model_name="test-model",
            batch_size=16,
            cache_dir=temp_cache_dir
        )
        
        stats = reranker.get_statistics()
        
        assert stats["model"] == "test-model"
        assert stats["type"] == "local"
        assert stats["batch_size"] == 16
        assert "total_rerankings_logged" in stats
        assert stats["total_rerankings_logged"] == 0  # No operations yet


class TestDualEncoderReranker:
    """Test cases for DualEncoderReranker (faster alternative)"""
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_dual_encoder_initialization(self, mock_sentence_transformer):
        """Test dual encoder reranker initialization"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        reranker = DualEncoderReranker(
            model_name="test-model",
            device="cpu"
        )
        
        assert reranker.device == "cpu"
        assert reranker.model == mock_model
        mock_sentence_transformer.assert_called_once_with("test-model", device="cpu")
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_dual_encoder_reranking(self, mock_sentence_transformer, sample_documents):
        """Test dual encoder reranking functionality"""
        mock_model = Mock()
        mock_model.encode.side_effect = [
            [0.1, 0.2, 0.3],  # Query embedding
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]  # Document embeddings
        ]
        
        # Mock similarity calculation
        import torch
        mock_similarities = torch.tensor([0.9, 0.8, 0.7])
        mock_model.similarity.return_value = [mock_similarities]
        
        mock_sentence_transformer.return_value = mock_model
        
        reranker = DualEncoderReranker()
        
        query = "test query"
        candidates = sample_documents[:3]
        top_k = 2
        
        reranked_docs = reranker.rerank(query, candidates, top_k=top_k)
        
        assert len(reranked_docs) == top_k
        assert all(isinstance(doc, Document) for doc in reranked_docs)
        
        # Check metadata
        for doc in reranked_docs:
            assert "dual_encoder_score" in doc.metadata
            assert isinstance(doc.metadata["dual_encoder_score"], float)
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_dual_encoder_empty_candidates(self, mock_sentence_transformer):
        """Test dual encoder with empty candidates"""
        mock_sentence_transformer.return_value = Mock()
        
        reranker = DualEncoderReranker()
        result = reranker.rerank("query", [], top_k=5)
        
        assert result == []


class TestRerankerIntegration:
    """Integration tests for reranker components"""
    
    @patch('dk_rag.core.retrieval.reranker.RERANKERS_AVAILABLE', True)
    @patch('dk_rag.core.retrieval.reranker.Reranker')
    def test_end_to_end_reranking_flow(self, mock_reranker_class, sample_documents, temp_cache_dir):
        """Test complete reranking workflow"""
        mock_reranker_class.return_value = MockReranker("test-model")
        
        reranker = CrossEncoderReranker(
            model_name="test-model",
            cache_dir=temp_cache_dir,
            batch_size=8
        )
        
        query = "productivity frameworks and time management techniques"
        
        # Test different reranking operations
        
        # 1. Basic reranking
        basic_results = reranker.rerank(query, sample_documents, top_k=3)
        assert len(basic_results) == 3
        
        # 2. Reranking with scores
        scored_results = reranker.rerank(query, sample_documents, top_k=2, return_scores=True)
        assert len(scored_results) == 2
        
        # 3. Batch reranking
        batch_results = reranker.batch_rerank(
            [query, "goal setting"],
            [sample_documents[:3], sample_documents[2:5]],
            top_k=2
        )
        assert len(batch_results) == 2
        
        # Verify all results are valid
        all_docs = basic_results + [doc for doc, _ in scored_results]
        for batch in batch_results:
            all_docs.extend(batch)
        
        assert all(isinstance(doc, Document) for doc in all_docs)
        assert all("rerank_score" in doc.metadata for doc in all_docs)
    
    @patch('dk_rag.core.retrieval.reranker.RERANKERS_AVAILABLE', True)
    @patch('dk_rag.core.retrieval.reranker.Reranker')
    def test_score_ordering_consistency(self, mock_reranker_class, sample_documents, temp_cache_dir):
        """Test that reranked documents are properly ordered by score"""
        mock_reranker_class.return_value = MockReranker("test-model")
        
        reranker = CrossEncoderReranker(cache_dir=temp_cache_dir)
        
        query = "productivity and efficiency"
        results = reranker.rerank(query, sample_documents, top_k=4, return_scores=True)
        
        # Verify scores are in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        
        # Verify metadata consistency
        for doc, score in results:
            assert abs(doc.metadata["rerank_score"] - score) < 0.001
    
    @patch('dk_rag.core.retrieval.reranker.RERANKERS_AVAILABLE', True)
    @patch('dk_rag.core.retrieval.reranker.Reranker')
    def test_large_candidate_set_handling(self, mock_reranker_class, temp_cache_dir):
        """Test reranking with large candidate sets"""
        mock_reranker_class.return_value = MockReranker("test-model")
        
        # Create large set of documents
        large_doc_set = []
        for i in range(50):
            large_doc_set.append(Document(
                page_content=f"Document {i} about productivity and time management" * (i % 5 + 1),
                metadata={"doc_id": f"doc_{i}"}
            ))
        
        reranker = CrossEncoderReranker(
            cache_dir=temp_cache_dir,
            batch_size=16
        )
        
        query = "productivity techniques"
        results = reranker.rerank(query, large_doc_set, top_k=10)
        
        assert len(results) == 10
        assert all(isinstance(doc, Document) for doc in results)
        
        # Verify top results have higher scores
        scores = [doc.metadata["rerank_score"] for doc in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_reranker_device_detection(self, temp_cache_dir):
        """Test automatic device detection"""
        with patch('dk_rag.core.retrieval.reranker.RERANKERS_AVAILABLE', True), \
             patch('dk_rag.core.retrieval.reranker.Reranker', return_value=MockReranker("test")), \
             patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            
            reranker = CrossEncoderReranker(cache_dir=temp_cache_dir)
            assert reranker.device == "cpu"
        
        with patch('dk_rag.core.retrieval.reranker.RERANKERS_AVAILABLE', True), \
             patch('dk_rag.core.retrieval.reranker.Reranker', return_value=MockReranker("test")), \
             patch('torch.cuda.is_available', return_value=True):
            
            reranker = CrossEncoderReranker(cache_dir=temp_cache_dir)
            assert reranker.device == "cuda"