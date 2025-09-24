"""
Unit tests for Hybrid Search (BM25 + Vector) Retriever
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock

from langchain.schema import Document

from ...core.retrieval.hybrid_retriever import HybridRetriever
from ...data.storage.bm25_store import BM25Store


class MockVectorStore:
    """Mock vector store for testing"""
    
    def __init__(self):
        self.documents = [
            Document(
                page_content="Productivity frameworks help organize work and increase efficiency.",
                metadata={"doc_id": "doc1", "source": "productivity_guide"}
            ),
            Document(
                page_content="Time management techniques include the Pomodoro Technique and time blocking.",
                metadata={"doc_id": "doc2", "source": "time_management"}
            ),
            Document(
                page_content="Goal setting frameworks like SMART goals provide structure for achievement.",
                metadata={"doc_id": "doc3", "source": "goal_setting"}
            ),
            Document(
                page_content="Workflow optimization involves eliminating bottlenecks and streamlining processes.",
                metadata={"doc_id": "doc4", "source": "workflow"}
            )
        ]
    
    def similarity_search(self, query, k=4):
        """Mock similarity search"""
        return self.documents[:k]
    
    def similarity_search_with_score(self, query, k=4):
        """Mock similarity search with scores"""
        scores = [0.9, 0.8, 0.7, 0.6]
        return [(doc, scores[i]) for i, doc in enumerate(self.documents[:k])]


class MockBM25Store:
    """Mock BM25 store for testing"""
    
    def __init__(self):
        self.documents = [
            "Productivity frameworks help organize work and increase efficiency.",
            "Time management techniques include the Pomodoro Technique and time blocking.",
            "Goal setting frameworks like SMART goals provide structure for achievement.",
            "Workflow optimization involves eliminating bottlenecks and streamlining processes."
        ]
        self.doc_ids = ["doc1", "doc2", "doc3", "doc4"]
    
    def search(self, query, k=4):
        """Mock BM25 search returning (doc_id, score) pairs"""
        # Simulate BM25 scores based on keyword matching
        scores = []
        query_words = set(query.lower().split())
        
        for i, doc in enumerate(self.documents):
            doc_words = set(doc.lower().split())
            score = len(query_words.intersection(doc_words)) * 2.5
            if score > 0:
                scores.append((self.doc_ids[i], score))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def get_document_by_id(self, doc_id):
        """Mock document retrieval by ID"""
        try:
            idx = self.doc_ids.index(doc_id)
            return self.documents[idx]
        except ValueError:
            return None
    
    def get_statistics(self):
        """Mock statistics"""
        return {
            "status": "built",
            "num_documents": len(self.documents),
            "avg_doc_length": 12.5
        }


@pytest.fixture
def mock_bm25_store():
    """Create mock BM25 store"""
    return MockBM25Store()


@pytest.fixture
def mock_vector_store():
    """Create mock vector store"""
    return MockVectorStore()


@pytest.fixture
def hybrid_retriever(mock_bm25_store, mock_vector_store):
    """Create hybrid retriever for testing"""
    return HybridRetriever(
        bm25_store=mock_bm25_store,
        vector_store=mock_vector_store,
        bm25_weight=0.4,
        vector_weight=0.6
    )


class TestHybridRetriever:
    """Test cases for hybrid retriever"""
    
    def test_initialization(self, hybrid_retriever):
        """Test hybrid retriever initialization"""
        assert hybrid_retriever.bm25_store is not None
        assert hybrid_retriever.vector_store is not None
        assert hybrid_retriever.bm25_weight == 0.4
        assert hybrid_retriever.vector_weight == 0.6
    
    def test_weight_normalization(self, mock_bm25_store, mock_vector_store):
        """Test automatic weight normalization"""
        # Create retriever with weights that don't sum to 1.0
        retriever = HybridRetriever(
            bm25_store=mock_bm25_store,
            vector_store=mock_vector_store,
            bm25_weight=0.3,
            vector_weight=0.4  # Sum = 0.7
        )
        
        # Weights should be normalized
        assert abs(retriever.bm25_weight + retriever.vector_weight - 1.0) < 0.001
        assert abs(retriever.bm25_weight - 0.3/0.7) < 0.001
        assert abs(retriever.vector_weight - 0.4/0.7) < 0.001
    
    def test_score_normalization(self, hybrid_retriever):
        """Test score normalization"""
        # Test with various score ranges
        scores1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized1 = hybrid_retriever._normalize_scores(scores1)
        
        assert min(normalized1) == 0.0
        assert max(normalized1) == 1.0
        assert len(normalized1) == len(scores1)
        
        # Test with negative scores
        scores2 = [-2.0, -1.0, 0.0, 1.0, 2.0]
        normalized2 = hybrid_retriever._normalize_scores(scores2)
        
        assert min(normalized2) == 0.0
        assert max(normalized2) == 1.0
        
        # Test with identical scores
        scores3 = [5.0, 5.0, 5.0]
        normalized3 = hybrid_retriever._normalize_scores(scores3)
        
        assert all(score == 0.5 for score in normalized3)
        
        # Test with empty list
        assert hybrid_retriever._normalize_scores([]) == []
    
    def test_basic_search(self, hybrid_retriever):
        """Test basic hybrid search"""
        query = "productivity frameworks"
        k = 3
        
        documents = hybrid_retriever.search(query, k=k)
        
        assert len(documents) <= k
        assert all(isinstance(doc, Document) for doc in documents)
        
        # Check that hybrid metadata was added
        for doc in documents:
            assert "retrieval_method" in doc.metadata
            assert doc.metadata["retrieval_method"] == "hybrid"
            assert "hybrid_score" in doc.metadata
            assert "bm25_score" in doc.metadata
            assert "vector_score" in doc.metadata
    
    def test_search_with_scores(self, hybrid_retriever):
        """Test hybrid search returning scores"""
        query = "time management techniques"
        k = 2
        
        results = hybrid_retriever.search_with_scores(query, k=k)
        
        assert len(results) <= k
        assert all(isinstance(item, tuple) for item in results)
        assert all(len(item) == 2 for item in results)
        
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, (int, float))
            assert score >= 0  # Normalized scores should be non-negative
    
    def test_custom_retrieval_counts(self, hybrid_retriever):
        """Test hybrid search with custom BM25 and vector retrieval counts"""
        query = "goal setting"
        k = 2
        bm25_k = 5
        vector_k = 4
        
        documents = hybrid_retriever.search(
            query,
            k=k,
            bm25_k=bm25_k,
            vector_k=vector_k
        )
        
        assert len(documents) <= k
        assert all(isinstance(doc, Document) for doc in documents)
    
    def test_custom_weights(self, hybrid_retriever):
        """Test hybrid search with custom weights"""
        query = "workflow optimization"
        k = 3
        
        # Test with BM25-heavy weighting
        documents1 = hybrid_retriever.search(
            query,
            k=k,
            bm25_weight=0.8,
            vector_weight=0.2
        )
        
        # Test with vector-heavy weighting
        documents2 = hybrid_retriever.search(
            query,
            k=k,
            bm25_weight=0.2,
            vector_weight=0.8
        )
        
        assert len(documents1) <= k
        assert len(documents2) <= k
        
        # Check that different weights might produce different results
        # (In this mock case, they might be the same, but the mechanism works)
        for doc in documents1:
            assert "hybrid_score" in doc.metadata
        
        for doc in documents2:
            assert "hybrid_score" in doc.metadata
    
    def test_result_fusion_logic(self, hybrid_retriever):
        """Test the result fusion mechanism"""
        # Create specific mock results for testing fusion
        bm25_results = [("doc1", 3.0), ("doc2", 2.0), ("doc3", 1.0)]
        vector_results = [
            (Document(page_content="Doc 1 content", metadata={"doc_id": "doc1"}), 0.9),
            (Document(page_content="Doc 2 content", metadata={"doc_id": "doc2"}), 0.7),
            (Document(page_content="Doc 4 content", metadata={"doc_id": "doc4"}), 0.8)
        ]
        
        fused = hybrid_retriever._fuse_results(bm25_results, vector_results)
        
        assert len(fused) > 0
        assert all(isinstance(item, tuple) for item in fused)
        assert all(len(item) == 2 for item in fused)
        
        # Check that results are sorted by combined score
        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)
        
        # Check metadata was added
        for doc, score in fused:
            assert "hybrid_score" in doc.metadata
            assert "bm25_score" in doc.metadata
            assert "vector_score" in doc.metadata
            assert doc.metadata["hybrid_score"] == score
    
    def test_reciprocal_rank_fusion(self, hybrid_retriever):
        """Test Reciprocal Rank Fusion method"""
        query = "productivity techniques"
        k = 3
        
        documents = hybrid_retriever.reciprocal_rank_fusion(query, k=k)
        
        assert len(documents) <= k
        assert all(isinstance(doc, Document) for doc in documents)
        
        # Check RRF-specific metadata
        for doc in documents:
            assert "rrf_score" in doc.metadata
            assert "retrieval_method" in doc.metadata
            assert doc.metadata["retrieval_method"] == "hybrid_rrf"
    
    def test_rrf_with_custom_k(self, hybrid_retriever):
        """Test RRF with custom k parameter"""
        query = "frameworks"
        k = 2
        rrf_k = 100  # Different from default 60
        
        documents = hybrid_retriever.reciprocal_rank_fusion(
            query,
            k=k,
            rrf_k=rrf_k
        )
        
        assert len(documents) <= k
        assert all(isinstance(doc, Document) for doc in documents)
    
    def test_empty_results_handling(self, mock_vector_store):
        """Test handling of empty BM25 results"""
        # Create BM25 store that returns no results
        empty_bm25 = Mock()
        empty_bm25.search.return_value = []
        empty_bm25.get_document_by_id.return_value = None
        
        retriever = HybridRetriever(
            bm25_store=empty_bm25,
            vector_store=mock_vector_store
        )
        
        documents = retriever.search("query", k=3)
        
        # Should still return vector results
        assert len(documents) > 0
        
        # Results should be from vector store only
        for doc in documents:
            assert doc.metadata["vector_score"] > 0
            assert doc.metadata["bm25_score"] == 0
    
    def test_vector_store_without_scores(self, mock_bm25_store):
        """Test with vector store that doesn't support similarity_search_with_score"""
        # Create vector store without score method
        vector_store_no_scores = Mock()
        vector_store_no_scores.similarity_search.return_value = [
            Document(page_content="Test doc", metadata={"doc_id": "test"})
        ]
        # Simulate missing similarity_search_with_score method
        delattr(vector_store_no_scores, 'similarity_search_with_score')
        
        retriever = HybridRetriever(
            bm25_store=mock_bm25_store,
            vector_store=vector_store_no_scores
        )
        
        documents = retriever.search("query", k=2)
        
        # Should still work with default scores
        assert len(documents) > 0
        for doc in documents:
            assert "hybrid_score" in doc.metadata
    
    def test_get_statistics(self, hybrid_retriever):
        """Test statistics retrieval"""
        stats = hybrid_retriever.get_statistics()
        
        assert "bm25_weight" in stats
        assert "vector_weight" in stats
        assert "bm25_statistics" in stats
        
        assert stats["bm25_weight"] == 0.4
        assert stats["vector_weight"] == 0.6
        assert isinstance(stats["bm25_statistics"], dict)


class TestHybridRetrievalIntegration:
    """Integration tests for hybrid retrieval"""
    
    def test_end_to_end_search_flow(self, mock_bm25_store, mock_vector_store):
        """Test complete search flow"""
        retriever = HybridRetriever(
            bm25_store=mock_bm25_store,
            vector_store=mock_vector_store,
            bm25_weight=0.3,
            vector_weight=0.7
        )
        
        query = "productivity and time management"
        
        # Test regular search
        documents = retriever.search(query, k=3)
        assert len(documents) <= 3
        
        # Test search with scores
        scored_results = retriever.search_with_scores(query, k=2)
        assert len(scored_results) <= 2
        
        # Test RRF search
        rrf_documents = retriever.reciprocal_rank_fusion(query, k=2)
        assert len(rrf_documents) <= 2
        
        # Verify all methods return valid documents
        all_docs = documents + [doc for doc, _ in scored_results] + rrf_documents
        assert all(isinstance(doc, Document) for doc in all_docs)
    
    def test_score_consistency(self, hybrid_retriever):
        """Test that scores are consistent and properly normalized"""
        query = "test query for scoring"
        
        # Get results with scores
        results = hybrid_retriever.search_with_scores(query, k=5)
        
        # Verify scores are in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        
        # Verify all scores are non-negative
        assert all(score >= 0 for score in scores)
        
        # Verify metadata consistency
        for doc, combined_score in results:
            assert abs(doc.metadata["hybrid_score"] - combined_score) < 0.001
            assert doc.metadata["bm25_score"] >= 0
            assert doc.metadata["vector_score"] >= 0
    
    def test_different_query_types(self, hybrid_retriever):
        """Test hybrid search with different types of queries"""
        queries = [
            "productivity",  # Single word
            "time management techniques",  # Multi-word phrase
            "how to be more productive at work",  # Question format
            "SMART goals framework implementation",  # Technical terms
            "",  # Empty query (edge case)
        ]
        
        for query in queries:
            try:
                documents = hybrid_retriever.search(query, k=2)
                # Should handle all query types gracefully
                assert isinstance(documents, list)
                assert all(isinstance(doc, Document) for doc in documents)
            except Exception as e:
                pytest.fail(f"Failed to handle query '{query}': {e}")
    
    def test_large_k_values(self, hybrid_retriever):
        """Test hybrid search with large k values"""
        query = "test query"
        
        # Test with k larger than available documents
        large_k = 100
        documents = hybrid_retriever.search(query, k=large_k)
        
        # Should not fail and should return available documents
        assert len(documents) <= 4  # Mock has 4 documents max
        assert all(isinstance(doc, Document) for doc in documents)
    
    def test_weight_edge_cases(self, mock_bm25_store, mock_vector_store):
        """Test hybrid retrieval with edge case weights"""
        # Test with BM25 only (vector weight = 0)
        retriever1 = HybridRetriever(
            bm25_store=mock_bm25_store,
            vector_store=mock_vector_store,
            bm25_weight=1.0,
            vector_weight=0.0
        )
        
        docs1 = retriever1.search("test", k=2)
        assert len(docs1) >= 0
        
        # Test with vector only (BM25 weight = 0)
        retriever2 = HybridRetriever(
            bm25_store=mock_bm25_store,
            vector_store=mock_vector_store,
            bm25_weight=0.0,
            vector_weight=1.0
        )
        
        docs2 = retriever2.search("test", k=2)
        assert len(docs2) >= 0
        
        # Both should return valid results
        all_docs = docs1 + docs2
        assert all(isinstance(doc, Document) for doc in all_docs)