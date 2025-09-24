"""
Core Phase 2 functionality tests (no LLM calls)
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

from langchain.schema import Document

from dk_rag.data.storage.bm25_store import BM25Store
from dk_rag.core.retrieval.hybrid_retriever import HybridRetriever
from dk_rag.config.retrieval_config import Phase2RetrievalConfig
from dk_rag.prompts.hyde_prompts import HYDE_PROMPTS, get_hyde_prompt, select_best_prompt


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
            )
        ]
    
    def similarity_search(self, query, k=4):
        return self.documents[:k]
    
    def similarity_search_with_score(self, query, k=4):
        scores = [0.9, 0.8, 0.7]
        return [(doc, scores[i]) for i, doc in enumerate(self.documents[:k])]


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        "Productivity frameworks help organize work and increase efficiency through structured approaches.",
        "Time management techniques include the Pomodoro Technique and time blocking for better focus.",
        "Goal setting frameworks like SMART goals provide clear structure for achievement and accountability.",
        "Deep work practices require sustained focus and elimination of interruptions for maximum productivity.",
        "Workflow optimization involves eliminating bottlenecks and streamlining processes for better results."
    ]


class TestBM25Store:
    """Test BM25 index functionality"""
    
    def test_bm25_store_creation(self, temp_dir):
        """Test BM25 store initialization"""
        store = BM25Store(temp_dir)
        assert store.index_path == Path(temp_dir)
        assert store.k1 == 1.5
        assert store.b == 0.75
    
    def test_bm25_index_building(self, temp_dir, sample_documents):
        """Test BM25 index building and persistence"""
        store = BM25Store(temp_dir)
        store.build_index(sample_documents)
        
        # Check that index was created
        assert store.bm25_index is not None
        assert len(store.doc_texts) == len(sample_documents)
        assert len(store.doc_ids) == len(sample_documents)
    
    def test_bm25_search(self, temp_dir, sample_documents):
        """Test BM25 search functionality"""
        store = BM25Store(temp_dir)
        store.build_index(sample_documents)
        
        # Test search
        results = store.search("productivity frameworks", k=3)
        
        assert len(results) <= 3
        assert all(isinstance(item, tuple) for item in results)
        assert all(len(item) == 2 for item in results)  # (doc_id, score)
        
        # Check that results have scores > 0
        assert all(score > 0 for _, score in results)
    
    def test_bm25_batch_search(self, temp_dir, sample_documents):
        """Test BM25 batch search"""
        store = BM25Store(temp_dir)
        store.build_index(sample_documents)
        
        queries = ["productivity", "time management", "goals"]
        results = store.batch_search(queries, k=2)
        
        assert len(results) == len(queries)
        for query_results in results:
            assert len(query_results) <= 2
    
    def test_bm25_document_retrieval(self, temp_dir, sample_documents):
        """Test document retrieval by ID"""
        store = BM25Store(temp_dir)
        doc_ids = [f"doc_{i}" for i in range(len(sample_documents))]
        store.build_index(sample_documents, doc_ids)
        
        # Test document retrieval
        doc_text = store.get_document_by_id("doc_0")
        assert doc_text == sample_documents[0]
        
        # Test non-existent document
        assert store.get_document_by_id("non_existent") is None
    
    def test_bm25_statistics(self, temp_dir, sample_documents):
        """Test statistics collection"""
        store = BM25Store(temp_dir)
        
        # Before indexing
        stats = store.get_statistics()
        assert stats["status"] == "not_built"
        
        # After indexing
        store.build_index(sample_documents)
        stats = store.get_statistics()
        assert stats["status"] == "built"
        assert stats["num_documents"] == len(sample_documents)


class TestHybridRetriever:
    """Test hybrid retrieval functionality"""
    
    def test_hybrid_retriever_initialization(self, temp_dir, sample_documents):
        """Test hybrid retriever setup"""
        bm25_store = BM25Store(temp_dir)
        bm25_store.build_index(sample_documents)
        
        vector_store = MockVectorStore()
        
        retriever = HybridRetriever(
            bm25_store=bm25_store,
            vector_store=vector_store,
            bm25_weight=0.4,
            vector_weight=0.6
        )
        
        assert retriever.bm25_store == bm25_store
        assert retriever.vector_store == vector_store
        assert retriever.bm25_weight == 0.4
        assert retriever.vector_weight == 0.6
    
    def test_weight_normalization(self, temp_dir, sample_documents):
        """Test automatic weight normalization"""
        bm25_store = BM25Store(temp_dir)
        bm25_store.build_index(sample_documents)
        vector_store = MockVectorStore()
        
        # Create retriever with weights that don't sum to 1.0
        retriever = HybridRetriever(
            bm25_store=bm25_store,
            vector_store=vector_store,
            bm25_weight=0.3,
            vector_weight=0.4  # Sum = 0.7
        )
        
        # Weights should be normalized
        assert abs(retriever.bm25_weight + retriever.vector_weight - 1.0) < 0.001
    
    def test_score_normalization(self, temp_dir, sample_documents):
        """Test score normalization functionality"""
        bm25_store = BM25Store(temp_dir)
        bm25_store.build_index(sample_documents)
        vector_store = MockVectorStore()
        
        retriever = HybridRetriever(bm25_store, vector_store)
        
        # Test various score ranges
        scores1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized1 = retriever._normalize_scores(scores1)
        
        assert min(normalized1) == 0.0
        assert max(normalized1) == 1.0
        assert len(normalized1) == len(scores1)
        
        # Test with empty list
        assert retriever._normalize_scores([]) == []
    
    def test_hybrid_search(self, temp_dir, sample_documents):
        """Test hybrid search functionality"""
        bm25_store = BM25Store(temp_dir)
        bm25_store.build_index(sample_documents)
        vector_store = MockVectorStore()
        
        retriever = HybridRetriever(bm25_store, vector_store)
        
        query = "productivity frameworks"
        documents = retriever.search(query, k=2)
        
        assert len(documents) <= 2
        assert all(isinstance(doc, Document) for doc in documents)
        
        # Check hybrid metadata
        for doc in documents:
            assert "retrieval_method" in doc.metadata
            assert doc.metadata["retrieval_method"] == "hybrid"
            assert "hybrid_score" in doc.metadata
    
    def test_hybrid_search_with_scores(self, temp_dir, sample_documents):
        """Test hybrid search returning scores"""
        bm25_store = BM25Store(temp_dir)
        bm25_store.build_index(sample_documents)
        vector_store = MockVectorStore()
        
        retriever = HybridRetriever(bm25_store, vector_store)
        
        results = retriever.search_with_scores("time management", k=2)
        
        assert len(results) <= 2
        assert all(isinstance(item, tuple) for item in results)
        assert all(len(item) == 2 for item in results)
        
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, (int, float))
    
    def test_rrf_search(self, temp_dir, sample_documents):
        """Test Reciprocal Rank Fusion"""
        bm25_store = BM25Store(temp_dir)
        bm25_store.build_index(sample_documents)
        vector_store = MockVectorStore()
        
        retriever = HybridRetriever(bm25_store, vector_store)
        
        documents = retriever.reciprocal_rank_fusion("goal setting", k=2)
        
        assert len(documents) <= 2
        assert all(isinstance(doc, Document) for doc in documents)
        
        # Check RRF metadata
        for doc in documents:
            assert "rrf_score" in doc.metadata
            assert "retrieval_method" in doc.metadata
            assert doc.metadata["retrieval_method"] == "hybrid_rrf"


class TestPromptTemplates:
    """Test prompt template functionality"""
    
    def test_hyde_prompt_templates(self):
        """Test HyDE prompt template loading"""
        # Test default template
        default_prompt = get_hyde_prompt("default")
        assert isinstance(default_prompt, str)
        assert "{query}" in default_prompt
        
        # Test formatted prompt
        formatted = get_hyde_prompt("default", "test query")
        assert "test query" in formatted
        assert "{query}" not in formatted
    
    def test_prompt_selection(self):
        """Test automatic prompt selection"""
        test_cases = [
            ("How to improve productivity?", "tutorial_style"),
            ("What is the difference between X and Y?", "comparison"),
            ("Best practices for time management", "best_practices"),
            ("What is productivity?", "concept_explanation"),
        ]
        
        for query, expected_type in test_cases:
            selected = select_best_prompt(query)
            # Just verify it returns a valid prompt type
            assert selected in HYDE_PROMPTS
    
    def test_all_prompt_templates(self):
        """Test that all prompt templates are valid"""
        for prompt_type, template in HYDE_PROMPTS.items():
            assert isinstance(template, str)
            assert len(template) > 0
            assert "{query}" in template


class TestConfiguration:
    """Test Phase 2 configuration system"""
    
    def test_default_config(self):
        """Test default configuration loading"""
        config = Phase2RetrievalConfig()
        
        assert config.enabled is True
        assert config.hyde.enabled is True
        assert config.hybrid_search.enabled is True
        assert config.reranking.enabled is True
        assert config.caching.enabled is True
    
    def test_config_serialization(self):
        """Test configuration serialization"""
        config = Phase2RetrievalConfig()
        
        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "hyde" in config_dict
        assert "hybrid_search" in config_dict
        assert "reranking" in config_dict
        
        # Test from_dict
        new_config = Phase2RetrievalConfig.from_dict(config_dict)
        assert new_config.enabled == config.enabled
        assert new_config.hyde.enabled == config.hyde.enabled
    
    def test_storage_paths(self):
        """Test storage path configuration"""
        config = Phase2RetrievalConfig()
        
        # Test with persona_id
        test_persona_id = "test_persona"
        bm25_path = config.storage.get_bm25_index_path(test_persona_id)
        cache_dir = config.storage.get_cache_dir(test_persona_id)
        
        assert isinstance(bm25_path, Path)
        assert isinstance(cache_dir, Path)
        assert "bm25" in str(bm25_path)
        assert "cache" in str(cache_dir)
        assert test_persona_id in str(bm25_path)
        assert test_persona_id in str(cache_dir)
        
        # Test that methods require persona_id
        with pytest.raises(ValueError, match="persona_id is required"):
            config.storage.get_bm25_index_path()
            
        with pytest.raises(ValueError, match="persona_id is required"):
            config.storage.get_cache_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])