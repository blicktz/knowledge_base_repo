"""
Unit tests for HyDE (Hypothetical Document Embeddings) Retriever
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import json

from langchain.schema import Document
from langchain.llms.base import BaseLLM
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

from ...core.retrieval.hyde_retriever import HyDERetriever
from ...prompts.hyde_prompts import HYDE_PROMPTS


class MockLLM(BaseLLM):
    """Mock LLM for testing"""
    
    def __init__(self, response="This is a test response"):
        super().__init__()
        self.response = response
    
    def _call(self, prompt: str, **kwargs) -> str:
        return self.response
    
    def invoke(self, input: str) -> str:
        return self.response
    
    @property
    def _llm_type(self) -> str:
        return "mock"


class MockEmbeddings(Embeddings):
    """Mock embeddings for testing"""
    
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class MockVectorStore(VectorStore):
    """Mock vector store for testing"""
    
    def __init__(self):
        super().__init__()
        self.documents = [
            Document(page_content="Test document 1", metadata={"source": "test1"}),
            Document(page_content="Test document 2", metadata={"source": "test2"}),
            Document(page_content="Test document 3", metadata={"source": "test3"})
        ]
    
    def add_texts(self, texts, metadatas=None, **kwargs):
        return ["doc1", "doc2", "doc3"]
    
    def similarity_search(self, query, k=4):
        return self.documents[:k]
    
    def similarity_search_by_vector(self, embedding, k=4):
        return self.documents[:k]
    
    def similarity_search_with_score(self, query, k=4):
        return [(doc, 0.9) for doc in self.documents[:k]]
    
    def similarity_search_with_score_by_vector(self, embedding, k=4):
        return [(doc, 0.9) for doc in self.documents[:k]]


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_llm():
    """Create mock LLM"""
    return MockLLM("This is a comprehensive test hypothesis about productivity and frameworks.")


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings"""
    return MockEmbeddings()


@pytest.fixture
def mock_vector_store():
    """Create mock vector store"""
    return MockVectorStore()


@pytest.fixture
def hyde_retriever(mock_llm, mock_embeddings, mock_vector_store, temp_cache_dir):
    """Create HyDE retriever for testing"""
    return HyDERetriever(
        llm=mock_llm,
        embeddings=mock_embeddings,
        vector_store=mock_vector_store,
        cache_dir=temp_cache_dir
    )


class TestHyDERetriever:
    """Test cases for HyDE retriever"""
    
    def test_initialization(self, hyde_retriever, temp_cache_dir):
        """Test HyDE retriever initialization"""
        assert hyde_retriever.llm is not None
        assert hyde_retriever.embeddings is not None
        assert hyde_retriever.vector_store is not None
        assert hyde_retriever.cache_dir == Path(temp_cache_dir)
        assert hyde_retriever.cache_dir.exists()
    
    def test_query_hash_generation(self, hyde_retriever):
        """Test query hash generation"""
        query1 = "What are productivity frameworks?"
        query2 = "What are productivity frameworks?"
        query3 = "Different query"
        
        hash1 = hyde_retriever._get_query_hash(query1)
        hash2 = hyde_retriever._get_query_hash(query2)
        hash3 = hyde_retriever._get_query_hash(query3)
        
        assert hash1 == hash2  # Same query should have same hash
        assert hash1 != hash3  # Different query should have different hash
        assert len(hash1) == 32  # MD5 hash length
    
    def test_generate_hypothesis(self, hyde_retriever):
        """Test hypothesis generation"""
        query = "What are the best productivity frameworks?"
        
        hypothesis = hyde_retriever.generate_hypothesis(query)
        
        assert isinstance(hypothesis, str)
        assert len(hypothesis) > 0
        assert "test hypothesis" in hypothesis.lower()
        
        # Check that LLM interaction was logged
        log_files = list(hyde_retriever.cache_dir.glob("hyde_*.json"))
        assert len(log_files) == 1
        
        # Verify log content
        with open(log_files[0], 'r') as f:
            log_data = json.load(f)
        
        assert "timestamp" in log_data
        assert log_data["original_query"] == query
        assert "prompt" in log_data
        assert log_data["response"] == hypothesis
        assert log_data["component"] == "HyDE"
    
    def test_generate_hypothesis_with_custom_template(self, hyde_retriever):
        """Test hypothesis generation with custom prompt template"""
        query = "How to improve productivity?"
        custom_template = "Answer this question briefly: {query}\nAnswer:"
        
        hypothesis = hyde_retriever.generate_hypothesis(
            query,
            prompt_template=custom_template
        )
        
        assert isinstance(hypothesis, str)
        assert len(hypothesis) > 0
    
    def test_generate_hypothesis_with_error(self, mock_embeddings, mock_vector_store, temp_cache_dir):
        """Test hypothesis generation with LLM error"""
        # Create a mock LLM that raises an exception
        error_llm = Mock()
        error_llm.invoke.side_effect = Exception("LLM error")
        error_llm.__str__ = lambda x: "MockErrorLLM"
        
        retriever = HyDERetriever(
            llm=error_llm,
            embeddings=mock_embeddings,
            vector_store=mock_vector_store,
            cache_dir=temp_cache_dir
        )
        
        query = "Test query"
        hypothesis = retriever.generate_hypothesis(query)
        
        # Should fall back to original query
        assert hypothesis == query
        
        # Should log the error
        log_files = list(retriever.cache_dir.glob("hyde_*.json"))
        assert len(log_files) == 1
        
        with open(log_files[0], 'r') as f:
            log_data = json.load(f)
        
        assert "ERROR:" in log_data["response"]
    
    def test_retrieve_with_hypothesis(self, hyde_retriever):
        """Test retrieval using hypothesis"""
        query = "What are productivity frameworks?"
        k = 2
        
        documents = hyde_retriever.retrieve(query, k=k, use_hypothesis=True)
        
        assert len(documents) == k
        assert all(isinstance(doc, Document) for doc in documents)
        
        # Check metadata was added
        for doc in documents:
            assert doc.metadata["retrieval_method"] == "hyde"
            assert doc.metadata["original_query"] == query
            assert doc.metadata["hypothesis_used"] is True
    
    def test_retrieve_without_hypothesis(self, hyde_retriever):
        """Test retrieval without using hypothesis"""
        query = "What are productivity frameworks?"
        k = 2
        
        documents = hyde_retriever.retrieve(query, k=k, use_hypothesis=False)
        
        assert len(documents) == k
        assert all(isinstance(doc, Document) for doc in documents)
        
        # Check metadata
        for doc in documents:
            assert doc.metadata["retrieval_method"] == "direct"
            assert doc.metadata["original_query"] == query
            assert doc.metadata["hypothesis_used"] is False
    
    def test_retrieve_with_scores(self, hyde_retriever):
        """Test retrieval with similarity scores"""
        query = "Test query for scoring"
        k = 2
        
        results = hyde_retriever.retrieve_with_scores(query, k=k)
        
        assert len(results) == k
        assert all(isinstance(item, tuple) for item in results)
        assert all(len(item) == 2 for item in results)
        
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, (int, float))
            assert doc.metadata["similarity_score"] == score
    
    def test_cached_hypotheses_retrieval(self, hyde_retriever):
        """Test retrieval of cached hypotheses"""
        # Generate some hypotheses to create cache
        queries = ["Query 1", "Query 2", "Query 3"]
        
        for query in queries:
            hyde_retriever.generate_hypothesis(query)
        
        cached_interactions = hyde_retriever.get_cached_hypotheses()
        
        assert len(cached_interactions) == len(queries)
        
        for interaction in cached_interactions:
            assert "timestamp" in interaction
            assert "original_query" in interaction
            assert "response" in interaction
            assert "component" in interaction
            assert interaction["component"] == "HyDE"
    
    def test_llm_logging_with_metadata(self, hyde_retriever):
        """Test LLM interaction logging with metadata"""
        query = "Test query"
        metadata = {"test_key": "test_value", "experiment": "unit_test"}
        
        hypothesis = hyde_retriever.generate_hypothesis(
            query,
            log_metadata=metadata
        )
        
        # Check log file was created
        log_files = list(hyde_retriever.cache_dir.glob("hyde_*.json"))
        assert len(log_files) == 1
        
        with open(log_files[0], 'r') as f:
            log_data = json.load(f)
        
        assert log_data["metadata"]["test_key"] == "test_value"
        assert log_data["metadata"]["experiment"] == "unit_test"
    
    def test_different_prompt_templates(self, hyde_retriever):
        """Test using different prompt templates"""
        query = "How to be more productive?"
        
        # Test default template
        hypothesis1 = hyde_retriever.generate_hypothesis(query)
        
        # Test detailed explanation template
        template2 = HYDE_PROMPTS["detailed_explanation"]
        hypothesis2 = hyde_retriever.generate_hypothesis(query, prompt_template=template2)
        
        # Both should be valid hypotheses
        assert isinstance(hypothesis1, str)
        assert isinstance(hypothesis2, str)
        assert len(hypothesis1) > 0
        assert len(hypothesis2) > 0
        
        # Should have created 2 log files
        log_files = list(hyde_retriever.cache_dir.glob("hyde_*.json"))
        assert len(log_files) == 2
    
    def test_vector_store_integration(self, hyde_retriever):
        """Test integration with vector store methods"""
        query = "Test vector integration"
        
        # Test with similarity_search_by_vector method
        documents = hyde_retriever.retrieve(query, k=3)
        
        assert len(documents) <= 3
        assert all(isinstance(doc, Document) for doc in documents)
        
        # Test with scores
        results = hyde_retriever.retrieve_with_scores(query, k=2)
        assert len(results) <= 2
        
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, (int, float))


class TestHyDEIntegration:
    """Integration tests for HyDE retriever"""
    
    def test_end_to_end_retrieval(self, mock_llm, mock_embeddings, mock_vector_store, temp_cache_dir):
        """Test complete end-to-end retrieval flow"""
        retriever = HyDERetriever(
            llm=mock_llm,
            embeddings=mock_embeddings,
            vector_store=mock_vector_store,
            cache_dir=temp_cache_dir
        )
        
        query = "What are the best productivity techniques for entrepreneurs?"
        
        # Perform retrieval
        documents = retriever.retrieve(query, k=3)
        
        # Verify results
        assert len(documents) == 3
        assert all(isinstance(doc, Document) for doc in documents)
        
        # Verify hypothesis was generated and logged
        log_files = list(Path(temp_cache_dir).glob("hyde_*.json"))
        assert len(log_files) == 1
        
        # Verify metadata
        for doc in documents:
            assert "retrieval_method" in doc.metadata
            assert "original_query" in doc.metadata
            assert "hypothesis_used" in doc.metadata
    
    def test_cache_directory_creation(self, mock_llm, mock_embeddings, mock_vector_store):
        """Test automatic cache directory creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "non_existent" / "cache"
            
            retriever = HyDERetriever(
                llm=mock_llm,
                embeddings=mock_embeddings,
                vector_store=mock_vector_store,
                cache_dir=str(cache_path)
            )
            
            assert retriever.cache_dir.exists()
            assert retriever.cache_dir.is_dir()
    
    def test_multiple_queries_logging(self, hyde_retriever):
        """Test logging of multiple queries"""
        queries = [
            "What is productivity?",
            "How to manage time effectively?",
            "Best frameworks for goal setting?"
        ]
        
        for query in queries:
            hyde_retriever.generate_hypothesis(query)
        
        # Check all queries were logged
        log_files = list(hyde_retriever.cache_dir.glob("hyde_*.json"))
        assert len(log_files) == len(queries)
        
        # Verify each log file contains correct query
        logged_queries = []
        for log_file in log_files:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                logged_queries.append(log_data["original_query"])
        
        assert set(logged_queries) == set(queries)