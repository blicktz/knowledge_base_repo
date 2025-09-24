"""
Embedding Wrapper for bridging ChromaDB embeddings to LangChain interface

This module provides a wrapper to make ChromaDB's SentenceTransformerEmbeddingFunction
compatible with LangChain's embedding interface.
"""

from typing import List, Any
from langchain.embeddings.base import Embeddings


class ChromaEmbeddingWrapper(Embeddings):
    """
    Wrapper class to make ChromaDB embeddings compatible with LangChain interface.
    
    ChromaDB uses SentenceTransformerEmbeddingFunction which doesn't have
    embed_query() method, but LangChain's Embeddings interface expects it.
    """
    
    def __init__(self, chroma_embedding_function, model_name: str = None):
        """
        Initialize with ChromaDB embedding function.
        
        Args:
            chroma_embedding_function: ChromaDB SentenceTransformerEmbeddingFunction
            model_name: Name of the embedding model (for logging/debugging)
        """
        self.chroma_function = chroma_embedding_function
        self._model_name = model_name or getattr(chroma_embedding_function, '_model_name', 'Unknown')
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of float embeddings
        """
        # ChromaDB embedding functions expect a list and return a list of embeddings
        embeddings = self.chroma_function([text])
        
        # Return the first (and only) embedding
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        else:
            raise ValueError("Failed to generate embedding for query")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if not texts:
            return []
            
        return self.chroma_function(texts)
    
    @property
    def model_name(self) -> str:
        """Get the embedding model name."""
        return self._model_name
    
    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return f"ChromaEmbeddingWrapper(model={self._model_name})"