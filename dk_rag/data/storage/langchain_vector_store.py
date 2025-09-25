"""
LangChain-Compatible Vector Store using ChromaDB

This module provides a wrapper around LangChain's Chroma vector store
to maintain our existing API while providing full LangChain compatibility.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_chroma import Chroma
from langchain.schema import Document

from ..models.persona_constitution import PersonaConstitution
from ...config.settings import Settings
from ...utils.logging import get_logger, get_component_logger
from ...core.retrieval.embedding_wrapper import ChromaEmbeddingWrapper
from ...utils.model_manager import get_model_manager


class LangChainVectorStore:
    """
    LangChain-compatible vector store wrapper for ChromaDB.
    
    Provides the same API as our custom VectorStore but uses
    LangChain's Chroma implementation for full compatibility.
    """
    
    def __init__(self, settings: Settings, persona_id: Optional[str] = None):
        """
        Initialize the LangChain vector store.
        
        Args:
            settings: Application settings containing vector DB configuration
            persona_id: Unique identifier for the persona (for multi-tenant support)
        """
        self.settings = settings
        # Require persona_id - no fallback to maintain multi-tenant isolation
        if persona_id is None:
            raise ValueError("persona_id is required - multi-tenant isolation requires explicit persona identification")
        self.persona_id = persona_id
        self.logger = get_component_logger("LCVecStore", persona_id)
        
        # Setup ChromaDB configuration
        self._setup_chroma()
        
        self.logger.info(f"LangChain vector store initialized for persona: {persona_id}")
    
    def _setup_chroma(self):
        """Setup ChromaDB with LangChain's Chroma wrapper."""
        config = self.settings.vector_db.config if hasattr(self.settings.vector_db, 'config') else self.settings.vector_db
        
        # Get configuration values
        if isinstance(config, dict):
            collection_base = config.get('collection_name', 'script_collection')
            embedding_model = config.get('embedding_model', 'sentence-transformers/all-mpnet-base-v2')
            persist_directory = config.get('persist_directory', './data/storage/chroma_db')
        else:
            collection_base = getattr(config, 'collection_name', 'script_collection')
            embedding_model = getattr(config, 'embedding_model', 'sentence-transformers/all-mpnet-base-v2')
            persist_directory = getattr(config, 'persist_directory', './data/storage/chroma_db')
        
        # Use persona-specific collection name
        collection_name = f"{self.persona_id}_documents"
        self.logger.info(f"Using persona-specific collection: {collection_name}")
        
        # Setup persona-specific persist directory
        persona_vector_db_path = self.settings.get_vector_db_path(self.persona_id)
        persist_dir = Path(persona_vector_db_path)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Create embedding function using ModelManager
        try:
            model_manager = get_model_manager()
            
            # Get ChromaDB embedding function from ModelManager (lazy loading)
            chroma_embedding_function = model_manager.get_chroma_embedding_function(
                embedding_model
            )
            
            if chroma_embedding_function is None:
                raise RuntimeError(f"Failed to load ChromaDB embedding function for {embedding_model}")
            
            # Wrap it for LangChain compatibility
            embedding_function = ChromaEmbeddingWrapper(chroma_embedding_function, embedding_model)
            
            self.logger.info(f"Initialized embedding function with model: {embedding_model} (using ModelManager)")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding function: {e}")
            raise
        
        # Initialize LangChain's Chroma
        try:
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=str(persist_dir)
            )
            
            self.logger.info(f"LangChain Chroma initialized - Collection: {collection_name}, Dir: {persist_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain Chroma: {e}")
            raise
        
        # Store the raw embedding function for compatibility
        self.embedding_function = chroma_embedding_function
    
    def add_documents(self, documents: List[Dict[str, Any]], show_progress: bool = True):
        """
        Add documents to the vector store with batch processing to avoid ChromaDB limits.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            show_progress: Whether to show progress updates
        """
        if not documents:
            self.logger.warning("No documents to add")
            return
        
        try:
            # Log configuration info before indexing
            self._log_configuration_info()
            
            # Convert our document format to LangChain Documents
            langchain_docs = []
            for doc in documents:
                content = doc.get('content', doc.get('document', ''))
                metadata = doc.get('metadata', {})
                
                langchain_docs.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
            
            # Log the start of indexing
            self.logger.info(f"Starting to index {len(langchain_docs)} documents/chunks in vector store...")
            
            # Batch processing to avoid ChromaDB batch size limits
            batch_size = 50  # Stay well under ChromaDB batch limits (max ~5461)
            total_batches = (len(langchain_docs) + batch_size - 1) // batch_size
            
            if len(langchain_docs) <= batch_size:
                # Small batch - add all at once
                self.vector_store.add_documents(langchain_docs)
                self.logger.info(f"Indexed {len(langchain_docs)} documents in vector store")
            else:
                # Large batch - process in chunks with progress reporting
                for i in range(0, len(langchain_docs), batch_size):
                    end_idx = min(i + batch_size, len(langchain_docs))
                    batch_num = i // batch_size + 1
                    
                    batch = langchain_docs[i:end_idx]
                    self.vector_store.add_documents(batch)
                    
                    # Progress logging
                    if show_progress:
                        self.logger.info(f"Batch {batch_num}/{total_batches}: Indexed {len(batch)} documents "
                                       f"({end_idx}/{len(langchain_docs)} total)")
            
            # Final summary
            final_stats = self.get_collection_stats()
            self.logger.info(f"Vector store indexing complete. Total documents: {final_stats.get('total_chunks', 0)}")
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search using text query.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_by_vector(self, embedding: List[float], k: int = 4) -> List[Document]:
        """
        Perform similarity search using embedding vector.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        return self.vector_store.similarity_search_by_vector(embedding, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores using text query.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def similarity_search_with_score_by_vector(self, embedding: List[float], k: int = 4) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores using embedding vector.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        return self.vector_store.similarity_search_by_vector_with_relevance_scores(embedding, k=k)
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents from the collection for BM25 indexing.
        
        Returns:
            List of document dictionaries
        """
        try:
            # Access the underlying ChromaDB collection
            collection = self.vector_store._collection
            results = collection.get(include=['documents', 'metadatas'])
            
            documents = []
            for i, doc in enumerate(results['documents']):
                documents.append({
                    'document': doc,
                    'metadata': results['metadatas'][i] if results['metadatas'] else {},
                    'id': results['ids'][i]
                })
            
            self.logger.debug(f"Retrieved {len(documents)} documents for BM25 indexing")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to get all documents: {e}")
            return []
    
    def delete_document(self, doc_id: str):
        """Delete a document from the collection."""
        try:
            self.vector_store.delete([doc_id])
            self.logger.debug(f"Deleted document: {doc_id}")
        except Exception as e:
            self.logger.error(f"Failed to delete document {doc_id}: {e}")
            raise
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            # Access the underlying ChromaDB collection
            collection = self.vector_store._collection
            
            # ChromaDB 1.1.0+ requires at least one parameter - get all IDs first
            all_results = collection.get()
            if all_results['ids']:
                collection.delete(ids=all_results['ids'])
                self.logger.info(f"Cleared {len(all_results['ids'])} documents from collection")
            else:
                self.logger.info("Collection is already empty")
                
        except Exception as e:
            self.logger.error(f"Failed to clear collection: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            # Access the underlying ChromaDB collection
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                'document_count': count,
                'collection_name': collection.name,
                'persona_id': self.persona_id
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {'document_count': 0, 'persona_id': self.persona_id}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection (backward compatibility)."""
        try:
            # Access the underlying ChromaDB collection
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                'total_chunks': count,
                'document_count': count,  # Keep both for backward compatibility
                'collection_name': collection.name,
                'persona_id': self.persona_id
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {
                'total_chunks': 0, 
                'document_count': 0, 
                'collection_name': 'unknown', 
                'persona_id': self.persona_id
            }
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            # Access the underlying ChromaDB collection
            collection = self.vector_store._collection
            collection_name = collection.name
            
            # Get the client and delete collection
            client = collection._client
            client.delete_collection(name=collection_name)
            
            self.logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            raise
    
    def close(self):
        """Close the vector store connection."""
        try:
            # ChromaDB doesn't need explicit closing, but we can log the closure
            self.logger.debug("Vector store connection closed")
        except Exception as e:
            self.logger.error(f"Error closing vector store: {e}")
    
    def _log_configuration_info(self):
        """Log configuration info before indexing (for compatibility)."""
        try:
            self.logger.info("=" * 60)
            self.logger.info("Vector Store Configuration:")
            self.logger.info(f"  Collection: {self.vector_store._collection.name}")
            self.logger.info(f"  Persona ID: {self.persona_id}")
            self.logger.info(f"  Embedding Model: {getattr(self.vector_store.embeddings, 'model_name', 'Unknown')}")
            
            # Get current stats
            current_stats = self.get_collection_stats()
            self.logger.info(f"Current chunks in collection: {current_stats.get('total_chunks', 0)}")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.warning(f"Failed to log configuration info: {e}")


# Alias for backward compatibility
VectorStore = LangChainVectorStore