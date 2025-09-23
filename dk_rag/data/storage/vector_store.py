"""
Vector store implementation using ChromaDB for document embeddings and retrieval.

This module provides a unified interface for storing and retrieving document chunks
with semantic search capabilities using sentence transformers.
"""

import os
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from ..models.persona_constitution import PersonaConstitution
from ...config.settings import Settings
from ...utils.logging import get_logger


class VectorStore:
    """
    Vector store for document embeddings using ChromaDB.
    
    Provides semantic search capabilities over document chunks using
    sentence transformer embeddings.
    """
    
    def __init__(self, settings: Settings, persona_id: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            settings: Application settings containing vector DB configuration
            persona_id: Unique identifier for the persona (for multi-tenant support)
        """
        self.settings = settings
        # Extract persona_id from settings if not provided directly
        if persona_id is None:
            persona_id = getattr(settings, 'persona_id', 'default')
        self.persona_id = persona_id
        self.logger = get_logger(__name__)
        
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Please install with: pip install chromadb")
        
        # Setup ChromaDB client
        self._setup_client()
        
        # Initialize embedding function
        self._setup_embedding_function()
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
    def _setup_client(self):
        """Setup ChromaDB client with persistence."""
        # Handle both dict and object configuration access
        config = self.settings.vector_db.config if hasattr(self.settings.vector_db, 'config') else self.settings.vector_db
        persist_directory = config.get('persist_directory', './data/storage/chroma_db') if isinstance(config, dict) else getattr(config, 'persist_directory', './data/storage/chroma_db')
        
        persist_dir = Path(persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Create ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.logger.info(f"ChromaDB client initialized with persistence at: {persist_dir}")
        
    def _setup_embedding_function(self):
        """Setup sentence transformer embedding function."""
        config = self.settings.vector_db.config if hasattr(self.settings.vector_db, 'config') else self.settings.vector_db
        model_name = config.get('embedding_model', 'sentence-transformers/all-mpnet-base-v2') if isinstance(config, dict) else getattr(config, 'embedding_model', 'sentence-transformers/all-mpnet-base-v2')
        
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
            self.logger.info(f"Embedding function initialized with model: {model_name}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize {model_name}, falling back to default: {e}")
            # Fallback to default embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
    
    def _get_or_create_collection(self):
        """Get or create the collection for this persona."""
        # Create persona-specific collection name
        config = self.settings.vector_db.config if hasattr(self.settings.vector_db, 'config') else self.settings.vector_db
        collection_base = config.get('collection_name', 'script_collection') if isinstance(config, dict) else getattr(config, 'collection_name', 'script_collection')
        collection_name = f"{collection_base}_{self.persona_id}"
        
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            self.logger.info(f"Retrieved existing collection: {collection_name}")
        except Exception:
            # Create new collection
            distance_metric = config.get('distance_metric', 'cosine') if isinstance(config, dict) else getattr(config, 'distance_metric', 'cosine')
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": distance_metric}
            )
            self.logger.info(f"Created new collection: {collection_name}")
        
        return collection
    
    def _log_configuration_info(self):
        """Log vector store configuration information."""
        try:
            config = self.settings.vector_db.config if hasattr(self.settings.vector_db, 'config') else self.settings.vector_db
            model_name = config.get('embedding_model', 'sentence-transformers/all-mpnet-base-v2') if isinstance(config, dict) else getattr(config, 'embedding_model', 'sentence-transformers/all-mpnet-base-v2')
            
            self.logger.info("=" * 60)
            self.logger.info("VECTOR STORE CONFIGURATION")
            self.logger.info("=" * 60)
            self.logger.info(f"Embedding Model: {model_name}")
            self.logger.info(f"Collection Name: {self.collection.name}")
            self.logger.info(f"Persona ID: {self.persona_id}")
            self.logger.info(f"Distance Metric: {config.get('distance_metric', 'cosine') if isinstance(config, dict) else getattr(config, 'distance_metric', 'cosine')}")
            
            # Get current collection stats
            current_stats = self.get_collection_stats()
            self.logger.info(f"Current chunks in collection: {current_stats.get('total_chunks', 0)}")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.warning(f"Failed to log configuration info: {e}")
    
    def add_documents(self, documents: Union[List[str], List[Dict[str, Any]]], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts or document dicts to add
            metadatas: List of metadata dictionaries for each document (optional if documents are dicts)
            ids: Optional list of IDs for documents (generated if not provided)
            
        Returns:
            List of document IDs that were added
        """
        if not documents:
            return []
        
        # Handle both string documents and dict documents
        if isinstance(documents[0], dict):
            # Extract content and metadata from document dicts
            doc_texts = [doc.get('content', '') for doc in documents]
            if metadatas is None:
                metadatas = [doc for doc in documents]  # Use full dict as metadata
        else:
            # Documents are already strings
            doc_texts = documents
            if metadatas is None:
                metadatas = [{'source': f'document_{i}'} for i in range(len(documents))]
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in doc_texts]
        
        # Ensure all lists have the same length
        if not (len(doc_texts) == len(metadatas) == len(ids)):
            raise ValueError("documents, metadatas, and ids must have the same length")
        
        try:
            # Log configuration info before indexing
            self._log_configuration_info()
            
            # Log the start of indexing
            self.logger.info(f"Starting to index {len(doc_texts)} documents/chunks in vector store...")
            
            # For large batches, add in smaller chunks to show progress
            batch_size = 50  # Process in batches of 50 for progress reporting
            total_batches = (len(doc_texts) + batch_size - 1) // batch_size
            
            if len(doc_texts) <= batch_size:
                # Small batch - add all at once
                self.collection.add(
                    documents=doc_texts,
                    metadatas=metadatas,
                    ids=ids
                )
                self.logger.info(f"Indexed {len(doc_texts)} documents in vector store")
            else:
                # Large batch - process in chunks with progress reporting
                for i in range(0, len(doc_texts), batch_size):
                    end_idx = min(i + batch_size, len(doc_texts))
                    batch_num = i // batch_size + 1
                    
                    self.collection.add(
                        documents=doc_texts[i:end_idx],
                        metadatas=metadatas[i:end_idx],
                        ids=ids[i:end_idx]
                    )
                    
                    self.logger.info(f"Vector indexing progress: batch {batch_num}/{total_batches} ({batch_num/total_batches*100:.1f}%) - {end_idx}/{len(doc_texts)} documents indexed")
                
                self.logger.info(f"Vector indexing complete: {len(doc_texts)} documents indexed in vector store")
            
            # Verify indexing was successful
            final_stats = self.get_collection_stats()
            self.logger.info(f"Post-indexing verification: Collection now contains {final_stats.get('total_chunks', 0)} total chunks")
            
            return ids
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            raise
    
    def search(self, query: str, n_results: int = 10, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            List of search results with documents, metadata, and scores
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'document': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'id': results['ids'][0][i] if results['ids'] and results['ids'][0] else None,
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else None
                    }
                    formatted_results.append(result)
            
            self.logger.debug(f"Search returned {len(formatted_results)} results for query: {query[:100]}...")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'document_count': count,  # Keep both for backward compatibility
                'collection_name': self.collection.name,
                'persona_id': self.persona_id
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {'total_chunks': 0, 'document_count': 0, 'collection_name': 'unknown', 'persona_id': self.persona_id}
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection.name)
            self.logger.info(f"Deleted collection: {self.collection.name}")
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            raise
    
    def update_document(self, doc_id: str, document: str, metadata: Dict[str, Any]):
        """Update a document in the collection."""
        try:
            self.collection.update(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata]
            )
            self.logger.debug(f"Updated document: {doc_id}")
        except Exception as e:
            self.logger.error(f"Failed to update document {doc_id}: {e}")
            raise
    
    def delete_document(self, doc_id: str):
        """Delete a document from the collection."""
        try:
            self.collection.delete(ids=[doc_id])
            self.logger.debug(f"Deleted document: {doc_id}")
        except Exception as e:
            self.logger.error(f"Failed to delete document {doc_id}: {e}")
            raise
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        try:
            results = self.collection.get(ids=[doc_id], include=['documents', 'metadatas'])
            
            if results['documents'] and results['documents'][0]:
                return {
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0] if results['metadatas'] else {},
                    'id': doc_id
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            # Get all document IDs
            all_docs = self.collection.get()
            if all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
                self.logger.info(f"Cleared {len(all_docs['ids'])} documents from collection")
            else:
                self.logger.info("Collection is already empty")
        except Exception as e:
            self.logger.error(f"Failed to clear collection: {e}")
            raise
    
    def close(self):
        """Close the vector store connection."""
        try:
            # ChromaDB doesn't need explicit closing, but we can log the closure
            self.logger.debug("Vector store connection closed")
        except Exception as e:
            self.logger.error(f"Error closing vector store: {e}")


class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    @staticmethod
    def create(settings: Settings, persona_id: Optional[str] = None) -> VectorStore:
        """
        Create a vector store instance based on configuration.
        
        Args:
            settings: Application settings
            persona_id: Unique identifier for the persona
            
        Returns:
            VectorStore instance
        """
        if settings.vector_db.provider == "chromadb":
            return VectorStore(settings, persona_id)
        else:
            raise ValueError(f"Unsupported vector store provider: {settings.vector_db.provider}")