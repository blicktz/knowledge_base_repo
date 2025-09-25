"""
Mental Models Knowledge Store

Dedicated ChromaDB store for mental models knowledge with completely isolated
database instance to avoid readonly database conflicts.
"""

import os
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime

from langchain_chroma import Chroma
from langchain.schema import Document

from ..models.persona_constitution import PersonaConstitution
from ...config.settings import Settings
from ...models.knowledge_types import KnowledgeType
from ...models.knowledge_results import IndexingResult
from ...utils.logging import get_logger
from ...core.retrieval.embedding_wrapper import ChromaEmbeddingWrapper
from ...utils.model_manager import get_model_manager


class MentalModelsStore:
    """
    Dedicated vector store for mental models knowledge.
    
    Uses completely isolated ChromaDB instance to prevent database locking
    conflicts with other knowledge types.
    """
    
    def __init__(
        self,
        settings: Settings,
        persona_id: str,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        """
        Initialize mental models vector store.
        
        Args:
            settings: Application settings
            persona_id: Unique persona identifier
            embedding_model: Name of the embedding model to use
        """
        self.settings = settings
        self.persona_id = persona_id
        self.embedding_model = embedding_model
        self.logger = get_logger(__name__)
        
        if not persona_id:
            raise ValueError("persona_id is required for multi-tenant isolation")
        
        # Initialize embedding function
        self._setup_embeddings()
        
        # Setup database path - completely isolated
        self.db_path = self._get_db_path()
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB collection
        self._initialize_collection()
        
        # Store collection stats
        self.collection_stats = {
            'created_timestamp': datetime.now().isoformat(),
            'collection_name': self.collection_name,
            'collection_path': str(self.db_path),
            'documents_indexed': 0,
            'last_indexed': None
        }
        
        self.logger.info(f"Mental models store initialized for persona: {persona_id}")
    
    def _setup_embeddings(self):
        """Setup embedding function for the collection using ModelManager."""
        try:
            model_manager = get_model_manager()
            
            # Get ChromaDB embedding function from ModelManager (lazy loading)
            chroma_embedding_function = model_manager.get_chroma_embedding_function(
                self.embedding_model
            )
            
            if chroma_embedding_function is None:
                raise RuntimeError(f"Failed to load ChromaDB embedding function for {self.embedding_model}")
            
            # Wrap for LangChain compatibility
            self.embedding_function = ChromaEmbeddingWrapper(
                chroma_embedding_function,
                self.embedding_model
            )
            
            self.logger.info(f"Initialized embedding function with model: {self.embedding_model} (using ModelManager)")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding function: {e}")
            raise
    
    def _get_db_path(self) -> Path:
        """
        Get isolated database directory for mental models.
        
        Returns:
            Path to isolated database directory
        """
        # Get base persona directory (parent of vector_db)
        base_persona_path = Path(self.settings.get_vector_db_path(self.persona_id)).parent
        return base_persona_path / "vector_db_mental_models"
    
    def _initialize_collection(self):
        """Initialize ChromaDB collection with isolated database."""
        self.collection_name = f"{self.persona_id}_mental_models"
        
        try:
            self.collection = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=str(self.db_path)  # Completely isolated path
            )
            
            self.logger.info(
                f"Created collection for Mental Models: "
                f"{self.collection_name} at {self.db_path}"
            )
            
        except Exception as e:
            error_msg = f"Failed to create mental models collection: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def index_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        rebuild: bool = False
    ) -> IndexingResult:
        """
        Index mental models documents.
        
        Args:
            documents: List of documents to index
            batch_size: Number of documents to process in each batch
            rebuild: Whether to rebuild the collection from scratch
            
        Returns:
            IndexingResult with indexing statistics and any errors
        """
        start_time = datetime.now()
        
        result = IndexingResult(
            knowledge_type=KnowledgeType.MENTAL_MODELS,
            persona_id=self.persona_id,
            documents_processed=len(documents)
        )
        
        if not documents:
            result.add_warning("No documents provided for indexing")
            return result
        
        try:
            self.logger.info(f"Indexing {len(documents)} mental models documents")
            
            # Clear existing documents if rebuilding
            if rebuild:
                self.logger.info("Rebuilding mental models collection")
                # TODO: Fix readonly database issue when clearing collection
                # self._clear_collection()  # DISABLED: Causes "readonly database" SQLite error
                self._initialize_collection()  # Recreate after clear
            
            # Index documents in batches
            total_indexed = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                try:
                    # Add documents to collection
                    self.collection.add_documents(batch)
                    total_indexed += len(batch)
                    
                    self.logger.debug(
                        f"Indexed batch {i//batch_size + 1}: "
                        f"{len(batch)} documents ({total_indexed}/{len(documents)} total)"
                    )
                    
                except Exception as e:
                    error_msg = f"Failed to index batch {i//batch_size + 1}: {e}"
                    result.add_error(error_msg)
                    continue
            
            # Update results
            result.documents_indexed = total_indexed
            result.vector_store_created = True
            result.indexing_duration_seconds = (datetime.now() - start_time).total_seconds()
            
            # Update collection stats
            self.collection_stats.update({
                'documents_indexed': total_indexed,
                'last_indexed': datetime.now().isoformat(),
                'last_batch_size': batch_size
            })
            
            # Calculate index size
            if self.db_path.exists():
                size_bytes = sum(
                    f.stat().st_size 
                    for f in self.db_path.rglob('*') 
                    if f.is_file()
                )
                result.index_size_mb = size_bytes / (1024 * 1024)
            
            if result.success:
                self.logger.info(
                    f"Successfully indexed {total_indexed} mental models documents "
                    f"in {result.indexing_duration_seconds:.2f}s"
                )
            else:
                self.logger.warning(
                    f"Partially indexed {total_indexed}/{len(documents)} "
                    f"mental models documents with {len(result.errors)} errors"
                )
            
        except Exception as e:
            error_msg = f"Critical error indexing mental models: {e}"
            result.add_error(error_msg)
            self.logger.error(error_msg)
        
        return result
    
    def search(
        self,
        query: str,
        k: int = 5,
        return_scores: bool = False,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        Search mental models documents.
        
        Args:
            query: Search query
            k: Number of results to return
            return_scores: Whether to return similarity scores
            filter_metadata: Optional metadata filters
            
        Returns:
            List of documents or (document, score) tuples
        """
        try:
            self.logger.debug(f"Searching mental models with query: {query[:100]}...")
            
            if return_scores:
                results = self.collection.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
            else:
                results = self.collection.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
            
            self.logger.debug(f"Found {len(results)} mental models results")
            return results
            
        except Exception as e:
            self.logger.error(f"Mental models search failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the mental models collection.
        
        Returns:
            Dictionary with collection statistics
        """
        stats = {
            'knowledge_type': 'mental_models',
            'display_name': 'Mental Models',
            'collection_exists': True,
            'persona_id': self.persona_id
        }
        
        # Add stored stats
        stats.update(self.collection_stats)
        
        # Add current document count
        try:
            if hasattr(self.collection, '_collection'):
                stats['current_document_count'] = self.collection._collection.count()
            else:
                stats['current_document_count'] = 'unknown'
        except Exception as e:
            stats['current_document_count'] = f'error: {e}'
        
        # Add directory size
        if self.db_path.exists():
            try:
                size_bytes = sum(
                    f.stat().st_size 
                    for f in self.db_path.rglob('*') 
                    if f.is_file()
                )
                stats['directory_size_mb'] = round(size_bytes / (1024 * 1024), 2)
            except Exception:
                stats['directory_size_mb'] = 'unknown'
        else:
            stats['directory_size_mb'] = 0.0
        
        return stats
    
    def collection_exists(self) -> bool:
        """
        Check if the mental models collection exists.
        
        Returns:
            True if collection exists, False otherwise
        """
        return self.db_path.exists() and any(self.db_path.iterdir())
    
    def _clear_collection(self):
        """Clear all documents from the collection."""
        try:
            if self.db_path.exists():
                # Remove all files and subdirectories, but keep the main directory
                import shutil
                for item in self.db_path.iterdir():
                    if item.is_file():
                        item.unlink()  # Remove file
                        self.logger.debug(f"Removed file: {item}")
                    elif item.is_dir():
                        shutil.rmtree(item)  # Remove subdirectory
                        self.logger.debug(f"Removed directory: {item}")
                
                self.logger.info(f"Cleared mental models collection files in: {self.db_path}")
            
            # Reset stats
            self.collection_stats.update({
                'documents_indexed': 0,
                'last_indexed': None
            })
                
        except Exception as e:
            self.logger.error(f"Failed to clear mental models collection: {e}")
            raise
    
    def delete_collection(self):
        """Permanently delete the mental models collection."""
        self.logger.warning(
            "Deleting mental models collection - this cannot be undone"
        )
        self._clear_collection()
    
    def cleanup(self):
        """Clean up resources and close connections."""
        self.logger.info("Cleaning up mental models store resources")
        
        try:
            if hasattr(self.collection, 'persist'):
                self.collection.persist()
            if hasattr(self.collection, '_client') and hasattr(self.collection._client, 'reset'):
                # Only reset if it's safe to do so
                pass  # ChromaDB reset is often disabled by config
            self.logger.debug("Mental models store cleanup complete")
        except Exception as e:
            self.logger.warning(f"Mental models store cleanup warning: {e}")