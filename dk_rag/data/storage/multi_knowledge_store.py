"""
Transcript Knowledge Store

Manages ChromaDB collections for transcript-based knowledge types while
providing a unified interface for indexing and retrieval operations.
Each knowledge type gets its own collection with isolated storage.
"""

import os
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime

from langchain_chroma import Chroma
from langchain.schema import Document

from ..models.persona_constitution import PersonaConstitution
from ...config.settings import Settings
from ...models.knowledge_types import KnowledgeType, validate_knowledge_type
from ...models.knowledge_results import IndexingResult
from ...utils.logging import get_logger, get_component_logger
from ...core.retrieval.embedding_wrapper import ChromaEmbeddingWrapper


class TranscriptKnowledgeStore:
    """
    Transcript-based knowledge vector store managing separate ChromaDB collections.
    
    Provides isolated storage and retrieval for different knowledge types
    while maintaining a unified API for indexing operations.
    """
    
    def __init__(
        self,
        settings: Settings,
        persona_id: str,
        embedding_model: str = None,
        language: str = "en"
    ):
        """
        Initialize multi-knowledge vector store.

        Args:
            settings: Application settings
            persona_id: Unique persona identifier
            embedding_model: Name of the embedding model to use (if None, uses language-appropriate model)
            language: Content language for selecting appropriate embedding model
        """
        self.settings = settings
        self.persona_id = persona_id
        self.language = language.strip() if language else "en"
        self.logger = get_component_logger("TKStore", persona_id)

        # Get language-appropriate embedding model if not specified
        if embedding_model is None:
            from ...utils.model_manager import get_model_manager
            model_manager = get_model_manager()
            embedding_model = model_manager.get_embedding_model_for_language(self.language)

        self.embedding_model = embedding_model
        
        if not persona_id:
            raise ValueError("persona_id is required for multi-tenant isolation")
        
        # Initialize embedding function
        self._setup_embeddings()
        
        # Store for initialized collections
        self._collections: Dict[KnowledgeType, Chroma] = {}
        self._collection_stats: Dict[KnowledgeType, Dict[str, Any]] = {}
        
        # Base directory for this persona's vector stores
        self.base_vector_path = Path(settings.get_vector_db_path(persona_id))
        self.base_vector_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Multi-knowledge store initialized for persona: {persona_id}")
    
    def _setup_embeddings(self):
        """Setup embedding function for all collections."""
        try:
            from chromadb.utils import embedding_functions
            
            chroma_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            
            # Wrap for LangChain compatibility
            self.embedding_function = ChromaEmbeddingWrapper(
                chroma_embedding_function,
                self.embedding_model
            )
            
            self.logger.info(f"Initialized embedding function with model: {self.embedding_model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding function: {e}")
            raise
    
    def get_knowledge_db_path(self, knowledge_type: KnowledgeType) -> Path:
        """
        Get isolated database directory for knowledge type.
        
        Each knowledge type gets its own completely separate database directory
        to avoid SQLite locking conflicts between different knowledge systems.
        
        Args:
            knowledge_type: Type of knowledge
            
        Returns:
            Path to isolated database directory for this knowledge type
        """
        knowledge_type = validate_knowledge_type(knowledge_type)
        
        # Get base persona directory (parent of vector_db)
        base_persona_path = self.base_vector_path.parent
        
        if knowledge_type == KnowledgeType.TRANSCRIPTS:
            # Existing transcript database location (unchanged)
            return self.base_vector_path
        elif knowledge_type == KnowledgeType.MENTAL_MODELS:
            # Separate database for mental models
            return base_persona_path / "vector_db_mental_models"
        elif knowledge_type == KnowledgeType.CORE_BELIEFS:
            # Separate database for core beliefs
            return base_persona_path / "vector_db_core_beliefs"
        else:
            raise ValueError(f"Unknown knowledge type: {knowledge_type}")
    
    def get_collection(self, knowledge_type: KnowledgeType) -> Chroma:
        """
        Get or create ChromaDB collection for a knowledge type.
        
        Args:
            knowledge_type: Type of knowledge for the collection
            
        Returns:
            Chroma collection instance
        """
        knowledge_type = validate_knowledge_type(knowledge_type)
        
        # Return existing collection if already initialized
        if knowledge_type in self._collections:
            return self._collections[knowledge_type]
        
        # Create new collection with isolated database directory
        collection_name = f"{self.persona_id}{knowledge_type.collection_suffix}"
        
        # Use completely separate database directory for each knowledge type
        db_path = self.get_knowledge_db_path(knowledge_type)
        db_path.mkdir(parents=True, exist_ok=True)
        
        try:
            collection = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_function,
                persist_directory=str(db_path)  # Isolated database directory
            )
            
            self._collections[knowledge_type] = collection
            
            # Initialize stats
            self._collection_stats[knowledge_type] = {
                'created_timestamp': datetime.now().isoformat(),
                'collection_name': collection_name,
                'collection_path': str(db_path),
                'documents_indexed': 0,
                'last_indexed': None
            }
            
            self.logger.info(
                f"Created collection for {knowledge_type.display_name}: "
                f"{collection_name} at {db_path}"
            )
            
            return collection
            
        except Exception as e:
            error_msg = f"Failed to create collection for {knowledge_type.display_name}: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def index_documents(
        self,
        documents: List[Document],
        knowledge_type: KnowledgeType,
        batch_size: int = 100,
        rebuild: bool = False
    ) -> IndexingResult:
        """
        Index documents in the appropriate knowledge type collection.
        
        Args:
            documents: List of documents to index
            knowledge_type: Type of knowledge being indexed
            batch_size: Number of documents to process in each batch
            rebuild: Whether to rebuild the collection from scratch
            
        Returns:
            IndexingResult with indexing statistics and any errors
        """
        knowledge_type = validate_knowledge_type(knowledge_type)
        start_time = datetime.now()
        
        result = IndexingResult(
            knowledge_type=knowledge_type,
            persona_id=self.persona_id,
            documents_processed=len(documents)
        )
        
        if not documents:
            result.add_warning("No documents provided for indexing")
            return result
        
        try:
            self.logger.info(
                f"Indexing {len(documents)} {knowledge_type.display_name.lower()} documents"
            )
            
            # Get or create collection
            collection = self.get_collection(knowledge_type)
            
            # Clear existing documents if rebuilding
            if rebuild:
                self.logger.info(f"Rebuilding collection for {knowledge_type.display_name}")
                self._clear_collection(knowledge_type)
                collection = self.get_collection(knowledge_type)  # Recreate after clear
            
            # Index documents in batches
            total_indexed = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                try:
                    # Add documents to collection
                    collection.add_documents(batch)
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
            if knowledge_type in self._collection_stats:
                self._collection_stats[knowledge_type].update({
                    'documents_indexed': total_indexed,
                    'last_indexed': datetime.now().isoformat(),
                    'last_batch_size': batch_size
                })
            
            # Calculate index size using isolated database path
            db_path = self.get_knowledge_db_path(knowledge_type)
            if db_path.exists():
                size_bytes = sum(
                    f.stat().st_size 
                    for f in db_path.rglob('*') 
                    if f.is_file()
                )
                result.index_size_mb = size_bytes / (1024 * 1024)
            
            if result.success:
                self.logger.info(
                    f"Successfully indexed {total_indexed} {knowledge_type.display_name.lower()} "
                    f"documents in {result.indexing_duration_seconds:.2f}s"
                )
            else:
                self.logger.warning(
                    f"Partially indexed {total_indexed}/{len(documents)} "
                    f"{knowledge_type.display_name.lower()} documents with {len(result.errors)} errors"
                )
            
        except Exception as e:
            error_msg = f"Critical error indexing {knowledge_type.display_name}: {e}"
            result.add_error(error_msg)
            self.logger.error(error_msg)
        
        return result
    
    def search(
        self,
        query: str,
        knowledge_type: KnowledgeType,
        k: int = 5,
        return_scores: bool = False,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        Search documents in a specific knowledge type collection.
        
        Args:
            query: Search query
            knowledge_type: Type of knowledge to search
            k: Number of results to return
            return_scores: Whether to return similarity scores
            filter_metadata: Optional metadata filters
            
        Returns:
            List of documents or (document, score) tuples
        """
        knowledge_type = validate_knowledge_type(knowledge_type)
        
        try:
            collection = self.get_collection(knowledge_type)
            
            self.logger.debug(
                f"Searching {knowledge_type.display_name.lower()} with query: {query[:100]}..."
            )
            
            if return_scores:
                results = collection.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
            else:
                results = collection.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
            
            self.logger.debug(
                f"Found {len(results)} results for {knowledge_type.display_name.lower()} search"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(
                f"Search failed for {knowledge_type.display_name}: {e}"
            )
            return []
    
    def get_collection_stats(self, knowledge_type: KnowledgeType) -> Dict[str, Any]:
        """
        Get statistics for a specific knowledge type collection.
        
        Args:
            knowledge_type: Type of knowledge to get stats for
            
        Returns:
            Dictionary with collection statistics
        """
        knowledge_type = validate_knowledge_type(knowledge_type)
        
        stats = {
            'knowledge_type': knowledge_type.value,
            'display_name': knowledge_type.display_name,
            'collection_exists': knowledge_type in self._collections,
            'persona_id': self.persona_id
        }
        
        # Add stored stats if available
        if knowledge_type in self._collection_stats:
            stats.update(self._collection_stats[knowledge_type])
        
        # Add current document count if collection exists
        if knowledge_type in self._collections:
            try:
                collection = self._collections[knowledge_type]
                # Try to get document count (may vary by Chroma version)
                if hasattr(collection, '_collection'):
                    stats['current_document_count'] = collection._collection.count()
                else:
                    stats['current_document_count'] = 'unknown'
            except Exception as e:
                stats['current_document_count'] = f'error: {e}'
        
        # Add directory size if exists using isolated database path
        db_path = self.get_knowledge_db_path(knowledge_type)
        if db_path.exists():
            try:
                size_bytes = sum(
                    f.stat().st_size 
                    for f in db_path.rglob('*') 
                    if f.is_file()
                )
                stats['directory_size_mb'] = round(size_bytes / (1024 * 1024), 2)
            except Exception:
                stats['directory_size_mb'] = 'unknown'
        else:
            stats['directory_size_mb'] = 0.0
        
        return stats
    
    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics for all knowledge types.
        
        Returns:
            Dictionary with statistics for all collections
        """
        all_stats = {
            'persona_id': self.persona_id,
            'embedding_model': self.embedding_model,
            'base_path': str(self.base_vector_path),
            'collections': {},
            'summary': {
                'total_collections': 0,
                'total_documents': 0,
                'total_size_mb': 0.0
            }
        }
        
        # Get stats for each knowledge type
        for knowledge_type in KnowledgeType:
            collection_stats = self.get_collection_stats(knowledge_type)
            all_stats['collections'][knowledge_type.value] = collection_stats
            
            if collection_stats.get('collection_exists', False):
                all_stats['summary']['total_collections'] += 1
                
                # Add to totals
                if 'documents_indexed' in collection_stats:
                    all_stats['summary']['total_documents'] += collection_stats['documents_indexed']
                
                if 'directory_size_mb' in collection_stats:
                    all_stats['summary']['total_size_mb'] += collection_stats['directory_size_mb']
        
        return all_stats
    
    def collection_exists(self, knowledge_type: KnowledgeType) -> bool:
        """
        Check if a collection exists for a knowledge type.
        
        Args:
            knowledge_type: Type of knowledge to check
            
        Returns:
            True if collection exists, False otherwise
        """
        knowledge_type = validate_knowledge_type(knowledge_type)
        
        # Check if already initialized
        if knowledge_type in self._collections:
            return True
        
        # Check if directory exists on disk using isolated database path
        db_path = self.get_knowledge_db_path(knowledge_type)
        return db_path.exists() and any(db_path.iterdir())
    
    def _clear_collection(self, knowledge_type: KnowledgeType):
        """
        Clear all documents from a collection.
        
        Args:
            knowledge_type: Type of knowledge collection to clear
        """
        knowledge_type = validate_knowledge_type(knowledge_type)
        
        try:
            if knowledge_type in self._collections:
                # Remove from memory
                del self._collections[knowledge_type]
            
            # Clear directory using isolated database path
            db_path = self.get_knowledge_db_path(knowledge_type)
            if db_path.exists():
                import shutil
                shutil.rmtree(db_path)
                self.logger.info(f"Cleared collection directory: {db_path}")
            
            # Reset stats
            if knowledge_type in self._collection_stats:
                del self._collection_stats[knowledge_type]
                
        except Exception as e:
            self.logger.error(f"Failed to clear collection for {knowledge_type.display_name}: {e}")
            raise
    
    def delete_collection(self, knowledge_type: KnowledgeType):
        """
        Permanently delete a knowledge type collection.
        
        Args:
            knowledge_type: Type of knowledge collection to delete
        """
        knowledge_type = validate_knowledge_type(knowledge_type)
        
        self.logger.warning(
            f"Deleting collection for {knowledge_type.display_name} - this cannot be undone"
        )
        
        self._clear_collection(knowledge_type)
    
    def cleanup(self):
        """Clean up resources and close connections."""
        self.logger.info("Cleaning up multi-knowledge store resources")
        
        # Close all collections
        for knowledge_type, collection in self._collections.items():
            try:
                if hasattr(collection, 'persist'):
                    collection.persist()
                self.logger.debug(f"Persisted collection for {knowledge_type.display_name}")
            except Exception as e:
                self.logger.warning(f"Failed to persist collection for {knowledge_type.display_name}: {e}")
        
        # Clear collections dict
        self._collections.clear()
        self._collection_stats.clear()
        
        self.logger.info("Multi-knowledge store cleanup complete")