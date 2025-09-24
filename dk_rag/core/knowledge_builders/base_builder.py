"""
Base Knowledge Builder

Abstract base class for all knowledge document builders.
Provides common functionality and interface for building searchable documents.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.schema import Document

from ...models.knowledge_types import KnowledgeType
from ...utils.logging import get_logger


class BaseKnowledgeBuilder(ABC):
    """
    Abstract base class for knowledge document builders.
    
    Defines the interface that all knowledge builders must implement
    and provides common functionality for document creation and validation.
    """
    
    def __init__(self, knowledge_type: KnowledgeType):
        """
        Initialize the builder.
        
        Args:
            knowledge_type: The type of knowledge this builder handles
        """
        self.knowledge_type = knowledge_type
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def build_documents(
        self,
        knowledge_data: List[Dict[str, Any]],
        persona_id: str,
        source_file: str = ""
    ) -> List[Document]:
        """
        Build searchable documents from knowledge data.
        
        Args:
            knowledge_data: List of processed knowledge items
            persona_id: ID of the persona these documents belong to
            source_file: Path to source file for provenance
            
        Returns:
            List of LangChain Document objects ready for indexing
        """
        pass
    
    @abstractmethod
    def format_content(self, item: Dict[str, Any]) -> str:
        """
        Format a knowledge item into searchable text content.
        
        Args:
            item: Single knowledge item
            
        Returns:
            Formatted text content for embedding and search
        """
        pass
    
    def create_base_metadata(
        self,
        item: Dict[str, Any],
        persona_id: str,
        source_file: str = "",
        doc_index: int = 0
    ) -> Dict[str, Any]:
        """
        Create base metadata common to all document types.
        
        Args:
            item: Knowledge item data
            persona_id: Persona identifier
            source_file: Source file path
            doc_index: Index of this document in the batch
            
        Returns:
            Base metadata dictionary
        """
        return {
            # Document identification
            'type': self.knowledge_type.value,
            'persona_id': persona_id,
            'doc_index': doc_index,
            'document_id': f"{persona_id}_{self.knowledge_type.value}_{doc_index}",
            
            # Source information
            'source_file': source_file,
            'content_hash': item.get('content_hash', ''),
            'original_index': item.get('index', doc_index),
            
            # Processing metadata
            'processing_timestamp': item.get('processing_timestamp', datetime.now().isoformat()),
            'indexing_timestamp': datetime.now().isoformat(),
            
            # Quality metrics
            'confidence_score': item.get('confidence_score', 0.0),
            'frequency': item.get('frequency', 0)
        }
    
    def validate_knowledge_item(self, item: Dict[str, Any], index: int) -> List[str]:
        """
        Validate a knowledge item before processing.
        
        Args:
            item: Knowledge item to validate
            index: Index for error reporting
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not isinstance(item, dict):
            errors.append(f"Item {index} is not a dictionary")
            return errors
        
        # Check for required base fields (can be overridden by subclasses)
        required_fields = self.get_required_fields()
        for field in required_fields:
            if field not in item:
                errors.append(f"Item {index} missing required field: {field}")
            elif not item[field]:
                errors.append(f"Item {index} has empty required field: {field}")
        
        return errors
    
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """
        Get list of required fields for this knowledge type.
        
        Returns:
            List of required field names
        """
        pass
    
    def filter_valid_items(
        self, 
        knowledge_data: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Filter knowledge data to only include valid items.
        
        Args:
            knowledge_data: List of knowledge items to filter
            
        Returns:
            Tuple of (valid_items, error_messages)
        """
        valid_items = []
        errors = []
        
        for i, item in enumerate(knowledge_data):
            validation_errors = self.validate_knowledge_item(item, i)
            if validation_errors:
                errors.extend(validation_errors)
            else:
                valid_items.append(item)
        
        if errors:
            self.logger.warning(f"Filtered out {len(knowledge_data) - len(valid_items)} invalid items")
        
        return valid_items, errors
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about the created documents.
        
        Args:
            documents: List of created documents
            
        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {'count': 0}
        
        # Calculate basic stats
        content_lengths = [len(doc.page_content) for doc in documents]
        confidence_scores = [
            doc.metadata.get('confidence_score', 0.0) 
            for doc in documents
        ]
        
        stats = {
            'count': len(documents),
            'avg_content_length': sum(content_lengths) / len(content_lengths),
            'min_content_length': min(content_lengths),
            'max_content_length': max(content_lengths),
            'avg_confidence_score': sum(confidence_scores) / len(confidence_scores),
            'knowledge_type': self.knowledge_type.value
        }
        
        return stats