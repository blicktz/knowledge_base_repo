"""
Mental Models Document Builder

Creates searchable documents from mental model data extracted from Phase 1 JSON.
Optimizes content for vector similarity search by combining name, description, 
steps, and categories into a coherent searchable text.
"""

from typing import List, Dict, Any
from langchain.schema import Document

from .base_builder import BaseKnowledgeBuilder
from ...models.knowledge_types import KnowledgeType


class MentalModelsBuilder(BaseKnowledgeBuilder):
    """
    Document builder for mental models knowledge base.
    
    Creates documents optimized for semantic search of problem-solving
    frameworks, methodologies, and structured approaches.
    """
    
    def __init__(self):
        """Initialize the mental models builder."""
        super().__init__(KnowledgeType.MENTAL_MODELS)
    
    def get_required_fields(self) -> List[str]:
        """
        Get required fields for mental model items.
        
        Returns:
            List of required field names
        """
        return ['name', 'description', 'steps', 'categories']
    
    def build_documents(
        self,
        knowledge_data: List[Dict[str, Any]],
        persona_id: str,
        source_file: str = ""
    ) -> List[Document]:
        """
        Build searchable documents from mental model data.
        
        Args:
            knowledge_data: List of mental model items
            persona_id: ID of the persona
            source_file: Source JSON file path
            
        Returns:
            List of LangChain Documents ready for vector indexing
        """
        self.logger.info(f"Building documents for {len(knowledge_data)} mental models")
        
        # Filter out invalid items
        valid_items, errors = self.filter_valid_items(knowledge_data)
        
        if errors:
            self.logger.warning(f"Found {len(errors)} validation errors in mental models")
            for error in errors[:5]:  # Log first 5 errors
                self.logger.warning(f"Validation error: {error}")
        
        documents = []
        
        for i, item in enumerate(valid_items):
            try:
                # Create searchable content
                content = self.format_content(item)
                
                # Create comprehensive metadata
                metadata = self.create_mental_model_metadata(
                    item, persona_id, source_file, i
                )
                
                # Create document
                document = Document(
                    page_content=content,
                    metadata=metadata
                )
                
                documents.append(document)
                
            except Exception as e:
                error_msg = f"Failed to build document for mental model {i}: {e}"
                self.logger.error(error_msg)
                continue
        
        self.logger.info(f"Successfully built {len(documents)} mental model documents")
        
        # Log statistics
        stats = self.get_document_stats(documents)
        self.logger.debug(f"Mental model document stats: {stats}")
        
        return documents
    
    def format_content(self, item: Dict[str, Any]) -> str:
        """
        Format mental model into searchable text content.
        
        Creates a coherent text that includes the framework name, description,
        detailed steps, and categories for optimal vector similarity matching.
        
        Args:
            item: Mental model data dictionary
            
        Returns:
            Formatted searchable text content
        """
        name = item.get('name', '').strip()
        description = item.get('description', '').strip()
        steps = item.get('steps', [])
        categories = item.get('categories', [])
        
        content_parts = []
        
        # Framework name (primary identifier)
        if name:
            content_parts.append(f"Framework: {name}")
        
        # Description (context and purpose)
        if description:
            content_parts.append(f"Description: {description}")
        
        # Steps (actionable process)
        if steps:
            content_parts.append("Steps:")
            for i, step in enumerate(steps, 1):
                step_text = str(step).strip()
                if step_text:
                    # Clean up step numbering if already present
                    if step_text.startswith(f"{i}."):
                        content_parts.append(step_text)
                    else:
                        content_parts.append(f"{i}. {step_text}")
        
        # Categories (topical classification)
        if categories:
            categories_text = ", ".join(str(cat).strip() for cat in categories if str(cat).strip())
            if categories_text:
                content_parts.append(f"Categories: {categories_text}")
        
        # Combine all parts with clear separation
        content = "\n\n".join(content_parts)
        
        return content
    
    def create_mental_model_metadata(
        self,
        item: Dict[str, Any],
        persona_id: str,
        source_file: str,
        doc_index: int
    ) -> Dict[str, Any]:
        """
        Create comprehensive metadata for mental model document.
        
        Args:
            item: Mental model data
            persona_id: Persona identifier
            source_file: Source file path
            doc_index: Document index
            
        Returns:
            Complete metadata dictionary
        """
        # Start with base metadata
        metadata = self.create_base_metadata(item, persona_id, source_file, doc_index)
        
        # Add mental model specific metadata
        metadata.update({
            # Core mental model fields (simple types only - complex data is in document content)
            'name': item.get('name', '').strip(),
            'description': item.get('description', '').strip(),
            
            # Derived metrics for search optimization
            'steps_count': len(item.get('steps', [])),
            'categories_count': len(item.get('categories', [])),
            'complexity_score': self._calculate_complexity_score(item),
            
            # Search hints for query matching
            'primary_category': self._get_primary_category(item),
            'framework_type': self._determine_framework_type(item)
        })
        
        return metadata
    
    def _calculate_complexity_score(self, item: Dict[str, Any]) -> float:
        """
        Calculate complexity score based on steps and description length.
        
        Args:
            item: Mental model data
            
        Returns:
            Complexity score from 0.0 to 1.0
        """
        steps_count = len(item.get('steps', []))
        description_length = len(item.get('description', ''))
        
        # Normalize components
        steps_score = min(steps_count / 10.0, 1.0)  # Max 10 steps = 1.0
        description_score = min(description_length / 500.0, 1.0)  # Max 500 chars = 1.0
        
        # Weighted combination
        complexity_score = (steps_score * 0.7) + (description_score * 0.3)
        
        return round(complexity_score, 3)
    
    def _get_primary_category(self, item: Dict[str, Any]) -> str:
        """
        Get the primary category for this mental model.
        
        Args:
            item: Mental model data
            
        Returns:
            Primary category string
        """
        categories = item.get('categories', [])
        if categories:
            # Return first category as primary
            return str(categories[0]).strip().lower()
        return "uncategorized"
    
    def _determine_framework_type(self, item: Dict[str, Any]) -> str:
        """
        Determine the type of framework based on content analysis.
        
        Args:
            item: Mental model data
            
        Returns:
            Framework type classification
        """
        name = item.get('name', '').lower()
        description = item.get('description', '').lower()
        categories = [str(cat).lower() for cat in item.get('categories', [])]
        
        # Content-based classification
        content_text = f"{name} {description} {' '.join(categories)}"
        
        if any(keyword in content_text for keyword in ['method', 'methodology', 'process', 'system']):
            return "methodology"
        elif any(keyword in content_text for keyword in ['framework', 'model', 'approach']):
            return "framework"
        elif any(keyword in content_text for keyword in ['strategy', 'tactic', 'technique']):
            return "strategy"
        elif any(keyword in content_text for keyword in ['principle', 'rule', 'law']):
            return "principle"
        else:
            return "concept"
    
    def _extract_search_keywords(self, item: Dict[str, Any]) -> List[str]:
        """
        Extract important keywords for search optimization.
        
        Args:
            item: Mental model data
            
        Returns:
            List of search keywords
        """
        keywords = set()
        
        # Extract from name
        name = item.get('name', '')
        if name:
            # Split on common delimiters and extract meaningful words
            name_words = self._extract_meaningful_words(name)
            keywords.update(name_words)
        
        # Extract from categories
        categories = item.get('categories', [])
        for category in categories:
            if category:
                cat_words = self._extract_meaningful_words(str(category))
                keywords.update(cat_words)
        
        # Extract from description (key terms only)
        description = item.get('description', '')
        if description:
            desc_words = self._extract_meaningful_words(description)
            # Only include longer words from description to avoid noise
            keywords.update([word for word in desc_words if len(word) > 4])
        
        return sorted(list(keywords))
    
    def _extract_meaningful_words(self, text: str) -> List[str]:
        """
        Extract meaningful words from text, filtering out common stopwords.
        
        Args:
            text: Text to extract words from
            
        Returns:
            List of meaningful words
        """
        # Basic stopwords to filter out
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'must', 'shall', 'it', 'he', 'she', 'they', 'we', 'you', 'i'
        }
        
        # Extract words and clean them
        words = []
        for word in text.lower().split():
            # Remove punctuation and clean
            clean_word = ''.join(char for char in word if char.isalnum()).strip()
            
            # Filter meaningful words
            if (clean_word and 
                len(clean_word) > 2 and 
                clean_word not in stopwords and
                not clean_word.isdigit()):
                words.append(clean_word)
        
        return words