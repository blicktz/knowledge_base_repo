"""
Core Beliefs Document Builder

Creates searchable documents from core belief data extracted from Phase 1 JSON.
Optimizes content for vector similarity search by combining belief statements,
categories, and supporting evidence into coherent searchable text.
"""

import hashlib
from typing import List, Dict, Any
from langchain.schema import Document

from .base_builder import BaseKnowledgeBuilder
from ...models.knowledge_types import KnowledgeType


class CoreBeliefsBuilder(BaseKnowledgeBuilder):
    """
    Document builder for core beliefs knowledge base.
    
    Creates documents optimized for semantic search of foundational principles,
    values, and belief statements with supporting evidence.
    """
    
    def __init__(self):
        """Initialize the core beliefs builder."""
        super().__init__(KnowledgeType.CORE_BELIEFS)
    
    def get_required_fields(self) -> List[str]:
        """
        Get required fields for core belief items.
        
        Returns:
            List of required field names
        """
        return ['statement', 'category', 'supporting_evidence']
    
    def build_documents(
        self,
        knowledge_data: List[Dict[str, Any]],
        persona_id: str,
        source_file: str = ""
    ) -> List[Document]:
        """
        Build searchable documents from core belief data.
        
        Args:
            knowledge_data: List of core belief items
            persona_id: ID of the persona
            source_file: Source JSON file path
            
        Returns:
            List of LangChain Documents ready for vector indexing
        """
        self.logger.info(f"Building documents for {len(knowledge_data)} core beliefs")
        
        # Filter out invalid items
        valid_items, errors = self.filter_valid_items(knowledge_data)
        
        if errors:
            self.logger.warning(f"Found {len(errors)} validation errors in core beliefs")
            for error in errors[:5]:  # Log first 5 errors
                self.logger.warning(f"Validation error: {error}")
        
        documents = []
        
        for i, item in enumerate(valid_items):
            try:
                # Create searchable content
                content = self.format_content(item)
                
                # Create comprehensive metadata
                metadata = self.create_core_belief_metadata(
                    item, persona_id, source_file, i
                )
                
                # Create document
                document = Document(
                    page_content=content,
                    metadata=metadata
                )
                
                documents.append(document)
                
            except Exception as e:
                error_msg = f"Failed to build document for core belief {i}: {e}"
                self.logger.error(error_msg)
                continue
        
        self.logger.info(f"Successfully built {len(documents)} core belief documents")
        
        # Log statistics
        stats = self.get_document_stats(documents)
        self.logger.debug(f"Core belief document stats: {stats}")
        
        return documents
    
    def format_content(self, item: Dict[str, Any]) -> str:
        """
        Format core belief into searchable text content.
        
        Creates a coherent text that includes the belief statement, category,
        and supporting evidence for optimal vector similarity matching.
        
        Args:
            item: Core belief data dictionary
            
        Returns:
            Formatted searchable text content
        """
        statement = item.get('statement', '').strip()
        category = item.get('category', '').strip()
        evidence = item.get('supporting_evidence', [])
        
        content_parts = []
        
        # Belief statement (primary content)
        if statement:
            content_parts.append(f"Belief: {statement}")
        
        # Category (topical classification)
        if category:
            content_parts.append(f"Category: {category}")
        
        # Supporting evidence (validation and context)
        if evidence:
            content_parts.append("Supporting Evidence:")
            for i, evidence_item in enumerate(evidence, 1):
                evidence_text = str(evidence_item).strip()
                if evidence_text:
                    content_parts.append(f"• {evidence_text}")
        
        # Combine all parts with clear separation
        content = "\n\n".join(content_parts)
        
        return content
    
    def create_core_belief_metadata(
        self,
        item: Dict[str, Any],
        persona_id: str,
        source_file: str,
        doc_index: int
    ) -> Dict[str, Any]:
        """
        Create comprehensive metadata for core belief document.
        
        Args:
            item: Core belief data
            persona_id: Persona identifier
            source_file: Source file path
            doc_index: Document index
            
        Returns:
            Complete metadata dictionary
        """
        # Start with base metadata
        metadata = self.create_base_metadata(item, persona_id, source_file, doc_index)

        # Extract statement once to avoid duplication
        statement = item.get('statement', '').strip()

        # Create hash-based unique identifier to avoid data duplication
        # (Core beliefs don't have a separate 'name' field in the artifact like mental models do)
        statement_hash = hashlib.md5(statement.encode()).hexdigest()[:16]
        unique_name = f"belief_{statement_hash}"

        # Add core belief specific metadata
        metadata.update({
            # Hash-based unique identifier (no duplication with statement)
            'name': unique_name,

            # Core belief fields
            'statement': statement,
            'category': item.get('category', '').strip(),

            # Structured data (stored as delimited string for ChromaDB compatibility)
            'supporting_evidence_text': '\n'.join(str(e) for e in item.get('supporting_evidence', [])),

            # Derived metrics for search optimization
            'evidence_count': len(item.get('supporting_evidence', [])),
            'evidence_strength': self._calculate_evidence_strength(item),
            'belief_type': self._classify_belief_type(item),
            
            # Search optimization
            'semantic_category': self._normalize_category(item.get('category', '')),
            'conviction_level': self._assess_conviction_level(item)
        })
        
        return metadata
    
    def _calculate_evidence_strength(self, item: Dict[str, Any]) -> float:
        """
        Calculate evidence strength based on quantity and quality of evidence.
        
        Args:
            item: Core belief data
            
        Returns:
            Evidence strength score from 0.0 to 1.0
        """
        evidence_list = item.get('supporting_evidence', [])
        
        if not evidence_list:
            return 0.0
        
        # Quantity component (number of evidence items)
        quantity_score = min(len(evidence_list) / 5.0, 1.0)  # Max 5 items = 1.0
        
        # Quality component (average length of evidence items)
        total_length = sum(len(str(evidence)) for evidence in evidence_list)
        avg_length = total_length / len(evidence_list)
        quality_score = min(avg_length / 200.0, 1.0)  # Max 200 chars avg = 1.0
        
        # Weighted combination
        evidence_strength = (quantity_score * 0.4) + (quality_score * 0.6)
        
        return round(evidence_strength, 3)
    
    def _classify_belief_type(self, item: Dict[str, Any]) -> str:
        """
        Classify the type of belief based on content analysis.
        
        Args:
            item: Core belief data
            
        Returns:
            Belief type classification
        """
        statement = item.get('statement', '').lower()
        category = item.get('category', '').lower()
        
        content_text = f"{statement} {category}"
        
        # Business/professional beliefs
        if any(keyword in content_text for keyword in ['business', 'marketing', 'sales', 'customer', 'profit']):
            return "business"
        
        # Personal development beliefs
        elif any(keyword in content_text for keyword in ['success', 'mindset', 'habit', 'growth', 'development']):
            return "personal_development"
        
        # Relationship beliefs
        elif any(keyword in content_text for keyword in ['relationship', 'people', 'trust', 'team', 'leadership']):
            return "relationships"
        
        # Financial beliefs
        elif any(keyword in content_text for keyword in ['money', 'wealth', 'financial', 'invest', 'income']):
            return "financial"
        
        # Philosophical beliefs
        elif any(keyword in content_text for keyword in ['life', 'truth', 'value', 'principle', 'philosophy']):
            return "philosophical"
        
        # Strategy/methodology beliefs
        elif any(keyword in content_text for keyword in ['strategy', 'method', 'approach', 'system', 'process']):
            return "strategic"
        
        else:
            return "general"
    
    def _normalize_category(self, category: str) -> str:
        """
        Normalize category for consistent searching.
        
        Args:
            category: Original category string
            
        Returns:
            Normalized category
        """
        if not category:
            return "uncategorized"
        
        normalized = category.lower().strip()
        
        # Map common variations to standard categories
        category_mapping = {
            'marketing': ['marketing', 'advertising', 'promotion'],
            'sales': ['sales', 'selling', 'salesmanship'],
            'entrepreneurship': ['entrepreneurship', 'business', 'startup', 'enterprise'],
            'mindset': ['mindset', 'psychology', 'mental', 'thinking'],
            'relationships': ['relationships', 'people', 'networking', 'social'],
            'management': ['management', 'leadership', 'team', 'organization'],
            'finance': ['finance', 'money', 'wealth', 'financial'],
            'strategy': ['strategy', 'planning', 'tactics', 'approach'],
            'productivity': ['productivity', 'efficiency', 'time', 'performance'],
            'communication': ['communication', 'speaking', 'presentation', 'persuasion']
        }
        
        for standard_category, variations in category_mapping.items():
            if any(var in normalized for var in variations):
                return standard_category
        
        return normalized
    
    def _assess_conviction_level(self, item: Dict[str, Any]) -> str:
        """
        Assess the conviction level of the belief based on language strength.
        
        Args:
            item: Core belief data
            
        Returns:
            Conviction level assessment
        """
        statement = item.get('statement', '').lower()
        confidence_score = item.get('confidence_score', 0.0)
        
        # Language indicators of strong conviction
        strong_indicators = [
            'must', 'never', 'always', 'essential', 'crucial', 'vital', 'critical',
            'fundamental', 'absolutely', 'definitely', 'certainly', 'undoubtedly'
        ]
        
        # Language indicators of moderate conviction
        moderate_indicators = [
            'should', 'important', 'necessary', 'typically', 'generally',
            'usually', 'often', 'likely', 'probably', 'tend to'
        ]
        
        # Count indicators
        strong_count = sum(1 for indicator in strong_indicators if indicator in statement)
        moderate_count = sum(1 for indicator in moderate_indicators if indicator in statement)
        
        # Assess based on confidence score and language
        if confidence_score >= 0.9 or strong_count >= 2:
            return "very_high"
        elif confidence_score >= 0.8 or strong_count >= 1:
            return "high"
        elif confidence_score >= 0.7 or moderate_count >= 2:
            return "moderate"
        elif confidence_score >= 0.6 or moderate_count >= 1:
            return "moderate_low"
        else:
            return "low"
    
    def _extract_primary_keywords(self, item: Dict[str, Any]) -> List[str]:
        """
        Extract primary keywords for search optimization.
        
        Args:
            item: Core belief data
            
        Returns:
            List of primary keywords
        """
        keywords = set()
        
        # Extract from statement (most important)
        statement = item.get('statement', '')
        if statement:
            statement_words = self._extract_meaningful_words(statement)
            # Prioritize longer, more specific words from statement
            keywords.update([word for word in statement_words if len(word) > 3])
        
        # Extract from category
        category = item.get('category', '')
        if category:
            category_words = self._extract_meaningful_words(category)
            keywords.update(category_words)
        
        # Extract key terms from evidence (selective)
        evidence_list = item.get('supporting_evidence', [])
        for evidence in evidence_list[:3]:  # Only first 3 pieces of evidence
            if evidence:
                evidence_words = self._extract_meaningful_words(str(evidence))
                # Only include longer, less common words from evidence
                keywords.update([word for word in evidence_words if len(word) > 5])
        
        return sorted(list(keywords))[:10]  # Limit to top 10 keywords
    
    def _extract_meaningful_words(self, text: str) -> List[str]:
        """
        Extract meaningful words from text, filtering out common stopwords.
        
        Args:
            text: Text to extract words from
            
        Returns:
            List of meaningful words
        """
        # Comprehensive stopwords for belief statements
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'must', 'shall', 'it', 'he', 'she', 'they', 'we', 'you', 'i',
            'who', 'what', 'when', 'where', 'why', 'how', 'if', 'then', 'than',
            'as', 'so', 'not', 'no', 'yes', 'all', 'any', 'some', 'most', 'many',
            'few', 'more', 'less', 'much', 'very', 'too', 'also', 'just', 'only'
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