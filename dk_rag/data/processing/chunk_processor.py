"""
Text chunking processor for the Virtual Influencer Persona Agent.

This module handles splitting documents into smaller, overlapping chunks
for efficient vector storage and retrieval.
"""

import re
import uuid
from typing import List, Dict, Any, Optional
from ...config.settings import Settings
from ...utils.logging import get_logger


class ChunkProcessor:
    """
    Processes documents by splitting them into smaller chunks for vector storage.
    
    Supports configurable chunk sizes with overlap to maintain context
    across chunk boundaries.
    """
    
    def __init__(self, settings: Optional[Settings] = None, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the chunk processor.
        
        Args:
            settings: Application settings (chunk config will be extracted if provided)
            chunk_size: Target number of words per chunk
            chunk_overlap: Number of words to overlap between chunks
        """
        self.logger = get_logger(__name__)
        
        # Extract chunk configuration from settings if provided
        if settings and hasattr(settings, 'vector_db'):
            config = settings.vector_db.config if hasattr(settings.vector_db, 'config') else settings.vector_db
            if isinstance(config, dict):
                self.chunk_size = config.get('chunk_size', chunk_size)
                self.chunk_overlap = config.get('chunk_overlap', chunk_overlap)
            else:
                self.chunk_size = getattr(config, 'chunk_size', chunk_size)
                self.chunk_overlap = getattr(config, 'chunk_overlap', chunk_overlap)
        else:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        
        self.logger.info(f"ChunkProcessor initialized: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of documents into smaller pieces.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of chunk dictionaries
        """
        if not documents:
            return []
        
        all_chunks = []
        total_docs = len(documents)
        
        self.logger.info(f"Starting to chunk {total_docs} documents...")
        
        for i, doc in enumerate(documents, 1):
            # Log progress every 10 documents or for the last document
            if i % 10 == 0 or i == total_docs:
                self.logger.info(f"Chunking progress: {i}/{total_docs} documents ({i/total_docs*100:.1f}%)")
            
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        self.logger.info(f"Chunking complete: {len(documents)} documents → {len(all_chunks)} chunks")
        return all_chunks
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single document into smaller pieces.
        
        Args:
            document: Document dictionary with 'content' and metadata
            
        Returns:
            List of chunk dictionaries
        """
        content = document.get('content', '')
        if not content:
            return []
        
        # Clean and prepare content
        content = self._prepare_content(content)
        words = content.split()
        
        # If document is smaller than chunk size, return as single chunk
        if len(words) <= self.chunk_size:
            chunk = self._create_chunk(
                content=content,
                parent_document=document,
                chunk_index=0,
                total_chunks=1,
                start_word=0,
                end_word=len(words)
            )
            return [chunk]
        
        # Split into overlapping chunks
        chunks = []
        chunk_index = 0
        start = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_content = ' '.join(chunk_words)
            
            chunk = self._create_chunk(
                content=chunk_content,
                parent_document=document,
                chunk_index=chunk_index,
                total_chunks=None,  # Will be set after all chunks are created
                start_word=start,
                end_word=end
            )
            
            chunks.append(chunk)
            chunk_index += 1
            
            # Move start position for next chunk (with overlap)
            start = max(start + self.chunk_size - self.chunk_overlap, start + 1)
        
        # Update total_chunks count in all chunks
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)
        
        self.logger.debug(f"Split document '{document.get('source', 'unknown')}' into {len(chunks)} chunks")
        return chunks
    
    def _prepare_content(self, content: str) -> str:
        """Clean and prepare content for chunking."""
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        # Remove excessive line breaks
        content = re.sub(r'\n{3,}', '\n\n', content)
        # Strip leading/trailing whitespace
        content = content.strip()
        return content
    
    def _create_chunk(
        self, 
        content: str, 
        parent_document: Dict[str, Any],
        chunk_index: int,
        total_chunks: Optional[int],
        start_word: int,
        end_word: int
    ) -> Dict[str, Any]:
        """Create a chunk dictionary with metadata."""
        
        # Generate unique chunk ID
        chunk_id = str(uuid.uuid4())
        
        # Start with parent document metadata
        chunk = parent_document.copy()
        
        # Update with chunk-specific content and metadata
        chunk.update({
            'content': content,
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'start_word': start_word,
            'end_word': end_word,
            'word_count': len(content.split()) if content else 0,
            'char_count': len(content) if content else 0,
            'is_chunk': True,
            'parent_source': parent_document.get('source', 'unknown'),
            'parent_filename': parent_document.get('filename', 'unknown')
        })
        
        # Create a descriptive source for the chunk
        parent_source = parent_document.get('source', 'unknown')
        chunk['source'] = f"{parent_source}#chunk_{chunk_index}"
        
        return chunk
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Convenience method to chunk raw text.
        
        Args:
            text: Raw text content to chunk
            metadata: Optional metadata to include in chunks
            
        Returns:
            List of chunk dictionaries
        """
        if metadata is None:
            metadata = {}
        
        document = {
            'content': text,
            'source': metadata.get('source', 'text_input'),
            **metadata
        }
        
        return self.chunk_document(document)
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about a list of chunks."""
        if not chunks:
            return {
                'total_chunks': 0,
                'total_words': 0,
                'total_characters': 0,
                'avg_words_per_chunk': 0,
                'avg_chars_per_chunk': 0,
                'unique_parent_documents': 0
            }
        
        total_words = sum(chunk.get('word_count', 0) for chunk in chunks)
        total_chars = sum(chunk.get('char_count', 0) for chunk in chunks)
        
        # Count unique parent documents
        parent_sources = set(chunk.get('parent_source', 'unknown') for chunk in chunks)
        
        return {
            'total_chunks': len(chunks),
            'total_words': total_words,
            'total_characters': total_chars,
            'avg_words_per_chunk': total_words / len(chunks),
            'avg_chars_per_chunk': total_chars / len(chunks),
            'unique_parent_documents': len(parent_sources),
            'largest_chunk_words': max((chunk.get('word_count', 0) for chunk in chunks), default=0),
            'smallest_chunk_words': min((chunk.get('word_count', 0) for chunk in chunks), default=0)
        }
    
    def merge_chunks(self, chunks: List[Dict[str, Any]], max_words: int = 1000) -> List[Dict[str, Any]]:
        """
        Merge small chunks together up to a maximum word limit.
        
        Useful for optimizing chunk sizes when many small chunks are created.
        
        Args:
            chunks: List of chunk dictionaries
            max_words: Maximum words per merged chunk
            
        Returns:
            List of merged chunk dictionaries
        """
        if not chunks:
            return []
        
        merged_chunks = []
        current_merge = None
        current_words = 0
        
        for chunk in chunks:
            chunk_words = chunk.get('word_count', 0)
            
            # If this chunk alone exceeds max_words, add it separately
            if chunk_words > max_words:
                # Add any pending merge first
                if current_merge:
                    merged_chunks.append(current_merge)
                    current_merge = None
                    current_words = 0
                
                merged_chunks.append(chunk)
                continue
            
            # If adding this chunk would exceed max_words, finalize current merge
            if current_merge and current_words + chunk_words > max_words:
                merged_chunks.append(current_merge)
                current_merge = None
                current_words = 0
            
            # Start new merge or add to existing merge
            if current_merge is None:
                current_merge = chunk.copy()
                current_words = chunk_words
            else:
                # Merge the content
                current_merge['content'] = current_merge['content'] + ' ' + chunk['content']
                current_merge['word_count'] = current_merge.get('word_count', 0) + chunk_words
                current_merge['char_count'] = len(current_merge['content'])
                current_merge['end_word'] = chunk.get('end_word', current_merge.get('end_word', 0))
                current_words += chunk_words
        
        # Add final merge if any
        if current_merge:
            merged_chunks.append(current_merge)
        
        self.logger.info(f"Merged {len(chunks)} chunks into {len(merged_chunks)} merged chunks")
        return merged_chunks
    
    def filter_chunks_by_size(self, chunks: List[Dict[str, Any]], min_words: int = 10, max_words: int = 2000) -> List[Dict[str, Any]]:
        """Filter chunks by word count criteria."""
        filtered = []
        
        for chunk in chunks:
            word_count = chunk.get('word_count', 0)
            if min_words <= word_count <= max_words:
                filtered.append(chunk)
            else:
                self.logger.debug(f"Filtered out chunk with {word_count} words (not in range {min_words}-{max_words})")
        
        if len(filtered) != len(chunks):
            self.logger.info(f"Filtered {len(chunks)} chunks to {len(filtered)} chunks based on size criteria")
        
        return filtered