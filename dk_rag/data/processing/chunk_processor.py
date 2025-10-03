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

# Optional Chinese tokenization
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


class ChunkProcessor:
    """
    Processes documents by splitting them into smaller chunks for vector storage.
    
    Supports configurable chunk sizes with overlap to maintain context
    across chunk boundaries.
    """
    
    def __init__(self, settings: Optional[Settings] = None, chunk_size: int = 500, chunk_overlap: int = 100, language: str = "en"):
        """
        Initialize the chunk processor.

        Args:
            settings: Application settings (chunk config will be extracted if provided)
            chunk_size: Target number of words/characters per chunk
            chunk_overlap: Number of words/characters to overlap between chunks
            language: Content language ('en' for English, 'zh' for Chinese)
        """
        self.logger = get_logger(__name__)
        self.language = language.strip() if language else "en"

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

        # For Chinese, warn if jieba is not available
        if self.language == "zh" and not JIEBA_AVAILABLE:
            self.logger.warning("jieba not available for Chinese text segmentation, using character-based chunking")

        unit = "characters" if self.language == "zh" else "words"
        self.logger.info(f"ChunkProcessor initialized: chunk_size={self.chunk_size} {unit}, overlap={self.chunk_overlap}, language={self.language}")
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of documents into smaller pieces with progress tracking.

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

        # Track statistics for debugging
        skipped_docs = 0
        error_docs = 0

        for i, doc in enumerate(documents, 1):
            # More frequent progress logging for Chinese to detect hangs
            if self.language == "zh":
                if i % 5 == 0 or i == total_docs or i <= 3:
                    self.logger.info(f"Chunking progress: {i}/{total_docs} documents ({i/total_docs*100:.1f}%)")
            else:
                if i % 10 == 0 or i == total_docs:
                    self.logger.info(f"Chunking progress: {i}/{total_docs} documents ({i/total_docs*100:.1f}%)")

            # Get document content size for logging
            content = doc.get('content', '')
            content_size = len(content)

            # Skip empty documents
            if not content:
                skipped_docs += 1
                self.logger.warning(f"Skipping empty document {i}: {doc.get('source', 'unknown')}")
                continue

            # Log first 3 documents in detail for Chinese text
            if self.language == "zh" and i <= 3:
                word_count = len(content.split()) if content else 0
                self.logger.info(f"Document {i} details: {content_size} chars, ~{word_count} tokens, source: {doc.get('source', 'unknown')[:50]}")

            try:
                # Chunk the document
                chunks = self.chunk_document(doc)
                all_chunks.extend(chunks)

                # Log if document produced many chunks (potential issue)
                if len(chunks) > 50:
                    self.logger.warning(f"Document {i} produced {len(chunks)} chunks (content size: {content_size} chars)")

            except Exception as e:
                error_docs += 1
                self.logger.error(f"Error chunking document {i} '{doc.get('source', 'unknown')}': {e}")
                # Continue processing other documents
                continue

        # Summary statistics
        if skipped_docs > 0 or error_docs > 0:
            self.logger.warning(f"Chunking summary: {skipped_docs} skipped, {error_docs} errors")

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

        # Language-specific chunking
        if self.language == "zh":
            return self._chunk_chinese_document(content, document)
        else:
            return self._chunk_english_document(content, document)

    def _chunk_english_document(self, content: str, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk English document by words."""
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

    def _chunk_chinese_document(self, content: str, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk Chinese document using word-aware segmentation.

        For unpunctuated transcribed Chinese text, uses jieba to segment into words,
        then chunks based on word boundaries to avoid mid-word splits.
        """
        char_count = len(content)

        # Log document size for debugging
        self.logger.debug(f"Chunking Chinese document '{document.get('source', 'unknown')}': {char_count} characters")

        # If document is smaller than chunk size, return as single chunk
        if char_count <= self.chunk_size:
            chunk = self._create_chunk(
                content=content,
                parent_document=document,
                chunk_index=0,
                total_chunks=1,
                start_word=0,
                end_word=char_count
            )
            return [chunk]

        # Use jieba for word segmentation if available
        if JIEBA_AVAILABLE:
            return self._chunk_chinese_by_words(content, document)
        else:
            # Fallback to simple character-based chunking with hard limits
            self.logger.warning("jieba not available, using simple character chunking")
            return self._chunk_chinese_by_characters_simple(content, document)

    def _chunk_chinese_by_words(self, content: str, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk Chinese text by word boundaries using jieba segmentation.

        This approach respects word boundaries in unpunctuated Chinese text,
        creating more semantically meaningful chunks.
        """
        try:
            # Segment text into words using jieba
            words = list(jieba.cut(content))
            total_words = len(words)

            self.logger.debug(f"Segmented into {total_words} words using jieba")

            # Convert chunk_size from characters to approximate word count
            # For Chinese, roughly 1.5-2.5 characters per word on average
            # Use conservative estimate: chunk_size chars / 2 = target words
            target_words_per_chunk = max(int(self.chunk_size / 2), 50)  # Minimum 50 words
            overlap_words = max(int(self.chunk_overlap / 2), 10)  # Minimum 10 words overlap

            # If document is smaller than target, return as single chunk
            if total_words <= target_words_per_chunk:
                chunk = self._create_chunk(
                    content=content,
                    parent_document=document,
                    chunk_index=0,
                    total_chunks=1,
                    start_word=0,
                    end_word=total_words
                )
                return [chunk]

            # Create overlapping chunks based on word boundaries
            chunks = []
            chunk_index = 0
            start_word_idx = 0

            # Safety limit to prevent infinite loops
            max_iterations = (total_words // max(1, target_words_per_chunk - overlap_words)) + 10

            while start_word_idx < total_words and chunk_index < max_iterations:
                # Calculate end word index
                end_word_idx = min(start_word_idx + target_words_per_chunk, total_words)

                # Extract chunk words and reconstruct text
                chunk_words = words[start_word_idx:end_word_idx]
                chunk_content = ''.join(chunk_words)

                # Create chunk
                chunk = self._create_chunk(
                    content=chunk_content,
                    parent_document=document,
                    chunk_index=chunk_index,
                    total_chunks=None,  # Will be set after all chunks are created
                    start_word=start_word_idx,
                    end_word=end_word_idx
                )

                chunks.append(chunk)
                chunk_index += 1

                # If this was the last chunk, break
                if end_word_idx >= total_words:
                    break

                # Move to next chunk with overlap
                next_start = end_word_idx - overlap_words

                # Ensure we don't go backwards and make progress
                if next_start <= start_word_idx:
                    next_start = end_word_idx - max(1, overlap_words // 2)

                # Ensure we're still making progress
                if next_start <= start_word_idx:
                    # Force at least 1 word progress
                    next_start = start_word_idx + 1

                start_word_idx = max(0, next_start)  # Prevent negative indices

            # Warn if we hit the iteration limit
            if chunk_index >= max_iterations:
                self.logger.warning(f"Hit max iteration limit ({max_iterations}) during word-based chunking")

            # Update total_chunks count in all chunks
            for chunk in chunks:
                chunk['total_chunks'] = len(chunks)

            self.logger.debug(f"Split document into {len(chunks)} word-based chunks (from {total_words} words)")
            return chunks

        except Exception as e:
            self.logger.error(f"Error in word-based Chinese chunking: {e}, falling back to character chunking")
            return self._chunk_chinese_by_characters_simple(content, document)

    def _chunk_chinese_by_characters_simple(self, content: str, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Simple character-based chunking with hard limits for safety.
        Used as fallback when jieba is not available or fails.
        """
        char_count = len(content)
        chunks = []
        chunk_index = 0
        start = 0

        # Hard limit on number of chunks to prevent infinite loops
        max_chunks = (char_count // (self.chunk_size - self.chunk_overlap)) + 10

        while start < char_count and chunk_index < max_chunks:
            # Calculate end position
            end = min(start + self.chunk_size, char_count)

            # For unpunctuated text, try to break at whitespace if available
            if end < char_count and end - start > 100:
                # Look for whitespace in the last 20% of the chunk
                search_start = end - int(self.chunk_size * 0.2)
                substring = content[search_start:end]

                # Find last whitespace character
                for i in range(len(substring) - 1, -1, -1):
                    if substring[i] in [' ', '\n', '\t', '\u3000']:  # Include full-width space
                        end = search_start + i + 1
                        break

            # Extract chunk content
            chunk_content = content[start:end]

            # Create chunk
            chunk = self._create_chunk(
                content=chunk_content,
                parent_document=document,
                chunk_index=chunk_index,
                total_chunks=None,
                start_word=start,
                end_word=end
            )

            chunks.append(chunk)
            chunk_index += 1

            # Move to next chunk with overlap
            next_start = end - self.chunk_overlap

            # Ensure we make progress
            if next_start <= start:
                next_start = start + max(1, self.chunk_size // 2)

            start = next_start

        # Update total_chunks count
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)

        if chunk_index >= max_chunks:
            self.logger.warning(f"Hit max chunks limit ({max_chunks}) for document, may be incomplete")

        self.logger.debug(f"Split document into {len(chunks)} character-based chunks")
        return chunks
    
    def _prepare_content(self, content: str) -> str:
        """Clean and prepare content for chunking."""
        if self.language == "zh":
            # For Chinese, preserve structure but remove excessive whitespace
            content = re.sub(r'\n{3,}', '\n\n', content)
            content = content.strip()
        else:
            # For English, normalize whitespace
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'\n{3,}', '\n\n', content)
            content = content.strip()
        return content

    def _split_chinese_text(self, content: str) -> List[str]:
        """
        Split Chinese text into tokens (characters or words).

        Args:
            content: Chinese text content

        Returns:
            List of tokens (jieba words if available, otherwise characters)
        """
        if JIEBA_AVAILABLE:
            # Use jieba for word segmentation
            return list(jieba.cut(content))
        else:
            # Fall back to character-based
            return list(content)

    def _find_chinese_sentence_boundary(self, text: str, target_pos: int) -> int:
        """
        Find the nearest Chinese sentence boundary near target position.

        Note: This function is deprecated for unpunctuated transcribed text.
        The new word-based chunking approach (_chunk_chinese_by_words) is preferred.

        Args:
            text: Chinese text
            target_pos: Target character position

        Returns:
            Adjusted position at sentence boundary
        """
        # Chinese sentence terminators (only useful if text has punctuation)
        terminators = ['。', '！', '？', '；', '\n']

        # Search forward for terminator (within 100 chars)
        for i in range(target_pos, min(target_pos + 100, len(text))):
            if text[i] in terminators:
                return i + 1

        # If no terminator found, return target
        # For unpunctuated text, this will always return target_pos
        return target_pos
    
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
        
        # Calculate word/character count based on language
        if self.language == "zh":
            # For Chinese, character count is more meaningful
            word_count = len(content) if content else 0
        else:
            # For English, use word count
            word_count = len(content.split()) if content else 0

        # Update with chunk-specific content and metadata
        chunk.update({
            'content': content,
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'start_word': start_word,
            'end_word': end_word,
            'word_count': word_count,
            'char_count': len(content) if content else 0,
            'is_chunk': True,
            'parent_source': parent_document.get('source', 'unknown'),
            'parent_filename': parent_document.get('filename', 'unknown'),
            'language': self.language
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