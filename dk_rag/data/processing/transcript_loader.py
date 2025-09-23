"""
Document loading and text processing utilities for the Virtual Influencer Persona Agent.

This module handles loading various document formats, text cleaning, and
preparation for analysis and vector storage.
"""

import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator
import logging

from ...config.settings import Settings
from ...utils.logging import get_logger


class Document:
    """Represents a loaded document with metadata."""
    
    def __init__(self, content: str, source: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a document.
        
        Args:
            content: Document text content
            source: Source path or identifier
            metadata: Optional metadata dictionary
        """
        self.content = content
        self.source = source
        self.metadata = metadata or {}
        self.word_count = len(content.split()) if content else 0
        self.char_count = len(content) if content else 0
    
    def __str__(self) -> str:
        return f"Document(source={self.source}, words={self.word_count})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Document to dictionary format expected by existing code."""
        return {
            'content': self.content,
            'source': self.source,
            **self.metadata
        }


class TranscriptLoader:
    """
    Loads and processes documents from various sources.
    
    Supports text files, markdown files, and other text-based formats
    with automatic encoding detection and text cleaning.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the transcript loader.
        
        Args:
            settings: Optional application settings
        """
        self.settings = settings
        self.logger = get_logger(__name__)
        
        # Text cleaning patterns
        self.cleanup_patterns = [
            (r'\r\n', '\n'),  # Normalize line endings
            (r'\r', '\n'),    # Convert old Mac line endings
            (r'\n{3,}', '\n\n'),  # Reduce excessive line breaks
            (r'[ \t]+', ' '),  # Normalize whitespace
            (r'^\s+', '', re.MULTILINE),  # Remove leading whitespace
            (r'\s+$', '', re.MULTILINE),  # Remove trailing whitespace
        ]
        
        # Supported file extensions
        self.supported_extensions = {'.txt', '.md', '.markdown', '.text'}
    
    def load_file(self, file_path: Union[str, Path]) -> Optional[Document]:
        """
        Load a single file.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Document object or None if loading failed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.warning(f"File not found: {file_path}")
            return None
        
        if file_path.suffix.lower() not in self.supported_extensions:
            self.logger.warning(f"Unsupported file extension: {file_path.suffix}")
            return None
        
        try:
            # Try to detect encoding and load file
            content = self._load_with_encoding_detection(file_path)
            
            if not content:
                self.logger.warning(f"Empty or unreadable file: {file_path}")
                return None
            
            # Clean the content
            cleaned_content = self._clean_text(content)
            
            # Create metadata
            metadata = self._extract_metadata(file_path, cleaned_content)
            
            document = Document(
                content=cleaned_content,
                source=str(file_path),
                metadata=metadata
            )
            
            self.logger.debug(f"Loaded document: {file_path} ({document.word_count} words)")
            return document
            
        except Exception as e:
            self.logger.error(f"Failed to load file {file_path}: {e}")
            return None
    
    def load_directory(self, directory_path: Union[str, Path], pattern: str = "*.txt", recursive: bool = False) -> List[Document]:
        """
        Load all matching files from a directory.
        
        Args:
            directory_path: Path to the directory
            pattern: File pattern to match (e.g., "*.txt", "*.md")
            recursive: Whether to search subdirectories
            
        Returns:
            List of Document objects
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            self.logger.error(f"Directory not found: {directory_path}")
            return []
        
        # Find matching files
        if recursive:
            files = list(directory_path.rglob(pattern))
        else:
            files = list(directory_path.glob(pattern))
        
        # Filter by supported extensions
        files = [f for f in files if f.suffix.lower() in self.supported_extensions]
        
        if not files:
            self.logger.warning(f"No matching files found in {directory_path} with pattern '{pattern}'")
            return []
        
        # Load all files
        documents = []
        for file_path in sorted(files):  # Sort for consistent ordering
            document = self.load_file(file_path)
            if document:
                documents.append(document)
        
        self.logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents
    
    def load_documents(self, documents_dir: Union[str, Path], file_pattern: str = "*.txt") -> List[Dict[str, Any]]:
        """
        Load documents from a directory with a file pattern.
        
        Args:
            documents_dir: Directory path containing documents
            file_pattern: File pattern to match (e.g., "*.txt", "*.md")
            
        Returns:
            List of document dictionaries
        """
        documents = self.load_directory(documents_dir, pattern=file_pattern, recursive=False)
        return [doc.to_dict() for doc in documents]
    
    def load_documents_from_paths(self, paths: List[Union[str, Path]]) -> List[Document]:
        """
        Load documents from a list of file paths.
        
        Args:
            paths: List of file paths to load
            
        Returns:
            List of Document objects
        """
        documents = []
        for path in paths:
            document = self.load_file(path)
            if document:
                documents.append(document)
        
        return documents
    
    def _load_with_encoding_detection(self, file_path: Path) -> str:
        """Load file with automatic encoding detection."""
        # Common encodings to try
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    self.logger.debug(f"Successfully loaded {file_path} with encoding: {encoding}")
                    return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                self.logger.error(f"Error reading {file_path} with {encoding}: {e}")
                continue
        
        # If all encodings fail, try with error handling
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                self.logger.warning(f"Loaded {file_path} with character replacement due to encoding issues")
                return content
        except Exception as e:
            self.logger.error(f"Failed to load {file_path} with any encoding: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Apply cleanup patterns
        for pattern, replacement, *flags in self.cleanup_patterns:
            if flags:
                text = re.sub(pattern, replacement, text, flags=flags[0])
            else:
                text = re.sub(pattern, replacement, text)
        
        # Remove control characters except for common whitespace
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t\r ')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract metadata from file and content."""
        metadata = {
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'file_extension': file_path.suffix.lower(),
            'word_count': len(content.split()) if content else 0,
            'char_count': len(content) if content else 0,
            'line_count': content.count('\n') + 1 if content else 0,
        }
        
        # Try to extract creation/modification dates
        try:
            stat = file_path.stat()
            metadata['created_at'] = stat.st_ctime
            metadata['modified_at'] = stat.st_mtime
        except Exception:
            pass
        
        # Extract title from filename or content
        title = self._extract_title(file_path, content)
        if title:
            metadata['title'] = title
        
        return metadata
    
    def _extract_title(self, file_path: Path, content: str) -> str:
        """Extract a title from filename or content."""
        # Try to get title from filename
        title = file_path.stem
        
        # Clean up filename-based title
        title = re.sub(r'[_-]+', ' ', title)  # Replace underscores/dashes with spaces
        title = re.sub(r'\d{4}\d{2}\d{2}', '', title)  # Remove dates
        title = re.sub(r'\s+', ' ', title).strip()  # Normalize spaces
        
        # If content is available, try to extract a better title
        if content:
            lines = content.split('\n')[:10]  # Check first 10 lines
            for line in lines:
                line = line.strip()
                # Look for markdown headers
                if line.startswith('#'):
                    potential_title = line.lstrip('#').strip()
                    if len(potential_title) > 5 and len(potential_title) < 100:
                        title = potential_title
                        break
                # Look for lines that might be titles (all caps, short, etc.)
                elif len(line) > 5 and len(line) < 100 and not line.endswith('.'):
                    # Check if it looks like a title
                    if line.isupper() or (line.count(' ') < 10 and line[0].isupper()):
                        title = line
                        break
        
        return title
    
    def get_corpus_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about a document corpus."""
        if not documents:
            return {
                'total_documents': 0,
                'total_words': 0,
                'total_characters': 0,
                'average_words_per_document': 0,
                'average_characters_per_document': 0,
                'file_types': {},
                'largest_document': None,
                'smallest_document': None
            }
        
        total_words = sum(doc.word_count for doc in documents)
        total_chars = sum(doc.char_count for doc in documents)
        
        # Count file types
        file_types = {}
        for doc in documents:
            ext = doc.metadata.get('file_extension', 'unknown')
            file_types[ext] = file_types.get(ext, 0) + 1
        
        # Find largest and smallest documents
        largest_doc = max(documents, key=lambda d: d.word_count)
        smallest_doc = min(documents, key=lambda d: d.word_count)
        
        return {
            'total_documents': len(documents),
            'total_words': total_words,
            'total_characters': total_chars,
            'average_words_per_document': total_words / len(documents),
            'average_characters_per_document': total_chars / len(documents),
            'file_types': file_types,
            'largest_document': {
                'source': largest_doc.source,
                'word_count': largest_doc.word_count,
                'title': largest_doc.metadata.get('title', 'Unknown')
            },
            'smallest_document': {
                'source': smallest_doc.source,
                'word_count': smallest_doc.word_count,
                'title': smallest_doc.metadata.get('title', 'Unknown')
            }
        }
    
    def split_document(self, document: Document, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
        """
        Split a document into smaller chunks.
        
        Args:
            document: Document to split
            chunk_size: Target number of words per chunk
            chunk_overlap: Number of words to overlap between chunks
            
        Returns:
            List of Document chunks
        """
        if not document.content:
            return []
        
        words = document.content.split()
        if len(words) <= chunk_size:
            return [document]  # Return original if it's already small enough
        
        chunks = []
        start = 0
        chunk_num = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_content = ' '.join(chunk_words)
            
            # Create metadata for chunk
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                'chunk_number': chunk_num,
                'chunk_start_word': start,
                'chunk_end_word': end,
                'total_chunks': None,  # Will be set after all chunks are created
                'is_chunk': True,
                'parent_source': document.source
            })
            
            chunk = Document(
                content=chunk_content,
                source=f"{document.source}#chunk_{chunk_num}",
                metadata=chunk_metadata
            )
            
            chunks.append(chunk)
            
            # Move start position for next chunk (with overlap)
            start = max(start + chunk_size - chunk_overlap, start + 1)
            chunk_num += 1
        
        # Update total_chunks in all chunk metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        self.logger.debug(f"Split document {document.source} into {len(chunks)} chunks")
        return chunks
    
    def filter_documents(self, documents: List[Document], min_words: int = 10, max_words: Optional[int] = None) -> List[Document]:
        """
        Filter documents by word count criteria.
        
        Args:
            documents: List of documents to filter
            min_words: Minimum word count
            max_words: Maximum word count (no limit if None)
            
        Returns:
            Filtered list of documents
        """
        filtered = []
        for doc in documents:
            if doc.word_count < min_words:
                continue
            if max_words is not None and doc.word_count > max_words:
                continue
            filtered.append(doc)
        
        if len(filtered) != len(documents):
            self.logger.info(f"Filtered {len(documents)} documents to {len(filtered)} documents based on word count criteria")
        
        return filtered
    
    def deduplicate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate documents based on content similarity.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Deduplicated list of documents
        """
        if not documents:
            return []
        
        unique_docs = []
        seen_content = set()
        
        for doc in documents:
            content = doc.get('content', '')
            if not content:
                continue
                
            # Use first 200 characters as fingerprint for deduplication
            fingerprint = content[:200].strip().lower()
            
            if fingerprint not in seen_content:
                seen_content.add(fingerprint)
                unique_docs.append(doc)
            else:
                self.logger.debug(f"Skipping duplicate document: {doc.get('source', 'unknown')}")
        
        if len(unique_docs) != len(documents):
            self.logger.info(f"Deduplicated {len(documents)} documents to {len(unique_docs)} unique documents")
        
        return unique_docs
    
    def get_document_summary(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get a summary of the document collection.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Summary statistics about the documents
        """
        if not documents:
            return {
                'total_documents': 0,
                'total_words': 0,
                'total_characters': 0,
                'average_words_per_document': 0,
                'largest_document_words': 0,
                'smallest_document_words': 0
            }
        
        word_counts = []
        total_chars = 0
        
        for doc in documents:
            content = doc.get('content', '')
            word_count = len(content.split()) if content else 0
            word_counts.append(word_count)
            total_chars += len(content)
        
        total_words = sum(word_counts)
        
        return {
            'total_documents': len(documents),
            'total_words': total_words,
            'total_characters': total_chars,
            'average_words_per_document': total_words / len(documents) if documents else 0,
            'largest_document_words': max(word_counts) if word_counts else 0,
            'smallest_document_words': min(word_counts) if word_counts else 0
        }