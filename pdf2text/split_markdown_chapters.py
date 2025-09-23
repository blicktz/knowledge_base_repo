#!/usr/bin/env python3
"""
Enhanced Markdown Chapter Splitter for Optimal Embedding Chunks

This script splits markdown books into optimally-sized chunks for embeddings:
1. First pass: Split by chapter patterns (various formats)
2. Second pass: Optimize chunk sizes by merging small chunks or splitting large ones

Designed specifically for Dan S. Kennedy books with their unique formatting patterns.
"""

import os
import re
import argparse
from typing import List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    content: str
    title: str
    word_count: int
    original_chapter: str = ""
    chunk_number: int = 0


class MarkdownChapterSplitter:
    def __init__(self, 
                 target_words: int = 2000, 
                 min_words: int = 1000, 
                 max_words: int = 3000,
                 verbose: bool = False):
        self.target_words = target_words
        self.min_words = min_words
        self.max_words = max_words
        self.verbose = verbose
        
        # Comprehensive chapter patterns for Dan S. Kennedy books
        self.chapter_patterns = [
            # Traditional format: "C H A P T E R 1", "C H A P T E R 2"
            r'(^\s*C\s+H\s+A\s+P\s+T\s+E\s+R\s+\*?\*?[\dIVXLCDM]+\*?\*?.*$)',
            
            # Word format: "Chapter One", "Chapter Two"
            r'(^\s*Chapter\s+(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty).*$)',
            
            # Standard format: "Chapter 1", "CHAPTER 2"
            r'(^\s*[Cc][Hh][Aa][Pp][Tt][Ee][Rr]\s+[\dIVXLCDM]+.*$)',
            
            # Success factors: "ULTIMATE MARKETING PLAN SUCCESS: FACTOR #1"
            r'(^\s*ULTIMATE\s+MARKETING\s+PLAN\s+SUCCESS.*FACTOR\s*#?[\dIVXLCDM]+.*$)',
            
            # Bonus chapters: "Bonus Chapter #1", "BONUS CHAPTER #2"
            r'(^\s*[Bb][Oo][Nn][Uu][Ss]\s+[Cc][Hh][Aa][Pp][Tt][Ee][Rr]\s*#?[\dIVXLCDM]+.*$)',
            
            # Appendix: "Appendix A", "APPENDIX B"
            r'(^\s*[Aa][Pp][Pp][Ee][Nn][Dd][Ii][Xx]\s+[A-Z\d]+.*$)',
            
            # Markdown headers for chapters
            r'(^#{1,3}\s+[Cc][Hh][Aa][Pp][Tt][Ee][Rr].*$)',
            
            # Section headers that might be chapter-like
            r'(^#{1,2}\s+[A-Z][A-Z\s]{10,}$)',  # All caps headers
        ]
        
        # Inline chapter patterns (found in table of contents)
        self.inline_chapter_pattern = r'CHAPTER\s+\d+:\s*[^C]*?(?=CHAPTER\s+\d+:|$)'
    
    def log(self, message: str):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def count_words(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())
    
    def extract_inline_chapters(self, content: str) -> List[Tuple[str, str]]:
        """Extract chapters from inline format like 'CHAPTER 1: Title CHAPTER 2: Title'"""
        chapters = []
        matches = re.findall(self.inline_chapter_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for i, match in enumerate(matches, 1):
            title = f"Chapter {i}"
            # Extract title from chapter content if possible
            title_match = re.match(r'CHAPTER\s+\d+:\s*([^.\n]+)', match.strip(), re.IGNORECASE)
            if title_match:
                title = f"Chapter {i}: {title_match.group(1).strip()}"
            
            chapters.append((title, match.strip()))
        
        return chapters
    
    def split_by_patterns(self, content: str, book_name: str) -> List[Chunk]:
        """First pass: Split content by chapter patterns"""
        chunks = []
        
        # Check for inline chapter format first
        inline_chapters = self.extract_inline_chapters(content)
        if len(inline_chapters) > 3:  # If we found multiple inline chapters
            self.log(f"Found {len(inline_chapters)} inline chapters")
            for title, chapter_content in inline_chapters:
                word_count = self.count_words(chapter_content)
                chunks.append(Chunk(
                    content=chapter_content,
                    title=title,
                    word_count=word_count,
                    original_chapter=title
                ))
            return chunks
        
        # Try regular pattern matching
        for pattern in self.chapter_patterns:
            parts = re.split(pattern, content, flags=re.IGNORECASE | re.MULTILINE)
            
            if len(parts) > 3:  # Found meaningful splits
                self.log(f"Using pattern: {pattern}")
                self.log(f"Found {len(parts)} parts")
                
                # Process the splits
                # Skip the first part if it's empty/header content
                start_idx = 1 if parts[0].strip() else 1
                
                for i in range(start_idx, len(parts), 2):
                    if i < len(parts):
                        heading = parts[i].strip()
                        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
                        
                        # Skip if heading is empty or body is too short
                        if not heading:
                            continue
                            
                        # Clean up the heading for title
                        title = re.sub(r'[*#\s]+', ' ', heading).strip()
                        title = re.sub(r'\s+', ' ', title)
                        
                        # Combine heading and body
                        if body:
                            chapter_content = heading + "\n\n" + body
                        else:
                            chapter_content = heading
                            
                        word_count = self.count_words(chapter_content)
                        
                        # Only add if there's substantial content
                        if word_count > 10:
                            chunks.append(Chunk(
                                content=chapter_content,
                                title=title,
                                word_count=word_count,
                                original_chapter=title
                            ))
                
                return chunks
        
        # If no patterns match, treat as single chunk
        self.log("No chapter patterns found, treating as single document")
        word_count = self.count_words(content)
        chunks.append(Chunk(
            content=content,
            title=book_name,
            word_count=word_count,
            original_chapter=book_name
        ))
        
        return chunks
    
    def split_large_chunk(self, chunk: Chunk) -> List[Chunk]:
        """Split a large chunk at paragraph boundaries"""
        paragraphs = chunk.content.split('\n\n')
        new_chunks = []
        current_content = ""
        current_words = 0
        part_number = 1
        
        for paragraph in paragraphs:
            para_words = self.count_words(paragraph)
            
            # If adding this paragraph would exceed max_words, create a new chunk
            if current_words + para_words > self.max_words and current_content:
                title = f"{chunk.title} (Part {part_number})"
                new_chunks.append(Chunk(
                    content=current_content.strip(),
                    title=title,
                    word_count=current_words,
                    original_chapter=chunk.original_chapter
                ))
                
                current_content = paragraph + "\n\n"
                current_words = para_words
                part_number += 1
            else:
                current_content += paragraph + "\n\n"
                current_words += para_words
        
        # Add the remaining content
        if current_content.strip():
            title = f"{chunk.title} (Part {part_number})" if part_number > 1 else chunk.title
            new_chunks.append(Chunk(
                content=current_content.strip(),
                title=title,
                word_count=current_words,
                original_chapter=chunk.original_chapter
            ))
        
        return new_chunks
    
    def optimize_chunk_sizes(self, chunks: List[Chunk]) -> List[Chunk]:
        """Second pass: Optimize chunk sizes for embeddings"""
        optimized_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # If chunk is too large, split it
            if current_chunk.word_count > self.max_words:
                self.log(f"Splitting large chunk: {current_chunk.title} ({current_chunk.word_count} words)")
                split_chunks = self.split_large_chunk(current_chunk)
                optimized_chunks.extend(split_chunks)
                i += 1
                continue
            
            # If chunk is too small, try to merge with next chunk
            if current_chunk.word_count < self.min_words and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                combined_words = current_chunk.word_count + next_chunk.word_count
                
                # Only merge if combined size is reasonable
                if combined_words <= self.max_words:
                    self.log(f"Merging small chunks: {current_chunk.title} + {next_chunk.title}")
                    merged_content = current_chunk.content + "\n\n" + next_chunk.content
                    merged_title = f"{current_chunk.title} & {next_chunk.title}"
                    
                    optimized_chunks.append(Chunk(
                        content=merged_content,
                        title=merged_title,
                        word_count=combined_words,
                        original_chapter=f"{current_chunk.original_chapter} & {next_chunk.original_chapter}"
                    ))
                    i += 2  # Skip both chunks
                    continue
            
            # Keep chunk as-is
            optimized_chunks.append(current_chunk)
            i += 1
        
        return optimized_chunks
    
    def generate_filename(self, book_name: str, chunk_number: int) -> str:
        """Generate a clean filename for the chunk"""
        return f"{book_name}_chunk_{chunk_number:03d}.md"
    
    def process_book(self, input_path: str, output_dir: str) -> Dict:
        """Process a single book file"""
        book_name = Path(input_path).stem
        self.log(f"\nProcessing: {book_name}")
        
        # Read the book content
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.log(f"Error reading {input_path}: {e}")
            return {"error": str(e)}
        
        # First pass: Split by chapters
        initial_chunks = self.split_by_patterns(content, book_name)
        self.log(f"Initial split: {len(initial_chunks)} chunks")
        
        # Second pass: Optimize sizes
        optimized_chunks = self.optimize_chunk_sizes(initial_chunks)
        self.log(f"After optimization: {len(optimized_chunks)} chunks")
        
        # Write chunks to files
        stats = {
            "book_name": book_name,
            "total_chunks": len(optimized_chunks),
            "word_counts": [],
            "files_created": []
        }
        
        for i, chunk in enumerate(optimized_chunks, 1):
            chunk.chunk_number = i
            filename = self.generate_filename(book_name, i)
            output_path = os.path.join(output_dir, filename)
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(chunk.content)
                
                stats["word_counts"].append(chunk.word_count)
                stats["files_created"].append(filename)
                self.log(f"  -> Created: {filename} ({chunk.word_count} words)")
                
            except Exception as e:
                self.log(f"Error writing {filename}: {e}")
        
        return stats
    
    def print_statistics(self, all_stats: List[Dict]):
        """Print processing statistics"""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        
        total_chunks = 0
        all_word_counts = []
        
        for stats in all_stats:
            if "error" not in stats:
                print(f"\n{stats['book_name']}:")
                print(f"  Chunks created: {stats['total_chunks']}")
                word_counts = stats['word_counts']
                if word_counts:
                    print(f"  Word count range: {min(word_counts)} - {max(word_counts)}")
                    print(f"  Average words: {sum(word_counts)//len(word_counts)}")
                
                total_chunks += stats['total_chunks']
                all_word_counts.extend(word_counts)
        
        if all_word_counts:
            print(f"\nOVERALL STATISTICS:")
            print(f"  Total chunks: {total_chunks}")
            print(f"  Word count range: {min(all_word_counts)} - {max(all_word_counts)}")
            print(f"  Average words per chunk: {sum(all_word_counts)//len(all_word_counts)}")
            print(f"  Target range: {self.min_words} - {self.max_words} words")


def main():
    parser = argparse.ArgumentParser(description='Split markdown books into optimally-sized chunks for embeddings')
    parser.add_argument('input', help='Directory containing markdown book files OR single markdown file')
    parser.add_argument('-o', '--output', default='output_chapters', 
                       help='Output directory (default: output_chapters)')
    parser.add_argument('--target-words', type=int, default=2000,
                       help='Target words per chunk (default: 2000)')
    parser.add_argument('--min-words', type=int, default=1000,
                       help='Minimum words per chunk (default: 1000)')
    parser.add_argument('--max-words', type=int, default=3000,
                       help='Maximum words per chunk (default: 3000)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--preview', action='store_true',
                       help='Preview mode - show what would be done without creating files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input '{args.input}' does not exist")
        return 1
    
    # Create output directory
    if not args.preview:
        os.makedirs(args.output, exist_ok=True)
    
    # Initialize splitter
    splitter = MarkdownChapterSplitter(
        target_words=args.target_words,
        min_words=args.min_words,
        max_words=args.max_words,
        verbose=args.verbose
    )
    
    # Find all markdown files
    md_files = []
    if os.path.isfile(args.input):
        if args.input.endswith('.md'):
            md_files.append(args.input)
        else:
            print(f"Error: '{args.input}' is not a markdown file")
            return 1
    elif os.path.isdir(args.input):
        for file in os.listdir(args.input):
            if file.endswith('.md'):
                md_files.append(os.path.join(args.input, file))
    else:
        print(f"Error: '{args.input}' is neither a file nor directory")
        return 1
    
    if not md_files:
        print(f"No .md files found in '{args.input}'")
        return 1
    
    print(f"Found {len(md_files)} markdown files to process")
    if args.preview:
        print("PREVIEW MODE - No files will be created")
    
    # Process each book
    all_stats = []
    for md_file in sorted(md_files):
        if args.preview:
            # In preview mode, just analyze without creating files
            book_name = Path(md_file).stem
            print(f"\nWould process: {book_name}")
            # You could add preview logic here
        else:
            stats = splitter.process_book(md_file, args.output)
            all_stats.append(stats)
    
    if not args.preview and all_stats:
        splitter.print_statistics(all_stats)
    
    return 0


if __name__ == "__main__":
    exit(main())