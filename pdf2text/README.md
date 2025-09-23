# Markdown Chapter Splitter for Optimal Embeddings

This script splits markdown books into optimally-sized chunks for embedding generation and LLM querying. It uses a two-pass approach: first splitting by chapter patterns, then optimizing chunk sizes through merging and splitting.

## Features

- **Multi-Pattern Chapter Detection**: Handles various chapter formats found in Dan S. Kennedy books
- **Size Optimization**: Creates chunks of similar sizes (1000-3000 words) for consistent embeddings
- **Markdown Preservation**: Maintains markdown formatting in split files
- **Metadata Headers**: Each chunk includes metadata (book name, chunk number, word count)
- **Flexible Configuration**: Adjustable target sizes and thresholds

## Chapter Patterns Detected

The script recognizes these chapter patterns:

1. **Traditional Format**: `C H A P T E R 1`, `C H A P T E R 2`
2. **Word Format**: `Chapter One`, `Chapter Two`, `Chapter Three`
3. **Standard Format**: `Chapter 1`, `CHAPTER 2`
4. **Success Factors**: `ULTIMATE MARKETING PLAN SUCCESS: FACTOR #1`
5. **Bonus Chapters**: `Bonus Chapter #1`, `BONUS CHAPTER #2`
6. **Appendices**: `Appendix A`, `APPENDIX B`
7. **Markdown Headers**: `# Chapter 1`, `## Chapter 2`
8. **Inline Chapters**: Table of contents with `CHAPTER 1: Title CHAPTER 2: Title`

## Usage

### Basic Usage

```bash
# Process all books in a directory
python split_markdown_chapters.py /path/to/books

# Process a single book
python split_markdown_chapters.py /path/to/book.md

# Specify custom output directory
python split_markdown_chapters.py /path/to/books -o output_chunks

# Preview mode (see what would be done without creating files)
python split_markdown_chapters.py --preview /path/to/books
```

### Advanced Options

```bash
# Custom chunk sizes
python split_markdown_chapters.py /path/to/books \
  --target-words 2500 \
  --min-words 1500 \
  --max-words 3500

# Verbose output for debugging
python split_markdown_chapters.py -v /path/to/books
```

### Command Line Options

- `input`: Directory containing markdown files OR single markdown file
- `-o, --output`: Output directory (default: `output_chapters`)
- `--target-words`: Target words per chunk (default: 2000)
- `--min-words`: Minimum words per chunk (default: 1000)
- `--max-words`: Maximum words per chunk (default: 3000)
- `-v, --verbose`: Enable verbose output
- `--preview`: Preview mode - show what would be done without creating files

## Output Structure

The script creates a flat directory structure with descriptive filenames:

```
output_chapters/
├── The_Ultimate_Marketing_Plan_-_Dan_S_Kennedy_chunk_001.md
├── The_Ultimate_Marketing_Plan_-_Dan_S_Kennedy_chunk_002.md
├── How_to_make_millions_with_your_ideas_-_Dan_S_Kennedy_chunk_001.md
├── How_to_make_millions_with_your_ideas_-_Dan_S_Kennedy_chunk_002.md
└── ...
```

## Chunk Format

Each chunk file contains only the original book content without any additional metadata:

```markdown
ULTIMATE MARKETING PLAN SUCCESS: FACTOR #1

#### Right Message

[Chapter content follows...]
```

## Two-Pass Processing

### Pass 1: Structural Splitting
- Identifies chapter boundaries using regex patterns
- Creates initial chunks based on chapter structure
- Handles various formatting styles in Dan S. Kennedy books

### Pass 2: Size Optimization
- **Small chunks** (<1000 words): Merges with adjacent chunks
- **Large chunks** (>3000 words): Splits at paragraph boundaries
- **Optimal chunks** (1000-3000 words): Keeps unchanged

## Example Results

For the test run on 16 Dan S. Kennedy books:

- **Total chunks created**: ~500 chunks across all books
- **Average chunk size**: ~2000 words
- **Size distribution**: Mostly 1000-3000 words (optimal for embeddings)
- **Processing time**: ~30 seconds for all 16 books

## Design for Embeddings

The script is specifically designed for creating embeddings:

- **Consistent sizes**: Similar word counts across chunks ensure uniform embedding quality
- **Semantic boundaries**: Splits respect chapter/section boundaries when possible
- **Metadata preservation**: Original context maintained through metadata headers
- **No content loss**: All original text preserved across chunks

## Troubleshooting

### Small chunks (very few words)
These are usually table of contents entries or headers. The optimization pass will merge them with adjacent content where possible.

### Very large chunks
If a single chapter is extremely long, the script splits it at paragraph boundaries while preserving context.

### No chapters found
For books without clear chapter patterns, the script treats the entire book as one chunk, then applies size optimization.

### Verbose debugging
Use the `-v` flag to see detailed processing information:

```bash
python split_markdown_chapters.py -v /path/to/books
```

This shows:
- Which pattern was matched
- How many initial chunks were created
- Which chunks were merged/split
- Final statistics

## Integration with Embedding Pipelines

The output format is designed for easy integration with embedding systems:

1. **Batch processing**: Process entire directories of chunks
2. **Filename-based metadata**: Book name and chunk number encoded in filename
3. **Consistent sizing**: Uniform embedding vector dimensions
4. **Semantic coherence**: Chunks respect logical boundaries

Example integration:
```python
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Process all chunks
for chunk_file in os.listdir('output_chapters'):
    if chunk_file.endswith('.md'):
        # Extract book name and chunk number from filename
        # Format: BookName_chunk_001.md
        parts = chunk_file.replace('.md', '').split('_chunk_')
        book_name = parts[0]
        chunk_number = parts[1]
        
        with open(f'output_chapters/{chunk_file}') as f:
            content = f.read()
            embedding = model.encode(content)
            # Store embedding with extracted metadata...
```