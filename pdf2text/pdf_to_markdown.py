#!/usr/bin/env python3
"""
Simple PDF to Markdown converter for LLM analysis.
Converts text-selectable PDFs to clean markdown format.
Supports both single file and batch processing.
"""

import argparse
import pathlib
import re
import sys
from typing import Optional, Tuple

try:
    import pymupdf4llm
except ImportError:
    print("Error: pymupdf4llm not installed. Run 'poetry install' first.")
    sys.exit(1)


# List of unwanted phrases to remove from markdown output
# Add new phrases here as needed
UNWANTED_PHRASES = [
    'OceanofPDF.com',
    # Add more unwanted phrases here in the future
]


def clean_markdown_text(text: str, clean_text: bool = True) -> str:
    """
    Clean unwanted phrases from markdown text.
    
    Args:
        text: The markdown text to clean
        clean_text: Whether to perform cleaning (default: True)
    
    Returns:
        Cleaned markdown text
    """
    if not clean_text or not UNWANTED_PHRASES:
        return text
    
    cleaned_text = text
    
    for phrase in UNWANTED_PHRASES:
        # Remove phrase with case-insensitive matching
        # Handle both standalone and inline occurrences
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        cleaned_text = pattern.sub('', cleaned_text)
    
    # Clean up extra whitespace that might be left behind
    # Remove multiple consecutive newlines (more than 2)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    # Remove trailing whitespace from lines
    cleaned_text = re.sub(r'[ \t]+$', '', cleaned_text, flags=re.MULTILINE)
    
    # Clean up multiple consecutive spaces
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
    
    return cleaned_text.strip()


def convert_pdf_to_markdown(
    input_path: str,
    output_path: Optional[str] = None,
    pages: Optional[list] = None,
    extract_images: bool = False,
    image_dir: str = "images",
    clean_text: bool = True
) -> str:
    """
    Convert PDF to markdown format.
    
    Args:
        input_path: Path to input PDF file
        output_path: Path to output markdown file (optional)
        pages: List of page numbers to process (0-based, optional)
        extract_images: Whether to extract images
        image_dir: Directory to save extracted images
        clean_text: Whether to clean unwanted phrases from output (default: True)
    
    Returns:
        Markdown text content
    """
    try:
        # Convert PDF to markdown
        kwargs = {}
        if pages:
            kwargs['pages'] = pages
        if extract_images:
            kwargs['write_images'] = True
            kwargs['image_path'] = image_dir
            kwargs['image_format'] = 'png'
            kwargs['dpi'] = 150
        
        markdown_text = pymupdf4llm.to_markdown(input_path, **kwargs)
        
        # Clean unwanted phrases from the text
        markdown_text = clean_markdown_text(markdown_text, clean_text)
        
        # Save to file if output path provided
        if output_path:
            pathlib.Path(output_path).write_bytes(markdown_text.encode('utf-8'))
            print(f"✓ Converted '{input_path}' to '{output_path}'")
        
        return markdown_text
        
    except FileNotFoundError:
        print(f"Error: PDF file '{input_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error converting PDF: {e}")
        sys.exit(1)


def parse_page_range(page_str: str) -> list:
    """Parse page range string like '0,2-5,7' into list of page numbers."""
    pages = []
    for part in page_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages))


def batch_convert_pdfs(
    input_dir: str,
    output_dir: str,
    pages: Optional[list] = None,
    extract_images: bool = False,
    image_dir: str = "images",
    clean_text: bool = True
) -> Tuple[int, int, int]:
    """
    Batch convert all PDF files in a directory to markdown.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save markdown files
        pages: List of page numbers to process (0-based, optional)
        extract_images: Whether to extract images
        image_dir: Directory to save extracted images
        clean_text: Whether to clean unwanted phrases from output (default: True)
    
    Returns:
        Tuple of (total_found, converted, skipped)
    """
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)
    
    # Validate input directory
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
    
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files (excluding hidden files)
    pdf_files = sorted(input_path.glob('*.pdf')) + sorted(input_path.glob('*.PDF'))
    pdf_files = list(set(pdf_files))  # Remove duplicates
    
    # Filter out hidden files (files starting with '.')
    pdf_files = [f for f in pdf_files if not f.name.startswith('.')]
    pdf_files.sort()
    
    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'")
        return 0, 0, 0
    
    print(f"Found {len(pdf_files)} PDF file(s) in '{input_dir}'")
    print(f"Output directory: '{output_dir}'")
    print("-" * 50)
    
    total_found = len(pdf_files)
    converted = 0
    skipped = 0
    failed = 0
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        # Generate output filename (same as input but with .md extension)
        output_filename = pdf_file.stem + '.md'
        output_file = output_path / output_filename
        
        # Check if output file already exists
        if output_file.exists():
            print(f"[{idx}/{total_found}] ⏭️  Skipping: {pdf_file.name} ('{output_filename}' already exists)")
            skipped += 1
            continue
        
        # Convert PDF to markdown
        print(f"[{idx}/{total_found}] ✓ Converting: {pdf_file.name} → {output_filename}")
        
        try:
            # Convert PDF to markdown
            kwargs = {}
            if pages:
                kwargs['pages'] = pages
            if extract_images:
                kwargs['write_images'] = True
                kwargs['image_path'] = image_dir
                kwargs['image_format'] = 'png'
                kwargs['dpi'] = 150
            
            markdown_text = pymupdf4llm.to_markdown(str(pdf_file), **kwargs)
            
            # Clean unwanted phrases from the text
            markdown_text = clean_markdown_text(markdown_text, clean_text)
            
            # Save to file
            output_file.write_bytes(markdown_text.encode('utf-8'))
            converted += 1
            
        except Exception as e:
            print(f"   ❌ Error converting '{pdf_file.name}': {e}")
            failed += 1
    
    # Print summary
    print("-" * 50)
    print("Batch Processing Summary:")
    print(f"  Total PDFs found:    {total_found}")
    print(f"  Successfully converted: {converted}")
    print(f"  Skipped (existing):    {skipped}")
    if failed > 0:
        print(f"  Failed:                {failed}")
    
    return total_found, converted, skipped


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown for LLM analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single file mode:
    %(prog)s document.pdf                    # Print to stdout
    %(prog)s document.pdf -o output.md       # Save to file
    %(prog)s document.pdf -p 0,2-4           # Convert pages 0, 2, 3, 4
    %(prog)s document.pdf -i                 # Extract images too

  Batch mode:
    %(prog)s --batch --input-dir pdf_folder --output-dir md_folder
    %(prog)s -b --input-dir pdfs --output-dir markdown
        """
    )
    
    # Batch mode arguments
    parser.add_argument(
        '-b', '--batch',
        action='store_true',
        help='Enable batch processing mode'
    )
    
    parser.add_argument(
        '--input-dir',
        help='Input directory containing PDF files (batch mode only)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory for markdown files (batch mode only)'
    )
    
    # Single file mode arguments
    parser.add_argument(
        'input_pdf',
        nargs='?',
        help='Input PDF file path (single file mode only)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output markdown file path (default: print to stdout)'
    )
    
    parser.add_argument(
        '-p', '--pages',
        help='Pages to convert (e.g., "0,2-4,7" for pages 0, 2, 3, 4, 7)'
    )
    
    parser.add_argument(
        '-i', '--images',
        action='store_true',
        help='Extract images from PDF'
    )
    
    parser.add_argument(
        '--image-dir',
        default='images',
        help='Directory to save extracted images (default: images)'
    )
    
    parser.add_argument(
        '--no-clean',
        action='store_true',
        help='Skip post-processing text cleaning (keep unwanted phrases)'
    )
    
    args = parser.parse_args()
    
    # Parse page range if provided
    pages = None
    if args.pages:
        try:
            pages = parse_page_range(args.pages)
        except ValueError as e:
            print(f"Error parsing page range: {e}")
            sys.exit(1)
    
    # Determine if text cleaning should be performed
    clean_text = not args.no_clean
    
    # Handle batch mode
    if args.batch:
        if not args.input_dir:
            print("Error: --input-dir is required in batch mode")
            sys.exit(1)
        if not args.output_dir:
            print("Error: --output-dir is required in batch mode")
            sys.exit(1)
        
        # Run batch conversion
        batch_convert_pdfs(
            args.input_dir,
            args.output_dir,
            pages,
            args.images,
            args.image_dir,
            clean_text
        )
        return
    
    # Handle single file mode
    if not args.input_pdf:
        print("Error: input_pdf is required in single file mode")
        print("Use --batch for batch processing or provide a PDF file path")
        sys.exit(1)
    
    # Validate input file
    if not pathlib.Path(args.input_pdf).exists():
        print(f"Error: File '{args.input_pdf}' does not exist.")
        sys.exit(1)
    
    # Convert PDF
    markdown_text = convert_pdf_to_markdown(
        args.input_pdf,
        args.output,
        pages,
        args.images,
        args.image_dir,
        clean_text
    )
    
    # Print to stdout if no output file specified
    if not args.output:
        print(markdown_text)


if __name__ == "__main__":
    main()