#!/usr/bin/env python3
"""
Simple PDF to Markdown converter for LLM analysis.
Converts text-selectable PDFs to clean markdown format.
"""

import argparse
import pathlib
import sys
from typing import Optional

try:
    import pymupdf4llm
except ImportError:
    print("Error: pymupdf4llm not installed. Run 'poetry install' first.")
    sys.exit(1)


def convert_pdf_to_markdown(
    input_path: str,
    output_path: Optional[str] = None,
    pages: Optional[list] = None,
    extract_images: bool = False,
    image_dir: str = "images"
) -> str:
    """
    Convert PDF to markdown format.
    
    Args:
        input_path: Path to input PDF file
        output_path: Path to output markdown file (optional)
        pages: List of page numbers to process (0-based, optional)
        extract_images: Whether to extract images
        image_dir: Directory to save extracted images
    
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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown for LLM analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf                    # Print to stdout
  %(prog)s document.pdf -o output.md       # Save to file
  %(prog)s document.pdf -p 0,2-4           # Convert pages 0, 2, 3, 4
  %(prog)s document.pdf -i                 # Extract images too
        """
    )
    
    parser.add_argument(
        'input_pdf',
        help='Input PDF file path'
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
    
    args = parser.parse_args()
    
    # Validate input file
    if not pathlib.Path(args.input_pdf).exists():
        print(f"Error: File '{args.input_pdf}' does not exist.")
        sys.exit(1)
    
    # Parse page range if provided
    pages = None
    if args.pages:
        try:
            pages = parse_page_range(args.pages)
        except ValueError as e:
            print(f"Error parsing page range: {e}")
            sys.exit(1)
    
    # Convert PDF
    markdown_text = convert_pdf_to_markdown(
        args.input_pdf,
        args.output,
        pages,
        args.images,
        args.image_dir
    )
    
    # Print to stdout if no output file specified
    if not args.output:
        print(markdown_text)


if __name__ == "__main__":
    main()