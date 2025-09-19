#!/usr/bin/env python3
"""
Standalone text formatter for transcripts.
Applies semantic paragraph formatting to existing text files.
"""

import argparse
import sys
from pathlib import Path

# Import the formatting function from the main audio script
try:
    from audio_to_text import format_transcript_semantic, get_device
except ImportError as e:
    print(f"Error: Could not import formatting functions: {e}")
    sys.exit(1)


def main():
    """Main entry point for standalone text formatting."""
    parser = argparse.ArgumentParser(
        description="Format text files with semantic paragraph breaks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Format a single transcript file
  %(prog)s transcript.txt
  %(prog)s transcript.txt -o formatted.txt
  
  # Batch format all txt files in a directory
  %(prog)s --batch --input-dir ./transcripts --output-dir ./formatted
  
  # Use CPU instead of GPU
  %(prog)s transcript.txt --device cpu
        """
    )
    
    # Positional argument for single file mode
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Path to text file for single file mode"
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        help="Output file path (single file mode). If not specified, overwrites input file."
    )
    
    # Batch processing
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch processing mode"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory containing text files (batch mode)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for formatted files (batch mode)"
    )
    
    # Processing options
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Device to use for semantic processing (default: auto)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimize output messages"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = get_device()
    else:
        device = args.device
    
    if args.batch:
        # Batch processing mode
        if not args.input_dir or not args.output_dir:
            parser.error("Batch mode requires --input-dir and --output-dir")
        
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            sys.exit(1)
        
        # Find all text files
        text_files = list(input_dir.glob("*.txt"))
        if not text_files:
            print(f"No .txt files found in {input_dir}")
            return
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Formatting {len(text_files)} text file(s)...")
        print(f"Input:  {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Device: {device.upper()}")
        print()
        
        # Process each file
        for i, text_file in enumerate(text_files, 1):
            output_file = output_dir / text_file.name
            
            print(f"[{i}/{len(text_files)}] {text_file.name}")
            
            try:
                # Read input file
                with open(text_file, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                
                # Apply formatting
                formatted_text = format_transcript_semantic(
                    raw_text, 
                    device=device, 
                    verbose=not args.quiet
                )
                
                # Write output file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(formatted_text)
                
                print(f"  ✓ Saved to: {output_file.name}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
            
            print()
        
        print("Batch formatting complete!")
    
    else:
        # Single file mode
        if not args.input_file:
            parser.error("Input file required for single file mode (or use --batch)")
        
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)
        
        output_path = Path(args.output) if args.output else input_path
        
        if not args.quiet:
            print(f"Formatting: {input_path.name}")
            print(f"Device: {device.upper()}")
        
        try:
            # Read input file
            with open(input_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            if not args.quiet:
                print(f"Original text: {len(raw_text)} characters")
            
            # Apply formatting
            formatted_text = format_transcript_semantic(
                raw_text, 
                device=device, 
                verbose=not args.quiet
            )
            
            # Write output file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            
            if not args.quiet:
                print(f"✓ Formatted text saved to: {output_path}")
                print(f"Formatted text: {len(formatted_text)} characters")
        
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()