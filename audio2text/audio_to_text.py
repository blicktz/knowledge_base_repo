#!/usr/bin/env python3
"""
Audio to Text converter using OpenAI's Whisper model.
Converts MP3 files to text transcripts for LLM analysis.
Supports both single file and batch processing.
"""

import argparse
import os
import sys
import time
import re
from pathlib import Path
from typing import Optional, List
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

try:
    import whisper
    import torch
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError as e:
    print(f"Error: Required module not installed: {e}")
    print("Run 'poetry install' to install dependencies.")
    sys.exit(1)

# Default model configuration
DEFAULT_MODEL = "large-v3"
SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo", "large-v3-turbo"]


def get_device():
    """Determine the best available device for inference."""
    if torch.backends.mps.is_available():
        return "mps"  # Mac Metal Performance Shaders
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_transcript_semantic(text: str, device: str = "mps", verbose: bool = False) -> str:
    """
    Format transcript with semantic paragraph breaks using sentence transformers.
    
    Args:
        text: Raw transcript text
        device: Device for sentence transformer model ("mps", "cuda", "cpu")
        verbose: Whether to print progress information
    
    Returns:
        Formatted text with semantic paragraph breaks
    """
    if verbose:
        print("  Formatting transcript with semantic analysis...")
        start_time = time.time()
    
    try:
        # Clean up the text first
        text = text.strip()
        if not text:
            return text
        
        # Split into sentences using simple regex (works well for transcripts)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 3:
            # Too few sentences, just return as single paragraph
            return ' '.join(sentences)
        
        # Load lightweight sentence transformer model
        try:
            # Use a fast, lightweight model that works well on MPS
            model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        except Exception:
            # Fallback to CPU if MPS fails
            model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Generate embeddings for all sentences
        embeddings = model.encode(sentences, convert_to_tensor=True)
        
        # Calculate semantic similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = torch.cosine_similarity(
                embeddings[i].unsqueeze(0), 
                embeddings[i + 1].unsqueeze(0)
            ).item()
            similarities.append(similarity)
        
        # Find natural paragraph breaks (low similarity = topic change)
        # Use adaptive threshold based on similarity distribution
        if similarities:
            threshold = np.percentile(similarities, 10)  # Only bottom 10% are paragraph breaks
            threshold = max(threshold, 0.5)  # Lower threshold for more grouping
            threshold = min(threshold, 0.8)  # Cap to prevent too few paragraphs
        else:
            threshold = 0.6
        
        # Group sentences into paragraphs
        paragraphs = []
        current_paragraph = [sentences[0]]
        
        for i, similarity in enumerate(similarities):
            if similarity < threshold and len(current_paragraph) >= 3:
                # Start new paragraph only on low similarity AND minimum length
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [sentences[i + 1]]
            elif len(current_paragraph) >= 8:
                # Force new paragraph if too long
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [sentences[i + 1]]
            else:
                current_paragraph.append(sentences[i + 1])
        
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Join paragraphs with double line breaks (blank line between paragraphs)
        formatted_text = '\n\n'.join(paragraphs)
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"  Semantic formatting completed in {format_time(elapsed)}")
            print(f"  Created {len(paragraphs)} paragraphs from {len(sentences)} sentences")
        
        return formatted_text
        
    except Exception as e:
        if verbose:
            print(f"  Warning: Semantic formatting failed ({e}), using basic formatting")
        
        # Fallback to simple sentence grouping
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group every 3-4 sentences into paragraphs
        paragraphs = []
        for i in range(0, len(sentences), 3):
            paragraph = ' '.join(sentences[i:i+3]).strip()
            if paragraph:
                paragraphs.append(paragraph)
        
        return '\n\n'.join(paragraphs)


def transcribe_audio(
    audio_path: Path,
    output_path: Optional[Path] = None,
    model_name: str = DEFAULT_MODEL,
    model_instance: Optional[whisper.Whisper] = None,
    verbose: bool = True
) -> str:
    """
    Transcribe an audio file to text.
    
    Args:
        audio_path: Path to the audio file
        output_path: Optional path to save the transcript
        model_name: Whisper model to use
        model_instance: Pre-loaded model instance (for batch processing)
        verbose: Whether to print progress information
    
    Returns:
        The transcribed text
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load model if not provided
    if model_instance is None:
        if verbose:
            print(f"Loading Whisper model '{model_name}'...")
        device = get_device()
        model = whisper.load_model(model_name, device=device)
        if verbose:
            print(f"Model loaded on {device.upper()}")
    else:
        model = model_instance
    
    # Transcribe
    if verbose:
        print(f"Transcribing: {audio_path.name}")
        start_time = time.time()
    
    result = model.transcribe(str(audio_path), fp16=False, verbose=False)
    raw_text = result["text"].strip()
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"  Completed in {format_time(elapsed)}")
    
    # Format the transcript with semantic paragraph breaks
    device = get_device()
    formatted_text = format_transcript_semantic(raw_text, device=device, verbose=verbose)
    
    # Save to file if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        if verbose:
            print(f"  Saved to: {output_path.name}")
    
    return formatted_text


def batch_transcribe(
    input_dir: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    skip_existing: bool = True,
    verbose: bool = True
) -> dict:
    """
    Batch transcribe all MP3 files in a directory.
    
    Args:
        input_dir: Directory containing MP3 files
        output_dir: Directory to save transcript files
        model_name: Whisper model to use
        skip_existing: Skip files that already have transcripts
        verbose: Whether to print progress information
    
    Returns:
        Dictionary with processing statistics
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all MP3 files
    audio_files = sorted(input_dir.glob("*.mp3"))
    if not audio_files:
        print(f"No MP3 files found in {input_dir}")
        return {"total": 0, "processed": 0, "skipped": 0, "failed": 0}
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model once for batch processing
    print(f"Loading Whisper model '{model_name}'...")
    device = get_device()
    model = whisper.load_model(model_name, device=device)
    print(f"Model loaded on {device.upper()}")
    print(f"Found {len(audio_files)} MP3 file(s) to process\n")
    
    # Process statistics
    stats = {
        "total": len(audio_files),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "failed_files": []
    }
    
    # Process each file
    for i, audio_file in enumerate(audio_files, 1):
        # Generate output path with .txt extension
        output_file = output_dir / f"{audio_file.stem}.txt"
        
        print(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
        # Skip if already exists
        if skip_existing and output_file.exists():
            print(f"  Skipped (transcript exists)")
            stats["skipped"] += 1
            continue
        
        try:
            # Transcribe
            start_time = time.time()
            _ = transcribe_audio(
                audio_file,
                output_file,
                model_name,
                model,
                verbose=False
            )
            elapsed = time.time() - start_time
            
            print(f"  Success! Saved as: {output_file.name}")
            print(f"  Time: {format_time(elapsed)}")
            stats["processed"] += 1
            
        except Exception as e:
            print(f"  Failed: {e}")
            stats["failed"] += 1
            stats["failed_files"].append(str(audio_file))
        
        print()  # Empty line between files
    
    # Print summary
    print("=" * 50)
    print("Batch Processing Complete")
    print("=" * 50)
    print(f"Total files:      {stats['total']}")
    print(f"Processed:        {stats['processed']}")
    print(f"Skipped:          {stats['skipped']}")
    print(f"Failed:           {stats['failed']}")
    
    if stats["failed_files"]:
        print("\nFailed files:")
        for f in stats["failed_files"]:
            print(f"  - {f}")
    
    return stats


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert audio files to text using Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file transcription
  %(prog)s podcast.mp3
  %(prog)s podcast.mp3 -o transcript.txt
  %(prog)s podcast.mp3 --model large-v3
  
  # Batch processing
  %(prog)s --batch --input-dir /path/to/mp3s --output-dir /path/to/texts
  %(prog)s --batch --input-dir ./podcasts --output-dir ./transcripts --model medium
  
Available models: tiny, base, small, medium, large, large-v2, large-v3, turbo, large-v3-turbo
Note: Larger models are more accurate but slower. 'turbo' is fastest with near large-v3 quality.
        """
    )
    
    # Positional argument for single file mode
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to audio file (MP3) for single file mode"
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        help="Output file path for transcript (single file mode)"
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
        help="Input directory containing MP3 files (batch mode)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for transcript files (batch mode)"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=DEFAULT_MODEL,
        help=f"Whisper model to use (default: {DEFAULT_MODEL})"
    )
    
    # Processing options
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Process files even if transcripts already exist"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimize output messages"
    )
    
    # Utility options
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Test setup by loading model and exiting"
    )
    
    args = parser.parse_args()
    
    # Setup only mode
    if args.setup_only:
        print("Testing Whisper setup...")
        device = get_device()
        print(f"Device: {device.upper()}")
        print(f"Loading model '{args.model}'...")
        model = whisper.load_model(args.model, device=device)
        print("✓ Setup successful! Ready to transcribe audio.")
        return
    
    # Validate arguments
    if args.batch:
        if not args.input_dir or not args.output_dir:
            parser.error("Batch mode requires --input-dir and --output-dir")
        
        # Batch processing
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        
        try:
            batch_transcribe(
                input_dir,
                output_dir,
                model_name=args.model,
                skip_existing=not args.no_skip_existing,
                verbose=not args.quiet
            )
        except Exception as e:
            print(f"Error during batch processing: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        # Single file mode
        if not args.audio_file:
            parser.error("Audio file required for single file mode (or use --batch)")
        
        audio_path = Path(args.audio_file)
        output_path = Path(args.output) if args.output else None
        
        try:
            text = transcribe_audio(
                audio_path,
                output_path,
                model_name=args.model,
                verbose=not args.quiet
            )
            
            # Print to stdout if no output file specified
            if not output_path and not args.quiet:
                print("\n" + "=" * 50)
                print("TRANSCRIPT:")
                print("=" * 50)
                print(text)
        
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()