#!/usr/bin/env python3
"""
RunPod Batch Audio Transcription Script
Optimized for processing large batches of audio files in containerized GPU environment.
"""

import os
import sys
import time
import json
import zipfile
import argparse
from pathlib import Path
from typing import List, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

try:
    import whisper
    import torch
except ImportError as e:
    print(f"Error: Required module not installed: {e}")
    sys.exit(1)

# Container paths
INPUT_DIR = Path("/workspace/input")
OUTPUT_DIR = Path("/workspace/output")
PROGRESS_FILE = Path("/workspace/progress.json")

def get_device():
    """Determine the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def save_progress(progress_data: Dict[str, Any]):
    """Save processing progress to JSON file for monitoring."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress_data, f, indent=2)

def extract_audio_files(zip_path: Path) -> List[Path]:
    """Extract audio files from uploaded zip archive."""
    print(f"📦 Extracting audio files from {zip_path.name}...")
    
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    audio_files = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.lower().endswith('.mp3') and not file_info.filename.startswith('__MACOSX'):
                # Extract to input directory
                extracted_path = INPUT_DIR / file_info.filename
                extracted_path.parent.mkdir(parents=True, exist_ok=True)
                
                with zip_ref.open(file_info) as source, open(extracted_path, 'wb') as target:
                    target.write(source.read())
                
                audio_files.append(extracted_path)
    
    print(f"✓ Extracted {len(audio_files)} audio files")
    return sorted(audio_files)

def slugify_filename(filename: str, max_length: int = 100) -> str:
    """Create safe filename for output."""
    import re
    import unicodedata
    
    # Remove extension
    if '.' in filename:
        name = filename.rsplit('.', 1)[0]
    else:
        name = filename
    
    # Normalize and clean
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = name.lower()
    
    # Replace problematic characters
    char_replacements = {
        '$': 'dollar', '⧸': '_', '/': '_', '\\': '_', ':': '_',
        '*': '_', '?': '_', '"': '_', '<': '_', '>': '_', '|': '_',
        '(': '_', ')': '_', '[': '_', ']': '_', '{': '_', '}': '_',
        '&': 'and', '%': 'percent', '#': '_', '@': 'at',
        '+': 'plus', '=': '_', '!': '_', '~': '_', '`': '_', '^': '_',
    }
    
    for char, replacement in char_replacements.items():
        name = name.replace(char, replacement)
    
    # Clean up spaces and underscores
    name = re.sub(r'[\s\-]+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    
    if not name:
        name = 'transcript'
    
    if len(name) > max_length:
        name = name[:max_length].rstrip('_')
    
    return name

def format_transcript_simple(text: str) -> str:
    """Simple transcript formatting for container environment."""
    import re
    
    if not text.strip():
        return text
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 3:
        return ' '.join(sentences)
    
    # Group every 3-4 sentences into paragraphs
    paragraphs = []
    for i in range(0, len(sentences), 3):
        paragraph = ' '.join(sentences[i:i+3]).strip()
        if paragraph:
            paragraphs.append(paragraph)
    
    return '\n\n'.join(paragraphs)

def transcribe_batch(audio_files: List[Path], model_name: str = "turbo") -> Dict[str, Any]:
    """Transcribe a batch of audio files."""
    print(f"🚀 Starting batch transcription with {model_name} model...")
    
    # Load model once
    device = get_device()
    print(f"📱 Using device: {device.upper()}")
    
    model = whisper.load_model(model_name, device=device)
    print(f"✓ Model loaded: {model_name}")
    
    stats = {
        "total": len(audio_files),
        "processed": 0,
        "failed": 0,
        "failed_files": [],
        "start_time": time.time(),
        "estimated_completion": None
    }
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for i, audio_file in enumerate(audio_files, 1):
        file_start_time = time.time()
        
        try:
            # Generate safe output filename
            safe_name = slugify_filename(audio_file.stem)
            output_file = OUTPUT_DIR / f"{safe_name}.txt"
            
            print(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")
            if safe_name != audio_file.stem:
                print(f"  Output: {output_file.name}")
            
            # Transcribe
            result = model.transcribe(str(audio_file), fp16=False, verbose=False)
            raw_text = result["text"].strip()
            
            # Format transcript
            formatted_text = format_transcript_simple(raw_text)
            
            # Save to file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(formatted_text)
            
            elapsed = time.time() - file_start_time
            stats["processed"] += 1
            
            # Calculate ETA
            avg_time_per_file = (time.time() - stats["start_time"]) / stats["processed"]
            remaining_files = stats["total"] - stats["processed"]
            eta_seconds = avg_time_per_file * remaining_files
            
            print(f"  ✓ Success! ({elapsed:.1f}s)")
            print(f"  📊 Progress: {stats['processed']}/{stats['total']} | ETA: {eta_seconds/60:.1f} mins")
            
            # Update progress for monitoring
            progress_data = {
                "current_file": i,
                "total_files": len(audio_files),
                "processed": stats["processed"],
                "failed": stats["failed"],
                "current_filename": audio_file.name,
                "eta_minutes": eta_seconds / 60,
                "percent_complete": (stats["processed"] / stats["total"]) * 100
            }
            save_progress(progress_data)
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            stats["failed"] += 1
            stats["failed_files"].append(str(audio_file))
            
            # Update progress even on failure
            progress_data = {
                "current_file": i,
                "total_files": len(audio_files),
                "processed": stats["processed"],
                "failed": stats["failed"],
                "current_filename": f"FAILED: {audio_file.name}",
                "error": str(e)
            }
            save_progress(progress_data)
        
        print()  # Empty line for readability
    
    # Final statistics
    total_time = time.time() - stats["start_time"]
    stats["total_time_minutes"] = total_time / 60
    
    return stats

def create_output_zip() -> Path:
    """Create zip file of all transcripts for easy download."""
    zip_path = Path("/workspace/transcripts.zip")
    
    print("📦 Creating transcript archive...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for txt_file in OUTPUT_DIR.glob("*.txt"):
            zipf.write(txt_file, txt_file.name)
    
    print(f"✓ Created: {zip_path.name}")
    return zip_path

def main():
    """Main entry point for containerized batch processing."""
    parser = argparse.ArgumentParser(description="RunPod Batch Audio Transcription")
    parser.add_argument("--input-zip", default="/workspace/audio_files.zip", 
                       help="Path to input zip file containing MP3s")
    parser.add_argument("--model", default="turbo", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"],
                       help="Whisper model to use")
    parser.add_argument("--no-zip-output", action="store_true",
                       help="Don't create output zip file")
    
    args = parser.parse_args()
    
    print("🎵 RunPod Batch Audio Transcription")
    print("=" * 50)
    
    try:
        # Check for input zip file
        input_zip = Path(args.input_zip)
        if not input_zip.exists():
            # Look for any zip file in workspace
            zip_files = list(Path("/workspace").glob("*.zip"))
            if zip_files:
                input_zip = zip_files[0]
                print(f"📁 Found input zip: {input_zip.name}")
            else:
                print("❌ No input zip file found")
                print("Expected: audio files uploaded as ZIP archive")
                sys.exit(1)
        
        # Extract audio files
        audio_files = extract_audio_files(input_zip)
        
        if not audio_files:
            print("❌ No MP3 files found in archive")
            sys.exit(1)
        
        # Process all files
        stats = transcribe_batch(audio_files, args.model)
        
        # Create output archive
        if not args.no_zip_output:
            create_output_zip()
        
        # Print final summary
        print("=" * 50)
        print("🎉 Batch Processing Complete!")
        print("=" * 50)
        print(f"📊 Total files:    {stats['total']}")
        print(f"✅ Processed:      {stats['processed']}")
        print(f"❌ Failed:         {stats['failed']}")
        print(f"⏱️  Total time:     {stats['total_time_minutes']:.1f} minutes")
        
        if stats['failed_files']:
            print("\n❌ Failed files:")
            for f in stats['failed_files']:
                print(f"  - {f}")
        
        print(f"\n📁 Output available in: {OUTPUT_DIR}")
        if not args.no_zip_output:
            print("📦 Download: transcripts.zip")
        
        # Final progress update
        final_progress = {
            "status": "completed",
            "total_files": stats['total'],
            "processed": stats['processed'],
            "failed": stats['failed'],
            "total_time_minutes": stats['total_time_minutes'],
            "failed_files": stats['failed_files']
        }
        save_progress(final_progress)
        
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        
        # Save error to progress file
        error_progress = {
            "status": "error",
            "error": str(e)
        }
        save_progress(error_progress)
        
        sys.exit(1)

if __name__ == "__main__":
    main()