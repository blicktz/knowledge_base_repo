#!/usr/bin/env python3
"""
RunPod FastAPI Audio Transcription Server
Provides HTTP API endpoints for easy audio file upload and transcript download.
"""

import os
import sys
import time
import json
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import hashlib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

try:
    import whisper
    import torch
except ImportError as e:
    print(f"Error: Required module not installed: {e}")
    sys.exit(1)

# Configuration
API_KEY = os.environ.get("TRANSCRIBE_API_KEY", "mv_mtvG2X4U_dqRgdWMvSEoFtpMjRJkL4zlkwEXYH2I")
UPLOAD_DIR = Path("/workspace/uploads")
OUTPUT_DIR = Path("/workspace/outputs")
JOBS_DIR = Path("/workspace/jobs")

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# FastAPI app
app = FastAPI(title="RunPod Transcription API", version="1.0.0")
security = HTTPBearer()

# Global model cache
model_cache = {}

# Job tracking
jobs_db = {}

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    created_at: str
    updated_at: str
    files_count: int
    processed_count: int
    failed_count: int
    error: Optional[str] = None
    download_url: Optional[str] = None

class TranscriptionRequest(BaseModel):
    model: str = "turbo"

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication."""
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials.credentials

def get_device():
    """Determine the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_model(model_name: str = "turbo"):
    """Load and cache Whisper model."""
    if model_name not in model_cache:
        device = get_device()
        print(f"Loading {model_name} model on {device}...")
        model_cache[model_name] = whisper.load_model(model_name, device=device)
    return model_cache[model_name]

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
    """Simple transcript formatting."""
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

async def process_audio_file(audio_path: Path, model, output_dir: Path) -> Dict[str, Any]:
    """Process a single audio file."""
    try:
        # Generate safe output filename
        safe_name = slugify_filename(audio_path.stem)
        output_file = output_dir / f"{safe_name}.txt"
        
        # Transcribe
        result = model.transcribe(str(audio_path), fp16=False, verbose=False)
        raw_text = result["text"].strip()
        
        # Format transcript
        formatted_text = format_transcript_simple(raw_text)
        
        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        
        return {
            "status": "success",
            "input_file": audio_path.name,
            "output_file": output_file.name,
            "transcript_length": len(formatted_text)
        }
    except Exception as e:
        return {
            "status": "failed",
            "input_file": audio_path.name,
            "error": str(e)
        }

async def process_job(job_id: str, audio_files: List[Path], model_name: str):
    """Process transcription job in background."""
    job = jobs_db[job_id]
    job["status"] = "processing"
    job["updated_at"] = datetime.now().isoformat()
    
    try:
        # Load model
        model = load_model(model_name)
        
        # Create job output directory
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        for i, audio_file in enumerate(audio_files):
            result = await process_audio_file(audio_file, model, job_output_dir)
            
            if result["status"] == "success":
                job["processed_count"] += 1
            else:
                job["failed_count"] += 1
            
            job["updated_at"] = datetime.now().isoformat()
        
        # Create results archive
        if job["processed_count"] > 0:
            archive_path = OUTPUT_DIR / f"{job_id}.zip"
            shutil.make_archive(str(archive_path.with_suffix('')), 'zip', job_output_dir)
            job["download_url"] = f"/download/{job_id}"
        
        job["status"] = "completed"
        job["updated_at"] = datetime.now().isoformat()
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["updated_at"] = datetime.now().isoformat()
    
    finally:
        # Clean up upload files
        for audio_file in audio_files:
            try:
                audio_file.unlink()
            except:
                pass

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "RunPod Transcription API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "status": "/status/{job_id}",
            "download": "/download/{job_id}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    device = get_device()
    return {
        "status": "healthy",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "models_cached": list(model_cache.keys())
    }

@app.post("/upload-single")
async def upload_single_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = "turbo",
    api_key: str = Depends(verify_api_key)
):
    """Upload a single audio file for transcription."""
    # Validate model
    valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"]
    if model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {valid_models}")
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Supported: MP3, WAV, M4A, FLAC, OGG")
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Create job directory
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    file_path = job_upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Create job record
    job = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "files_count": 1,
        "processed_count": 0,
        "failed_count": 0,
        "model": model,
        "error": None,
        "download_url": None
    }
    jobs_db[job_id] = job
    
    # Start background processing
    background_tasks.add_task(process_job, job_id, [file_path], model)
    
    return {
        "job_id": job_id,
        "status": "pending",
        "files_count": 1,
        "file_name": file.filename,
        "file_size": len(content),
        "message": "File uploaded successfully. Processing started.",
        "status_url": f"/status/{job_id}"
    }

@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model: str = "turbo",
    api_key: str = Depends(verify_api_key)
):
    """Upload audio files for transcription."""
    # Validate model
    valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"]
    if model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {valid_models}")
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Create job directory
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded files
    audio_files = []
    for file in files:
        # Validate file type
        if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg')):
            continue
        
        # Save file
        file_path = job_upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        audio_files.append(file_path)
    
    if not audio_files:
        shutil.rmtree(job_upload_dir)
        raise HTTPException(status_code=400, detail="No valid audio files uploaded")
    
    # Create job record
    job = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "files_count": len(audio_files),
        "processed_count": 0,
        "failed_count": 0,
        "model": model,
        "error": None,
        "download_url": None
    }
    jobs_db[job_id] = job
    
    # Start background processing
    background_tasks.add_task(process_job, job_id, audio_files, model)
    
    return {
        "job_id": job_id,
        "status": "pending",
        "files_count": len(audio_files),
        "message": "Files uploaded successfully. Processing started.",
        "status_url": f"/status/{job_id}"
    }

@app.get("/status/{job_id}")
async def get_status(job_id: str, api_key: str = Depends(verify_api_key)):
    """Get job status."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    return JobStatus(**job)

@app.get("/download/{job_id}")
async def download_results(job_id: str, api_key: str = Depends(verify_api_key)):
    """Download transcription results."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {job['status']}, not completed")
    
    archive_path = OUTPUT_DIR / f"{job_id}.zip"
    if not archive_path.exists():
        raise HTTPException(status_code=404, detail="Results file not found")
    
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=f"transcripts_{job_id}.zip"
    )

@app.delete("/job/{job_id}")
async def delete_job(job_id: str, api_key: str = Depends(verify_api_key)):
    """Delete job and its files."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete files
    job_upload_dir = UPLOAD_DIR / job_id
    job_output_dir = OUTPUT_DIR / job_id
    archive_path = OUTPUT_DIR / f"{job_id}.zip"
    
    for path in [job_upload_dir, job_output_dir, archive_path]:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    
    # Remove from database
    del jobs_db[job_id]
    
    return {"message": "Job deleted successfully"}

@app.on_event("startup")
async def startup_event():
    """Preload default model on startup."""
    print("Starting RunPod Transcription API Server...")
    print(f"Device: {get_device()}")
    print(f"API Key configured: {'Yes' if API_KEY != 'your-secret-api-key-here' else 'No (using default)'}")
    
    # Preload turbo model
    print("Preloading turbo model...")
    load_model("turbo")
    print("Server ready!")

def main():
    """Main entry point."""
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()