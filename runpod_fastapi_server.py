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
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import hashlib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

try:
    import whisper
    import torch
    import aiofiles
except ImportError as e:
    logger.error(f"Required module not installed: {e}")
    logger.error("Please install missing packages with: pip install openai-whisper torch aiofiles")
    sys.exit(1)

# Configuration
API_KEY = os.environ.get("TRANSCRIBE_API_KEY", "mv_mtvG2X4U_dqRgdWMvSEoFtpMjRJkL4zlkwEXYH2I")
UPLOAD_DIR = Path("/workspace/uploads")
OUTPUT_DIR = Path("/workspace/outputs")
JOBS_DIR = Path("/workspace/jobs")

# Multi-client concurrent processing configuration
WORKER_COUNT = int(os.environ.get("WORKER_COUNT", 6))
MODEL_INSTANCES = int(os.environ.get("MODEL_INSTANCES", 6))
MODEL_NAME = os.environ.get("MODEL_NAME", "turbo")

# RAM filesystem configuration
USE_RAM_FILESYSTEM = os.environ.get("USE_RAM_FILESYSTEM", "true").lower() == "true"
RAM_FILESYSTEM_SIZE = os.environ.get("RAM_FILESYSTEM_SIZE", "40g")

# Directory setup will be handled in startup_event after RAM filesystem setup

# FastAPI app
app = FastAPI(title="RunPod Transcription API", version="1.0.0")
security = HTTPBearer()

# Model pool for concurrent processing
model_pool = []

# Job tracking with thread safety
jobs_db = {}
jobs_db_lock = asyncio.Lock()

# Processing queue system
processing_queue = asyncio.Queue()
queue_worker_tasks = []  # List to track all worker tasks

class JobStatus(BaseModel):
    job_id: str
    status: str  # uploaded, queued, processing, completed, failed
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

def setup_ram_filesystem():
    """Set up RAM filesystem using /dev/shm for faster I/O operations."""
    if not USE_RAM_FILESYSTEM:
        logger.info("RAM filesystem disabled, using disk storage")
        return False
    
    try:
        # Use /dev/shm which is already mounted as tmpfs in containers
        # This is a standard RAM disk available in most Linux environments
        shm_path = Path("/dev/shm")
        
        # Check if /dev/shm exists and is writable
        if not shm_path.exists():
            logger.warning("/dev/shm not available, falling back to disk storage")
            return False
        
        # Check if /dev/shm is actually tmpfs
        result = subprocess.run(
            ["df", "-T", str(shm_path)], capture_output=True, text=True, timeout=5
        )
        if "tmpfs" not in result.stdout:
            logger.warning("/dev/shm is not tmpfs, might not be RAM-based")
        
        # Create our transcription directory in /dev/shm
        ram_base = shm_path / "transcription"
        ram_base.mkdir(parents=True, exist_ok=True)
        
        # Update global paths to use RAM filesystem
        global UPLOAD_DIR, OUTPUT_DIR, JOBS_DIR
        UPLOAD_DIR = ram_base / "uploads"
        OUTPUT_DIR = ram_base / "outputs"
        JOBS_DIR = ram_base / "jobs"
        
        # Create subdirectories
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Check available space in /dev/shm
        statvfs = os.statvfs(str(shm_path))
        available_gb = (statvfs.f_bavail * statvfs.f_frsize) / (1024**3)
        total_gb = (statvfs.f_blocks * statvfs.f_frsize) / (1024**3)
        
        logger.info(f"✅ RAM filesystem configured using /dev/shm")
        logger.info(f"   Total: {total_gb:.1f}GB, Available: {available_gb:.1f}GB")
        logger.info(f"   Uploads: {UPLOAD_DIR}")
        logger.info(f"   Outputs: {OUTPUT_DIR}")
        logger.info(f"   Jobs: {JOBS_DIR}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup RAM filesystem: {e}")
        logger.info("Falling back to disk storage")
        return False

def get_ram_usage():
    """Get current RAM usage statistics."""
    try:
        # Read memory info from /proc/meminfo
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        mem_total = 0
        mem_available = 0
        
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                mem_total = int(line.split()[1]) * 1024  # Convert KB to bytes
            elif line.startswith('MemAvailable:'):
                mem_available = int(line.split()[1]) * 1024  # Convert KB to bytes
        
        mem_used = mem_total - mem_available
        usage_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
        
        return {
            "total_gb": round(mem_total / (1024**3), 1),
            "used_gb": round(mem_used / (1024**3), 1),
            "available_gb": round(mem_available / (1024**3), 1),
            "usage_percent": round(usage_percent, 1)
        }
    except Exception as e:
        logger.warning(f"Could not get RAM usage: {e}")
        return {
            "total_gb": 0,
            "used_gb": 0,
            "available_gb": 0,
            "usage_percent": 0
        }

def get_gpu_memory_usage():
    """Get GPU memory usage if CUDA is available."""
    try:
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        device_count = torch.cuda.device_count()
        gpu_info = {}
        
        for i in range(device_count):
            mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)   # GB
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            
            gpu_info[f"gpu_{i}"] = {
                "allocated_gb": round(mem_allocated, 1),
                "reserved_gb": round(mem_reserved, 1),
                "total_gb": round(mem_total, 1),
                "usage_percent": round((mem_reserved / mem_total * 100), 1)
            }
        
        return gpu_info
    except Exception as e:
        logger.warning(f"Could not get GPU memory usage: {e}")
        return {"error": str(e)}

async def cleanup_job_files(job_id: str, audio_files: List[Path]):
    """Enhanced cleanup routine for memory management."""
    try:
        # Clean up upload files
        for audio_file in audio_files:
            try:
                if audio_file.exists():
                    audio_file.unlink()
            except Exception as e:
                logger.warning(f"Could not remove upload file {audio_file}: {e}")
        
        # Clean up upload directory if empty
        job_upload_dir = UPLOAD_DIR / job_id
        try:
            if job_upload_dir.exists() and not any(job_upload_dir.iterdir()):
                job_upload_dir.rmdir()
        except Exception as e:
            logger.warning(f"Could not remove upload directory {job_upload_dir}: {e}")
        
        # Force garbage collection to free RAM
        import gc
        gc.collect()
        
        # Force GPU memory cleanup if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"Error during enhanced cleanup for job {job_id}: {e}")

async def cleanup_old_jobs(max_age_hours: int = 24):
    """Clean up old completed/failed jobs to free up space."""
    try:
        async with jobs_db_lock:
            current_time = datetime.now()
            jobs_to_delete = []
            
            for job_id, job in jobs_db.items():
                if job["status"] in ["completed", "failed"]:
                    updated_at = datetime.fromisoformat(job["updated_at"])
                    age_hours = (current_time - updated_at).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        jobs_to_delete.append(job_id)
            
            # Delete old jobs
            for job_id in jobs_to_delete:
                await delete_job_files(job_id)
                del jobs_db[job_id]
                logger.info(f"Cleaned up old job: {job_id}")
                
            if jobs_to_delete:
                logger.info(f"Cleaned up {len(jobs_to_delete)} old jobs")
                
    except Exception as e:
        logger.error(f"Error during old job cleanup: {e}")

async def delete_job_files(job_id: str):
    """Delete all files associated with a job."""
    try:
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
    except Exception as e:
        logger.warning(f"Error deleting job files for {job_id}: {e}")

def load_model_instance(model_name: str = "turbo", instance_id: int = 0):
    """Load a Whisper model instance."""
    device = get_device()
    logger.info(f"Loading model instance {instance_id}: {model_name} on {device}...")
    return whisper.load_model(model_name, device=device)

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
        
        # Save to file (async)
        async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
            await f.write(formatted_text)
        
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

async def process_job(job_id: str, audio_files: List[Path], model_name: str, model_instance):
    """Process transcription job in background."""
    async with jobs_db_lock:
        job = jobs_db[job_id]
        job["status"] = "processing"
        job["updated_at"] = datetime.now().isoformat()
    
    try:
        # Use provided model instance
        model = model_instance
        
        # Create job output directory
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        for i, audio_file in enumerate(audio_files):
            result = await process_audio_file(audio_file, model, job_output_dir)
            
            async with jobs_db_lock:
                if result["status"] == "success":
                    job["processed_count"] += 1
                else:
                    job["failed_count"] += 1
                
                job["updated_at"] = datetime.now().isoformat()
        
        # Create results archive
        async with jobs_db_lock:
            if job["processed_count"] > 0:
                archive_path = OUTPUT_DIR / f"{job_id}.zip"
                shutil.make_archive(str(archive_path.with_suffix('')), 'zip', job_output_dir)
                job["download_url"] = f"/download/{job_id}"
            
            job["status"] = "completed"
            job["updated_at"] = datetime.now().isoformat()
        
    except Exception as e:
        async with jobs_db_lock:
            job["status"] = "failed"
            job["error"] = str(e)
            job["updated_at"] = datetime.now().isoformat()
    
    finally:
        # Enhanced cleanup for memory management
        await cleanup_job_files(job_id, audio_files)

async def queue_worker(worker_id: int, model_instance):
    """Background worker that processes jobs from the queue."""
    logger.info(f"Worker {worker_id} started with dedicated model instance")
    
    while True:
        try:
            # Get next job from queue
            job_data = await processing_queue.get()
            job_id = job_data["job_id"]
            
            async with jobs_db_lock:
                if job_id not in jobs_db:
                    logger.warning(f"Worker {worker_id}: Job {job_id} not found in database")
                    processing_queue.task_done()
                    continue
                
                # Update status to processing
                jobs_db[job_id]["status"] = "processing"
                jobs_db[job_id]["updated_at"] = datetime.now().isoformat()
                jobs_db[job_id]["worker_id"] = worker_id
                
                # Get job details
                file_path = Path(jobs_db[job_id]["file_path"])
                model_name = jobs_db[job_id]["model"]
            
            logger.info(f"Worker {worker_id}: Processing job {job_id}")
            
            # Process the job with dedicated model instance
            await process_job(job_id, [file_path], model_name, model_instance)
            
            processing_queue.task_done()
            logger.info(f"Worker {worker_id}: Completed job {job_id}")
            
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
            if 'job_id' in locals():
                async with jobs_db_lock:
                    if job_id in jobs_db:
                        jobs_db[job_id]["status"] = "failed"
                        jobs_db[job_id]["error"] = str(e)
                        jobs_db[job_id]["updated_at"] = datetime.now().isoformat()

@app.post("/process/{job_id}")
async def start_processing(job_id: str, api_key: str = Depends(verify_api_key)):
    """Start processing an uploaded file."""
    async with jobs_db_lock:
        if job_id not in jobs_db:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs_db[job_id]
        
        if job["status"] != "uploaded":
            raise HTTPException(
                status_code=400, 
                detail=f"Job status is '{job['status']}', expected 'uploaded'"
            )
        
        # Add job to processing queue
        job["status"] = "queued"
        job["updated_at"] = datetime.now().isoformat()
    
    await processing_queue.put({"job_id": job_id})
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Job added to processing queue",
        "queue_size": processing_queue.qsize(),
        "active_workers": WORKER_COUNT
    }

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "RunPod Transcription API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload-single",
            "process": "/process/{job_id}",
            "status": "/status/{job_id}",
            "download": "/download/{job_id}",
            "health": "/health",
            "system": "/system"
        },
        "workflow": "1. Upload file → 2. Start processing → 3. Check status → 4. Download results",
        "concurrency": {
            "max_workers": WORKER_COUNT,
            "model_instances": MODEL_INSTANCES
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
        "model_instances": len(model_pool),
        "active_workers": len(queue_worker_tasks),
        "queue_size": processing_queue.qsize()
    }

@app.get("/system")
async def system_status(api_key: str = Depends(verify_api_key)):
    """Get detailed system status including worker and job information."""
    async with jobs_db_lock:
        # Count jobs by status
        job_stats = {
            "total": len(jobs_db),
            "uploaded": sum(1 for j in jobs_db.values() if j["status"] == "uploaded"),
            "queued": sum(1 for j in jobs_db.values() if j["status"] == "queued"),
            "processing": sum(1 for j in jobs_db.values() if j["status"] == "processing"),
            "completed": sum(1 for j in jobs_db.values() if j["status"] == "completed"),
            "failed": sum(1 for j in jobs_db.values() if j["status"] == "failed")
        }
        
        # Get active processing jobs
        active_jobs = [
            {
                "job_id": j["job_id"],
                "worker_id": j.get("worker_id"),
                "status": j["status"],
                "updated_at": j["updated_at"]
            }
            for j in jobs_db.values() if j["status"] == "processing"
        ]
    
    return {
        "workers": {
            "configured": WORKER_COUNT,
            "active": len(queue_worker_tasks),
            "model_instances": MODEL_INSTANCES
        },
        "queue": {
            "size": processing_queue.qsize(),
            "max_size": processing_queue.maxsize if processing_queue.maxsize else "unlimited"
        },
        "jobs": job_stats,
        "active_processing": active_jobs,
        "device": {
            "type": get_device(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        },
        "model": {
            "name": MODEL_NAME,
            "instances_loaded": len(model_pool)
        },
        "memory": {
            "ram": get_ram_usage(),
            "gpu": get_gpu_memory_usage()
        },
        "storage": {
            "ram_filesystem_enabled": USE_RAM_FILESYSTEM,
            "upload_dir": str(UPLOAD_DIR),
            "output_dir": str(OUTPUT_DIR)
        }
    }

@app.post("/upload-single")
async def upload_single_file(
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
    
    # Save uploaded file (async)
    file_path = job_upload_dir / file.filename
    content = await file.read()
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)
    
    # Create job record
    job = {
        "job_id": job_id,
        "status": "uploaded",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "files_count": 1,
        "processed_count": 0,
        "failed_count": 0,
        "model": model,
        "file_path": str(file_path),
        "error": None,
        "download_url": None,
        "worker_id": None  # Track which worker processes this job
    }
    
    async with jobs_db_lock:
        jobs_db[job_id] = job
    
    # File uploaded successfully - processing will be started separately
    
    return {
        "job_id": job_id,
        "status": "uploaded",
        "files_count": 1,
        "file_name": file.filename,
        "file_size": len(content),
        "message": "File uploaded successfully. Use /process/{job_id} to start transcription.",
        "status_url": f"/status/{job_id}",
        "process_url": f"/process/{job_id}"
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
        
        # Save file (async)
        file_path = job_upload_dir / file.filename
        content = await file.read()
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
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
        "download_url": None,
        "worker_id": None
    }
    
    async with jobs_db_lock:
        jobs_db[job_id] = job
    
    # Start background processing - pick a model from pool
    model_instance = model_pool[job_id.__hash__() % len(model_pool)] if model_pool else None
    background_tasks.add_task(process_job, job_id, audio_files, model, model_instance)
    
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
    async with jobs_db_lock:
        if job_id not in jobs_db:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs_db[job_id].copy()
    
    # Remove worker_id from response if present
    if "worker_id" in job:
        del job["worker_id"]
    
    return JobStatus(**job)

@app.get("/download/{job_id}")
async def download_results(job_id: str, api_key: str = Depends(verify_api_key)):
    """Download transcription results."""
    async with jobs_db_lock:
        if job_id not in jobs_db:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs_db[job_id].copy()
    
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
    async with jobs_db_lock:
        if job_id not in jobs_db:
            raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete files using enhanced cleanup
    await delete_job_files(job_id)
    
    # Remove from database
    async with jobs_db_lock:
        del jobs_db[job_id]
    
    # Force cleanup to free memory
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {"message": "Job deleted successfully"}

@app.on_event("startup")
async def startup_event():
    """Initialize model pool and start multiple background workers on startup."""
    global model_pool, queue_worker_tasks, UPLOAD_DIR, OUTPUT_DIR, JOBS_DIR
    
    logger.info("Starting RunPod Transcription API Server...")
    logger.info(f"Device: {get_device()}")
    logger.info(f"API Key configured: {'Yes' if API_KEY != 'mv_mtvG2X4U_dqRgdWMvSEoFtpMjRJkL4zlkwEXYH2I' else 'No (using default)'}")
    logger.info(f"Worker Count: {WORKER_COUNT}")
    logger.info(f"Model Instances: {MODEL_INSTANCES}")
    logger.info(f"Model Name: {MODEL_NAME}")
    logger.info(f"RAM Filesystem: {'Enabled' if USE_RAM_FILESYSTEM else 'Disabled'}")
    
    # Setup RAM filesystem if enabled
    ram_fs_setup = setup_ram_filesystem()
    
    # Fallback to disk if RAM filesystem setup failed
    if not ram_fs_setup:
        try:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not create directories (probably running locally): {e}")
            # Use local temp directories for testing
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "transcription_test"
            UPLOAD_DIR = temp_dir / "uploads"
            OUTPUT_DIR = temp_dir / "outputs" 
            JOBS_DIR = temp_dir / "jobs"
            
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Working directories: upload={UPLOAD_DIR}, output={OUTPUT_DIR}")
    
    # Load model pool for concurrent processing
    logger.info(f"Loading {MODEL_INSTANCES} model instances...")
    for i in range(MODEL_INSTANCES):
        logger.info(f"Loading model instance {i + 1}/{MODEL_INSTANCES}...")
        model = load_model_instance(MODEL_NAME, i)
        model_pool.append(model)
    
    # Log GPU memory usage after loading models
    gpu_usage = get_gpu_memory_usage()
    if "error" not in gpu_usage:
        for gpu_id, info in gpu_usage.items():
            logger.info(f"{gpu_id}: {info['usage_percent']}% used ({info['reserved_gb']:.1f}GB/{info['total_gb']:.1f}GB)")
    
    logger.info(f"Starting {WORKER_COUNT} background workers...")
    for worker_id in range(WORKER_COUNT):
        # Assign model to worker (round-robin if fewer models than workers)
        model_instance = model_pool[worker_id % len(model_pool)]
        
        # Create and start worker task
        worker_task = asyncio.create_task(queue_worker(worker_id, model_instance))
        queue_worker_tasks.append(worker_task)
        logger.info(f"Worker {worker_id} started")
    
    # Start periodic cleanup task
    async def periodic_cleanup():
        while True:
            await asyncio.sleep(3600)  # Run every hour
            await cleanup_old_jobs(max_age_hours=24)
    
    asyncio.create_task(periodic_cleanup())
    
    # Log final system status
    ram_usage = get_ram_usage()
    logger.info(f"RAM usage: {ram_usage['usage_percent']}% ({ram_usage['used_gb']:.1f}GB/{ram_usage['total_gb']:.1f}GB)")
    logger.info(f"Server ready with {WORKER_COUNT} concurrent workers!")

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