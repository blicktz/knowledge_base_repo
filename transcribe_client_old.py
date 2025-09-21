#!/usr/bin/env python3
"""
Async Parallel Client for RunPod FastAPI Transcription Server
Processes up to 6 files concurrently for maximum server utilization.
Usage: python transcribe_client.py /path/to/audio/files /path/to/output [--server URL] [--api-key KEY]
"""

import os
import sys
import time
import json
import argparse
import zipfile
import random
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional
import httpx
from httpx import RequestError
from dotenv import load_dotenv

def slugify_filename(filename: str, max_length: int = 100) -> str:
    """Create safe filename for output (matches server logic)."""
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

class ProgressManager:
    """Manages progress display for concurrent file processing."""
    
    def __init__(self, total_files: int):
        self.total_files = total_files
        self.completed = 0
        self.failed = 0
        self.active_jobs = {}  # {job_id: {file: name, status: uploading/processing/downloading}}
        self.lock = asyncio.Lock()
        self.start_time = time.time()
    
    async def update_status(self, job_id: str, file_name: str, status: str):
        """Update status for a specific job."""
        async with self.lock:
            self.active_jobs[job_id] = {"file": file_name, "status": status}
            self._display_progress()
    
    async def mark_completed(self, job_id: str, success: bool = True):
        """Mark a job as completed (success or failure)."""
        async with self.lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1
            
            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            self._display_progress()
    
    def _display_progress(self):
        """Display current progress (called with lock held)."""
        elapsed = time.time() - self.start_time
        
        # Clear previous lines
        sys.stdout.write('\033[2K\r')  # Clear current line
        
        # Main progress line
        progress_pct = (self.completed / self.total_files * 100) if self.total_files > 0 else 0
        active_count = len(self.active_jobs)
        
        print(f"\r📊 Progress: {self.completed}/{self.total_files} completed ({progress_pct:.1f}%), "
              f"{self.failed} failed, {active_count} active | "
              f"⏱️  {elapsed:.0f}s elapsed", end='')
        
        # Show active jobs (limit to prevent spam)
        if self.active_jobs:
            active_list = list(self.active_jobs.values())[:3]  # Show max 3 active jobs
            for i, job_info in enumerate(active_list):
                file_name = job_info['file'][:30] + '...' if len(job_info['file']) > 30 else job_info['file']
                status_emoji = {
                    'uploading': '📤',
                    'starting': '🚀', 
                    'processing': '⚙️',
                    'downloading': '📥'
                }.get(job_info['status'], '🔄')
                print(f"\n   {status_emoji} {file_name}: {job_info['status']}", end='')
            
            if len(self.active_jobs) > 3:
                print(f"\n   ... and {len(self.active_jobs) - 3} more", end='')
        
        sys.stdout.flush()
    
    def print_final_summary(self):
        """Print final summary after all processing is complete."""
        elapsed = time.time() - self.start_time
        print(f"\n\n🎉 Processing Complete!")
        print(f"📊 Final Results:")
        print(f"   ✅ Completed: {self.completed}")
        print(f"   ❌ Failed: {self.failed}")
        print(f"   ⏱️  Total time: {elapsed:.1f} seconds")
        if self.completed > 0:
            avg_time = elapsed / self.completed
            print(f"   📈 Average per file: {avg_time:.1f} seconds")

class TranscriptionClient:
    """Async client for RunPod FastAPI transcription server."""
    
    def __init__(self, server_url: str, api_key: str, http_client: httpx.AsyncClient):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.http_client = http_client
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Connection": "keep-alive",
            "User-Agent": "transcription-client-async/1.0"
        }
    
    async def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            response = await self.http_client.get(
                f"{self.server_url}/health", 
                headers=self.headers, 
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Server is healthy (Device: {data.get('device', 'unknown')})")
                return True
            else:
                print(f"❌ Server returned status {response.status_code}")
                return False
        except RequestError as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
    
    async def upload_file(self, audio_file: Path, model: str = "turbo", max_retries: int = 3) -> Optional[str]:
        """Upload a single audio file."""
        for attempt in range(max_retries):
            try:
                with open(audio_file, 'rb') as f:
                    files = {'file': (audio_file.name, f, 'audio/mpeg')}
                    data = {'model': model}
                    
                    response = await self.http_client.post(
                        f"{self.server_url}/upload-single",
                        files=files,
                        data=data,
                        headers=self.headers,
                        timeout=1800  # 30 minutes timeout for large files
                    )
                
                if response.status_code == 200:
                    data = response.json()
                    job_id = data.get('job_id')
                    return job_id
                else:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                    
            except RequestError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
        
        return None
    
    async def start_processing(self, job_id: str, max_retries: int = 3) -> tuple[bool, bool]:
        """Start processing an uploaded file."""
        for attempt in range(max_retries):
            try:
                response = await self.http_client.post(
                    f"{self.server_url}/process/{job_id}",
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return True, False
                else:
                    # Parse error response to detect if job is already completed
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', response.text)
                        
                        # Check if the error indicates job is already completed
                        if "status is 'completed'" in error_detail:
                            return True, True
                        elif "status is 'failed'" in error_detail:
                            return False, False
                        elif "status is" in error_detail:
                            # Check actual status
                            status = await self.check_status(job_id)
                            if status:
                                if status.get('status') == 'completed':
                                    return True, True
                                elif status.get('status') == 'failed':
                                    return False, False
                                elif status.get('status') == 'processing':
                                    return True, False
                        
                        if "status is" not in str(error_detail) and attempt < max_retries - 1:
                            await asyncio.sleep(2)
                        elif "status is" in str(error_detail):
                            break
                            
                    except (json.JSONDecodeError, KeyError):
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                        
            except RequestError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
        
        return False, False
    
    async def check_status(self, job_id: str) -> Optional[dict]:
        """Check job status."""
        try:
            response = await self.http_client.get(
                f"{self.server_url}/status/{job_id}",
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except RequestError:
            return None
    
    async def wait_for_completion(self, job_id: str, check_interval: int = 3) -> bool:
        """Wait for job to complete."""
        while True:
            status = await self.check_status(job_id)
            
            if not status:
                await asyncio.sleep(check_interval)
                continue
            
            if status.get('status') == 'completed':
                return True
            elif status.get('status') == 'failed':
                return False
            
            await asyncio.sleep(check_interval)
    
    async def download_results(self, job_id: str, output_dir: Path) -> bool:
        """Download transcription results."""
        try:
            response = await self.http_client.get(
                f"{self.server_url}/download/{job_id}",
                headers=self.headers,
                timeout=120
            )
            
            if response.status_code == 200:
                # Save zip file
                output_dir.mkdir(parents=True, exist_ok=True)
                zip_path = output_dir / f"transcripts_{job_id}.zip"
                
                with open(zip_path, 'wb') as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                
                # Extract zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                
                # Remove zip file after extraction
                zip_path.unlink()
                
                return True
            else:
                return False
                
        except RequestError:
            return False
    
    async def cleanup_job(self, job_id: str):
        """Delete job from server."""
        try:
            await self.http_client.delete(
                f"{self.server_url}/job/{job_id}",
                headers=self.headers,
                timeout=10
            )
        except:
            pass  # Ignore cleanup errors
            print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
            
            if not audio_file.exists():
                print(f"❌ File not found: {audio_file}")
                failed_uploads += 1
                continue
            
            job_id = self.upload_file_with_progress(audio_file, model)
            
            if job_id:
                job_ids.append(job_id)
                successful_uploads += 1
                print(f"✅ Queued for processing")
            else:
                failed_uploads += 1
                print(f"❌ Upload failed")
        
        print(f"\n📊 Upload Summary:")
        print(f"   ✅ Successful: {successful_uploads}")
        print(f"   ❌ Failed: {failed_uploads}")
        print(f"   📋 Job IDs: {len(job_ids)}")
        
        return job_ids
    
    def start_processing(self, job_id: str, max_retries: int = 5) -> tuple[bool, bool]:
        """Start processing an uploaded file.
        
        Returns:
            (success: bool, job_already_completed: bool)
        """
        print(f"🚀 Starting processing for job {job_id[:8]}...")
        
        for attempt in range(max_retries):
            try:
                response = httpx.post(
                    f"{self.server_url}/process/{job_id}",
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Processing started! Status: {data.get('status')}")
                    if 'queue_size' in data:
                        print(f"📋 Queue position: {data['queue_size']}")
                    return True, False
                else:
                    # Parse error response to detect if job is already completed
                    error_detail = response.text  # Default fallback
                    is_status_error = False
                    
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', response.text)
                        
                        # Check if the error indicates job is already completed
                        if "status is 'completed'" in error_detail:
                            print(f"⚡ Job completed very quickly! Ready for download.")
                            return True, True
                        
                        # Check if job failed during processing
                        elif "status is 'failed'" in error_detail:
                            print(f"❌ Job failed during processing: {error_detail}")
                            return False, False
                        
                        # For other status-related errors, check actual status before retrying
                        elif "status is" in error_detail:
                            is_status_error = True
                            # Try to get the actual status
                            current_status = self.check_status(job_id)
                            if current_status:
                                status = current_status.get('status')
                                if status == 'completed':
                                    print(f"⚡ Job completed while starting processing! Ready for download.")
                                    return True, True
                                elif status == 'failed':
                                    print(f"❌ Job failed: {current_status.get('error', 'Unknown error')}")
                                    return False, False
                                elif status == 'processing':
                                    print(f"✅ Job is already processing!")
                                    return True, False
                        
                        print(f"❌ Failed to start processing (attempt {attempt + 1}/{max_retries}): {error_detail}")
                        
                    except (json.JSONDecodeError, KeyError):
                        print(f"❌ Failed to start processing (attempt {attempt + 1}/{max_retries}): {response.text}")
                    
                    # Only retry for actual errors, not status mismatches
                    if attempt < max_retries - 1 and not is_status_error:
                        print(f"⏳ Retrying in 3 seconds...")
                        time.sleep(3)
                    elif is_status_error:
                        # Don't retry status mismatches, they won't resolve
                        break
                        
            except RequestError as e:
                print(f"❌ Processing start error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in 3 seconds...")
                    time.sleep(3)
        
        print(f"❌ Failed to start processing for {job_id[:8]} after {max_retries} attempts")
        return False, False
    
    def check_status(self, job_id: str) -> Optional[dict]:
        """Check job status with CloudFlare-optimized timeout and request tracking."""
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        try:
            # Add request ID header for tracking
            headers_with_id = {**self.headers, "X-Request-ID": request_id}
            
            response = httpx.get(
                f"{self.server_url}/status/{job_id}",
                headers=headers_with_id,
                timeout=30  # Reduced from 60s for faster CloudFlare failures
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                # Only log on slow responses or occasional success for tracking
                if elapsed > 10 or random.random() < 0.1:  # Log 10% of fast responses
                    print(f"🔍 Status check {request_id}: {elapsed:.1f}s")
                return response.json()
            else:
                print(f"❌ Status check {request_id} failed ({elapsed:.1f}s): {response.text}")
                return None
                
        except httpx.ReadTimeout:
            elapsed = time.time() - start_time
            print(f"❌ Status check {request_id} timeout ({elapsed:.1f}s) - likely CloudFlare proxy issue")
            return None
        except httpx.ConnectTimeout:
            elapsed = time.time() - start_time
            print(f"❌ Status check {request_id} connection timeout ({elapsed:.1f}s) - CloudFlare connection issue")
            return None
        except RequestError as e:
            elapsed = time.time() - start_time
            print(f"❌ Status check {request_id} error ({elapsed:.1f}s): {e}")
            return None
    
    def wait_for_completion(self, job_id: str, check_interval: int = 5) -> bool:
        """Wait for job to complete with exponential backoff for CloudFlare timeouts."""
        print(f"\n⏳ Processing job {job_id}...")
        
        last_status = None
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        spinner_idx = 0
        consecutive_errors = 0
        max_consecutive_errors = 10
        base_retry_delay = 2  # Start with 2 seconds
        
        while True:
            status = self.check_status(job_id)
            
            if not status:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"\n❌ Too many consecutive network errors ({consecutive_errors}). Giving up.")
                    return False
                
                # Exponential backoff with jitter for CloudFlare issues
                retry_delay = min(base_retry_delay * (2 ** (consecutive_errors - 1)), 30)  # Cap at 30s
                jitter = random.uniform(0.5, 1.5)  # Add 50% jitter
                actual_delay = retry_delay * jitter
                
                print(f"\n⚠️  Network error ({consecutive_errors}/{max_consecutive_errors}), retrying in {actual_delay:.1f} seconds...")
                time.sleep(actual_delay)
                continue
            else:
                consecutive_errors = 0  # Reset error counter on successful response
            
            # Update progress display
            if status != last_status:
                processed = status.get('processed_count', 0)
                total = status.get('files_count', 0)
                failed = status.get('failed_count', 0)
                job_status = status.get('status', 'unknown')
                
                if job_status == 'uploaded':
                    print(f"\r📁 File uploaded, waiting to start processing...", end='', flush=True)
                elif job_status == 'queued':
                    print(f"\r📋 Queued for processing...", end='', flush=True)
                elif job_status == 'processing':
                    print(f"\r{spinner[spinner_idx]} Processing: {processed}/{total} files completed, {failed} failed", end='', flush=True)
                    spinner_idx = (spinner_idx + 1) % len(spinner)
                else:
                    print(f"\r{spinner[spinner_idx]} Status: {job_status}", end='', flush=True)
                    spinner_idx = (spinner_idx + 1) % len(spinner)
                
                last_status = status
            
            # Check completion
            if status.get('status') == 'completed':
                print(f"\n✅ Job completed! Processed: {status.get('processed_count')}, Failed: {status.get('failed_count')}")
                return True
            elif status.get('status') == 'failed':
                print(f"\n❌ Job failed: {status.get('error', 'Unknown error')}")
                return False
            
            time.sleep(check_interval)
    
    def wait_for_multiple_jobs(self, job_ids: List[str], check_interval: int = 5) -> List[str]:
        """Wait for multiple jobs to complete and return completed job IDs."""
        if not job_ids:
            return []
        
        print(f"\n⏳ Monitoring {len(job_ids)} processing jobs...")
        
        completed_jobs = []
        failed_jobs = []
        remaining_jobs = job_ids.copy()
        
        while remaining_jobs:
            for job_id in remaining_jobs.copy():
                status = self.check_status(job_id)
                
                if not status:
                    print(f"\n⚠️  Could not get status for job {job_id[:8]}...")
                    remaining_jobs.remove(job_id)
                    continue
                
                job_status = status.get('status', 'unknown')
                
                if job_status == 'completed':
                    completed_jobs.append(job_id)
                    remaining_jobs.remove(job_id)
                    print(f"\n✅ Job {job_id[:8]}... completed!")
                elif job_status == 'failed':
                    failed_jobs.append(job_id)
                    remaining_jobs.remove(job_id)
                    print(f"\n❌ Job {job_id[:8]}... failed: {status.get('error', 'Unknown error')}")
            
            if remaining_jobs:
                # Show progress for remaining jobs
                in_progress = len([j for j in remaining_jobs if self.check_status(j) and self.check_status(j).get('status') == 'processing'])
                pending = len(remaining_jobs) - in_progress
                
                print(f"\r🔄 Jobs remaining: {len(remaining_jobs)} (Processing: {in_progress}, Pending: {pending})", end='', flush=True)
                time.sleep(check_interval)
        
        print(f"\n\n📊 Final Results:")
        print(f"   ✅ Completed: {len(completed_jobs)}")
        print(f"   ❌ Failed: {len(failed_jobs)}")
        
        return completed_jobs
    
    def download_results(self, job_id: str, output_dir: Path) -> bool:
        """Download transcription results."""
        print(f"\n📥 Downloading results...")
        
        try:
            response = httpx.get(
                f"{self.server_url}/download/{job_id}",
                headers=self.headers,
                timeout=120  # Keep longer timeout for download due to file size
            )
            
            if response.status_code == 200:
                # Save zip file
                output_dir.mkdir(parents=True, exist_ok=True)
                zip_path = output_dir / f"transcripts_{job_id}.zip"
                
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                
                print(f"✅ Downloaded archive: {zip_path}")
                
                # Extract zip file
                print(f"📦 Extracting transcripts...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                
                # Remove zip file after extraction
                zip_path.unlink()
                
                # Count extracted files
                txt_files = list(output_dir.glob("*.txt"))
                print(f"✅ Extracted {len(txt_files)} transcript files to {output_dir}")
                
                return True
            else:
                print(f"❌ Download failed: {response.text}")
                return False
                
        except RequestError as e:
            print(f"❌ Download error: {e}")
            return False
    
    def cleanup_job(self, job_id: str):
        """Delete job from server."""
        try:
            response = httpx.delete(
                f"{self.server_url}/job/{job_id}",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                print(f"🗑️  Cleaned up job on server")
        except:
            pass  # Ignore cleanup errors

def check_existing_files(audio_files: List[Path], output_dir: Path) -> tuple[List[Path], List[Path]]:
    """Check which files already have transcripts and which need processing.
    
    Returns:
        (files_to_process, existing_files)
    """
    files_to_process = []
    existing_files = []
    
    for audio_file in audio_files:
        # Use same logic as server to determine output filename
        safe_name = slugify_filename(audio_file.stem)
        expected_output = output_dir / f"{safe_name}.txt"
        
        if expected_output.exists():
            existing_files.append(audio_file)
        else:
            files_to_process.append(audio_file)
    
    return files_to_process, existing_files

def find_audio_files(input_path: Path) -> List[Path]:
    """Find all audio files in directory or return single file."""
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
    
    if input_path.is_file():
        if input_path.suffix.lower() in audio_extensions:
            return [input_path]
        else:
            return []
    elif input_path.is_dir():
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f"*{ext}"))
            audio_files.extend(input_path.glob(f"*{ext.upper()}"))
        # Filter out hidden files (starting with '.')
        audio_files = [f for f in audio_files if not f.name.startswith('.')]
        return sorted(audio_files)
    else:
        return []

def main():
    """Main entry point."""
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Simple client for RunPod FastAPI Transcription Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe a directory of MP3s
  python transcribe_client.py ~/audio ~/transcripts
  
  # Transcribe with specific server
  python transcribe_client.py ~/audio ~/output --server https://your-pod-id.runpod.io:8080
  
  # Use custom API key
  python transcribe_client.py ~/audio ~/output --api-key your-secret-key
  
  # Use different model
  python transcribe_client.py ~/audio ~/output --model large-v3
        """
    )
    
    parser.add_argument("input", type=Path, help="Input directory containing audio files or single audio file")
    parser.add_argument("output", type=Path, help="Output directory for transcripts")
    parser.add_argument("--server", default="http://localhost:8080", help="Server URL (default: http://localhost:8080)")
    parser.add_argument("--api-key", default="your-secret-api-key-here", help="API key for authentication (overrides TRANSCRIBE_API_KEY from .env)")
    parser.add_argument("--model", default="turbo", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"],
                       help="Whisper model to use (default: turbo)")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't delete job from server after completion")
    parser.add_argument("--force", action="store_true", help="Process all files even if transcripts already exist")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"❌ Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Find audio files
    audio_files = find_audio_files(args.input)
    if not audio_files:
        print(f"❌ No audio files found in: {args.input}")
        print("   Supported formats: MP3, WAV, M4A, FLAC, OGG")
        sys.exit(1)
    
    # Check for existing transcripts (unless --force is used)
    if args.force:
        files_to_process = audio_files
        existing_files = []
    else:
        files_to_process, existing_files = check_existing_files(audio_files, args.output)
    
    # Print file status summary
    print(f"📊 File Status Summary:")
    print(f"   📁 Total audio files found: {len(audio_files)}")
    if existing_files:
        print(f"   ✅ Already transcribed: {len(existing_files)}")
        if len(existing_files) <= 5:  # Show details for small numbers
            for existing_file in existing_files:
                safe_name = slugify_filename(existing_file.stem)
                print(f"      - {existing_file.name} → {safe_name}.txt")
        else:
            print(f"      (Use --force to re-process existing files)")
    print(f"   🆕 New files to process: {len(files_to_process)}")
    
    if not files_to_process:
        print(f"\n✅ All files already transcribed! Use --force to re-process.")
        sys.exit(0)
    
    print(f"🎵 RunPod Transcription Client")
    print(f"{'=' * 50}")
    print(f"📁 Input:  {args.input}")
    print(f"📁 Output: {args.output}")
    print(f"🔧 Model:  {args.model}")
    print(f"🌐 Server: {args.server}")
    print(f"📊 Files:  {len(audio_files)} audio files found")
    print(f"{'=' * 50}")
    
    # Get server URL from environment or use default
    server_url = os.environ.get("RUNPOD_SERVER_URL", args.server)
    
    # Get API key: prioritize TRANSCRIBE_API_KEY from .env, then command line argument
    api_key = os.environ.get("TRANSCRIBE_API_KEY", args.api_key)
    
    # Initialize client
    client = TranscriptionClient(server_url, api_key)
    
    # Check server health
    print("\n🔍 Checking server connection...")
    if not client.health_check():
        print("\n❌ Cannot connect to server. Please check:")
        print(f"   - Server URL: {server_url}")
        print(f"   - API key: {'Set' if api_key != 'your-secret-api-key-here' else 'Using default'}")
        print(f"   - Is the RunPod container running?")
        sys.exit(1)
    
    # Process files one by one with new two-step workflow
    print(f"\n📤 Processing {len(files_to_process)} files sequentially...")
    completed_jobs = []
    failed_jobs = []
    skipped_jobs = len(existing_files)
    
    for i, audio_file in enumerate(files_to_process, 1):
        print(f"\n[{i}/{len(files_to_process)}] Processing: {audio_file.name}")
        
        if not audio_file.exists():
            print(f"❌ File not found: {audio_file}")
            failed_jobs.append(audio_file.name)
            continue
        
        # Step 1: Upload file
        job_id = client.upload_file_with_progress(audio_file, args.model)
        if not job_id:
            print(f"❌ Upload failed for {audio_file.name}")
            failed_jobs.append(audio_file.name)
            continue
        
        # Step 2: Start processing
        processing_started, job_already_completed = client.start_processing(job_id)
        if not processing_started:
            print(f"❌ Failed to start processing for {audio_file.name}")
            failed_jobs.append(audio_file.name)
            continue
        
        # Step 3: Wait for completion (skip if already completed)
        job_completed = job_already_completed
        if not job_already_completed:
            job_completed = client.wait_for_completion(job_id)
        
        if job_completed:
            if job_already_completed:
                print(f"⚡ Job completed instantly for {audio_file.name} (optimized server!)")
            else:
                print(f"✅ Processing completed for {audio_file.name}")
            
            completed_jobs.append(job_id)
            
            # Step 4: Download result immediately
            print(f"📥 Downloading result for {audio_file.name}...")
            if client.download_results(job_id, args.output):
                print(f"✅ Downloaded transcript for {audio_file.name}")
            else:
                print(f"⚠️  Failed to download transcript for {audio_file.name}")
            
            # Step 5: Cleanup
            if not args.no_cleanup:
                client.cleanup_job(job_id)
        else:
            print(f"❌ Processing failed for {audio_file.name}")
            failed_jobs.append(audio_file.name)
    
    # Summary
    print(f"\n📊 Processing Summary:")
    print(f"   ✅ Completed: {len(completed_jobs)}")
    print(f"   ❌ Failed: {len(failed_jobs)}")
    if skipped_jobs > 0:
        print(f"   ⏭️  Skipped (already exist): {skipped_jobs}")
    
    if not completed_jobs and not skipped_jobs:
        print("❌ No files processed successfully")
        sys.exit(1)
    
    print(f"\n🎉 Transcription complete!")
    print(f"📊 Total files: {len(audio_files)}")
    print(f"✅ Successfully completed: {len(completed_jobs)}")
    print(f"❌ Failed: {len(failed_jobs)}")
    if skipped_jobs > 0:
        print(f"⏭️  Skipped (already exist): {skipped_jobs}")
    print(f"📁 Results saved to: {args.output}")

if __name__ == "__main__":
    main()