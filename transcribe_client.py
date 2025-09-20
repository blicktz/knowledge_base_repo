#!/usr/bin/env python3
"""
Simple client for RunPod FastAPI Transcription Server
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
from pathlib import Path
from typing import List, Optional
import httpx
from httpx import RequestError
from tqdm import tqdm
from dotenv import load_dotenv

class TranscriptionClient:
    """Client for RunPod FastAPI transcription server."""
    
    def __init__(self, server_url: str, api_key: str):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Connection": "keep-alive",
            "User-Agent": "transcription-client/1.0"
        }
    
    def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            response = httpx.get(f"{self.server_url}/health", headers=self.headers, timeout=30)
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
    
    def upload_file_with_progress(self, audio_file: Path, model: str = "turbo", max_retries: int = 5) -> Optional[str]:
        """Upload a single audio file with progress tracking and retry logic."""
        file_size = audio_file.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"📤 Uploading: {audio_file.name} ({file_size_mb:.1f} MB)")
        
        for attempt in range(max_retries):
            try:
                # Create progress bar
                progress_bar = tqdm(
                    total=file_size,
                    unit='B',
                    unit_scale=True,
                    desc=f"Upload {audio_file.name}",
                    leave=False
                )
                
                # Use httpx native file upload (simpler but with basic progress)
                with open(audio_file, 'rb') as f:
                    files = {'file': (audio_file.name, f, 'audio/mpeg')}
                    data = {'model': model}
                    
                    # Start upload indicator
                    progress_bar.update(0)
                    
                    response = httpx.post(
                        f"{self.server_url}/upload-single",
                        files=files,
                        data=data,
                        headers=self.headers,
                        timeout=1800  # 30 minutes timeout for large files
                    )
                    
                    # Complete progress bar (we can't track real-time progress with basic httpx)
                    progress_bar.update(file_size)
                
                progress_bar.close()
                
                if response.status_code == 200:
                    data = response.json()
                    job_id = data.get('job_id')
                    print(f"✅ Upload successful! Job ID: {job_id}")
                    return job_id
                else:
                    print(f"❌ Upload failed (attempt {attempt + 1}/{max_retries}): {response.text}")
                    if attempt < max_retries - 1:
                        print(f"⏳ Retrying in 5 seconds...")
                        time.sleep(5)
                    
            except RequestError as e:
                progress_bar.close() if 'progress_bar' in locals() else None
                print(f"❌ Upload error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in 5 seconds...")
                    time.sleep(5)
        
        print(f"❌ Failed to upload {audio_file.name} after {max_retries} attempts")
        return None
    
    def upload_files_individually(self, audio_files: List[Path], model: str = "turbo") -> List[str]:
        """Upload multiple files one by one with progress tracking."""
        print(f"\n📤 Uploading {len(audio_files)} files individually...")
        
        job_ids = []
        successful_uploads = 0
        failed_uploads = 0
        
        for i, audio_file in enumerate(audio_files, 1):
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
    
    def start_processing(self, job_id: str, max_retries: int = 5) -> bool:
        """Start processing an uploaded file."""
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
                    return True
                else:
                    print(f"❌ Failed to start processing (attempt {attempt + 1}/{max_retries}): {response.text}")
                    if attempt < max_retries - 1:
                        print(f"⏳ Retrying in 3 seconds...")
                        time.sleep(3)
                        
            except RequestError as e:
                print(f"❌ Processing start error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in 3 seconds...")
                    time.sleep(3)
        
        print(f"❌ Failed to start processing for {job_id[:8]} after {max_retries} attempts")
        return False
    
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
    print(f"\n📤 Processing {len(audio_files)} files sequentially...")
    completed_jobs = []
    failed_jobs = []
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
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
        if not client.start_processing(job_id):
            print(f"❌ Failed to start processing for {audio_file.name}")
            failed_jobs.append(audio_file.name)
            continue
        
        # Step 3: Wait for completion
        if client.wait_for_completion(job_id):
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
    
    if not completed_jobs:
        print("❌ No files processed successfully")
        sys.exit(1)
    
    print(f"\n🎉 Transcription complete!")
    print(f"📊 Total files: {len(audio_files)}")
    print(f"✅ Successfully completed: {len(completed_jobs)}")
    print(f"❌ Failed: {len(failed_jobs)}")
    print(f"📁 Results saved to: {args.output}")

if __name__ == "__main__":
    main()