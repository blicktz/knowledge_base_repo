#!/usr/bin/env python3
"""
Async Parallel Client for RunPod FastAPI Transcription Server
Processes up to 6 files concurrently for maximum server utilization.
Usage: python transcribe_client_async.py /path/to/audio/files /path/to/output [--server URL] [--api-key KEY]
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
        
        # Clear previous lines (simple approach)
        print('\r' + ' ' * 120 + '\r', end='')  # Clear line
        
        # Main progress line
        progress_pct = (self.completed / self.total_files * 100) if self.total_files > 0 else 0
        active_count = len(self.active_jobs)
        
        status_line = (f"📊 {self.completed}/{self.total_files} completed ({progress_pct:.1f}%), "
                      f"{self.failed} failed, {active_count} active | {elapsed:.0f}s")
        
        print(f"\r{status_line}", end='')
        
        # Show active jobs details on new lines (limit to 3)
        if self.active_jobs:
            active_list = list(self.active_jobs.values())[:3]
            for job_info in active_list:
                file_name = job_info['file'][:40] + '...' if len(job_info['file']) > 40 else job_info['file']
                status_emoji = {
                    'uploading': '📤',
                    'starting': '🚀', 
                    'processing': '⚙️',
                    'downloading': '📥'
                }.get(job_info['status'], '🔄')
                print(f"\n   {status_emoji} {file_name}: {job_info['status']}", end='')
            
            if len(self.active_jobs) > 3:
                print(f"\n   ... and {len(self.active_jobs) - 3} more files", end='')
        
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

# Core async processing functions
async def process_single_file(client: TranscriptionClient, audio_file: Path, 
                            output_dir: Path, model: str, progress_mgr: ProgressManager,
                            no_cleanup: bool = False) -> bool:
    """Process a single file with progress tracking."""
    job_id = None
    file_name = audio_file.name
    
    try:
        # Upload with progress
        await progress_mgr.update_status("temp", file_name, "uploading")
        job_id = await client.upload_file(audio_file, model)
        if not job_id:
            raise Exception("Upload failed")
        
        # Start processing
        await progress_mgr.update_status(job_id, file_name, "starting")
        processing_started, job_already_completed = await client.start_processing(job_id)
        if not processing_started:
            raise Exception("Failed to start processing")
        
        # Wait for completion (skip if already completed)
        if not job_already_completed:
            await progress_mgr.update_status(job_id, file_name, "processing")
            if not await client.wait_for_completion(job_id):
                raise Exception("Processing failed")
        
        # Download results
        await progress_mgr.update_status(job_id, file_name, "downloading")
        if not await client.download_results(job_id, output_dir):
            raise Exception("Download failed")
        
        # Cleanup (optional)
        if not no_cleanup:
            await client.cleanup_job(job_id)
        
        await progress_mgr.mark_completed(job_id, success=True)
        return True
        
    except Exception as e:
        if job_id:
            await progress_mgr.mark_completed(job_id, success=False)
        else:
            await progress_mgr.mark_completed(f"failed-{file_name}", success=False)
        return False

async def process_files_concurrently(client: TranscriptionClient, files: List[Path], 
                                   output_dir: Path, model: str, max_concurrent: int = 6,
                                   no_cleanup: bool = False) -> List[bool]:
    """Process all files with up to N concurrent workers."""
    progress_mgr = ProgressManager(len(files))
    
    # Create semaphore to limit concurrent workers
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_process(file):
        async with semaphore:
            return await process_single_file(client, file, output_dir, model, progress_mgr, no_cleanup)
    
    # Process all files concurrently (bounded by semaphore)
    print(f"\n🚀 Processing {len(files)} files with up to {max_concurrent} concurrent workers...")
    results = await asyncio.gather(*[bounded_process(f) for f in files], return_exceptions=True)
    
    # Convert exceptions to False
    results = [r if isinstance(r, bool) else False for r in results]
    
    progress_mgr.print_final_summary()
    return results

def check_existing_files(audio_files: List[Path], output_dir: Path) -> tuple[List[Path], List[Path]]:
    """Check which files already have transcripts and which need processing."""
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

async def main():
    """Main async entry point."""
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Async Parallel Client for RunPod FastAPI Transcription Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe a directory of MP3s (up to 6 files concurrently)
  python transcribe_client_async.py ~/audio ~/transcripts
  
  # Transcribe with specific server
  python transcribe_client_async.py ~/audio ~/output --server https://your-pod-id.runpod.io:8080
  
  # Use custom API key and model
  python transcribe_client_async.py ~/audio ~/output --api-key your-secret-key --model large-v3
  
  # Limit concurrent workers
  python transcribe_client_async.py ~/audio ~/output --max-concurrent 3
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
    parser.add_argument("--max-concurrent", type=int, default=6, help="Maximum concurrent workers (default: 6)")
    
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
        return
    
    print(f"🎵 RunPod Async Transcription Client")
    print(f"{'=' * 50}")
    print(f"📁 Input:  {args.input}")
    print(f"📁 Output: {args.output}")
    print(f"🔧 Model:  {args.model}")
    print(f"🌐 Server: {args.server}")
    print(f"📊 Files:  {len(files_to_process)} files to process")
    print(f"⚡ Max concurrent: {args.max_concurrent}")
    print(f"{'=' * 50}")
    
    # Get server URL from environment or use default
    server_url = os.environ.get("RUNPOD_SERVER_URL", args.server)
    
    # Get API key: prioritize TRANSCRIBE_API_KEY from .env, then command line argument
    api_key = os.environ.get("TRANSCRIBE_API_KEY", args.api_key)
    
    # Create async HTTP client and transcription client
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=60.0, read=1800.0, write=60.0, pool=60.0)) as http_client:
        client = TranscriptionClient(server_url, api_key, http_client)
        
        # Check server health
        print("\n🔍 Checking server connection...")
        if not await client.health_check():
            print("\n❌ Cannot connect to server. Please check:")
            print(f"   - Server URL: {server_url}")
            print(f"   - API key: {'Set' if api_key != 'your-secret-api-key-here' else 'Using default'}")
            print(f"   - Is the RunPod container running?")
            sys.exit(1)
        
        # Process files concurrently
        results = await process_files_concurrently(
            client, files_to_process, args.output, args.model, 
            args.max_concurrent, args.no_cleanup
        )
        
        # Final summary
        successful = sum(1 for r in results if r)
        failed = len(results) - successful
        skipped = len(existing_files)
        
        print(f"\n🎉 Transcription complete!")
        print(f"📊 Total files: {len(audio_files)}")
        print(f"✅ Successfully completed: {successful}")
        print(f"❌ Failed: {failed}")
        if skipped > 0:
            print(f"⏭️  Skipped (already exist): {skipped}")
        print(f"📁 Results saved to: {args.output}")

if __name__ == "__main__":
    asyncio.run(main())