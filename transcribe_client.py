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
import logging
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncGenerator
import httpx
from httpx import RequestError, ConnectError, TimeoutException
import json
from dotenv import load_dotenv

# Global logger (will be configured in setup_logging)
logger = None

def setup_logging(debug: bool):
    """Configure logging based on debug flag."""
    global logger
    level = logging.DEBUG if debug else logging.WARNING
    format_str = '%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    
    # Also configure httpx logging if in debug mode
    if debug:
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.INFO)
    
    return logger

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
        self.failed_files = {}  # {file_name: error_message} - track why files failed
        self.lock = asyncio.Lock()
        self.start_time = time.time()
    
    async def update_status(self, job_id: str, file_name: str, status: str):
        """Update status for a specific job."""
        async with self.lock:
            self.active_jobs[job_id] = {"file": file_name, "status": status}
            self._display_progress()
    
    async def mark_completed(self, job_id: str, success: bool = True, error_msg: str = None):
        """Mark a job as completed (success or failure)."""
        async with self.lock:
            # Get file name for error tracking
            file_name = None
            if job_id in self.active_jobs:
                file_name = self.active_jobs[job_id].get("file", "Unknown file")
            
            if success:
                self.completed += 1
            else:
                self.failed += 1
                # Track why this file failed
                if file_name and error_msg:
                    self.failed_files[file_name] = error_msg
                    logger.error(f"File '{file_name}' failed: {error_msg}")
            
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
                    'chunked_uploading': '📦',
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
        
        # Print details about failed files
        if self.failed_files:
            print(f"\n❌ Failed Files Details:")
            for file_name, error in self.failed_files.items():
                # Truncate filename if too long
                display_name = file_name[:50] + '...' if len(file_name) > 50 else file_name
                print(f"   • {display_name}")
                print(f"     Error: {error}")

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
        
        # Multi-chunk upload configuration
        self.chunk_size = 15 * 1024 * 1024  # 15MB chunks
        self.large_file_threshold = 30 * 1024 * 1024  # 30MB threshold
    
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
    
    async def upload_file(self, audio_file: Path, model: str = "turbo", max_retries: int = 5) -> Optional[str]:
        """Upload a single audio file using chunked streaming."""
        file_size_mb = audio_file.stat().st_size / (1024 * 1024)
        logger.info(f"Starting streaming upload: {audio_file.name} ({file_size_mb:.1f} MB)")
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Streaming upload attempt {attempt + 1}/{max_retries} for {audio_file.name}")
                
                # Create multipart form data manually for streaming
                boundary = f"----formdata-httpx-{uuid.uuid4().hex}"
                
                async def generate_multipart() -> AsyncGenerator[bytes, None]:
                    # Model field
                    yield f'--{boundary}\r\n'.encode()
                    yield f'Content-Disposition: form-data; name="model"\r\n\r\n'.encode()
                    yield f'{model}\r\n'.encode()
                    
                    # File field header
                    yield f'--{boundary}\r\n'.encode()
                    yield f'Content-Disposition: form-data; name="file"; filename="{audio_file.name}"\r\n'.encode()
                    yield f'Content-Type: audio/mpeg\r\n\r\n'.encode()
                    
                    # Stream file content in chunks
                    chunk_size = 1024 * 1024  # 1MB chunks
                    bytes_sent = 0
                    with open(audio_file, 'rb') as f:
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            yield chunk
                            bytes_sent += len(chunk)
                            
                            # Log progress for large files (less frequent)
                            if bytes_sent % (20 * 1024 * 1024) == 0:  # Every 20MB
                                progress_mb = bytes_sent / (1024 * 1024)
                                logger.info(f"Upload progress: {progress_mb:.1f}MB of {file_size_mb:.1f}MB ({progress_mb/file_size_mb*100:.1f}%)")
                    
                    # End boundary
                    yield f'\r\n--{boundary}--\r\n'.encode()
                
                headers = {
                    **self.headers,
                    'Content-Type': f'multipart/form-data; boundary={boundary}'
                }
                
                upload_start = time.time()
                response = await self.http_client.post(
                    f"{self.server_url}/upload-single",
                    content=generate_multipart(),
                    headers=headers,
                    timeout=1800  # 30 minutes timeout
                )
                upload_duration = time.time() - upload_start
                
                logger.debug(f"Streaming upload response: status={response.status_code}, duration={upload_duration:.1f}s")
                
                if response.status_code == 200:
                    data = response.json()
                    job_id = data.get('job_id')
                    total_duration = time.time() - start_time
                    logger.info(f"Streaming upload successful: {audio_file.name} -> job_id={job_id} (took {total_duration:.1f}s)")
                    return job_id
                    
                elif response.status_code == 499:  # Client disconnect
                    logger.warning(f"Client disconnect during upload (attempt {attempt + 1}/{max_retries}): {audio_file.name}")
                    if attempt < max_retries - 1:
                        backoff = min(2 ** (attempt + 1), 30)
                        logger.info(f"Retrying after disconnect in {backoff}s...")
                        await asyncio.sleep(backoff)
                        
                elif response.status_code == 503:  # Server overloaded/under pressure
                    error_detail = "Server overloaded"
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', 'Server overloaded')
                    except:
                        pass
                    logger.warning(f"Server overloaded (503) - {error_detail}: {audio_file.name}")
                    if attempt < max_retries - 1:
                        backoff = min(5 * (2 ** attempt), 60)  # Longer backoff for overload
                        logger.info(f"Server overloaded, retrying after {backoff}s...")
                        await asyncio.sleep(backoff)
                        
                elif response.status_code >= 500:  # Server errors
                    error_detail = f"HTTP {response.status_code}"
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', response.text[:500])
                    except:
                        error_detail = response.text[:500] if response.text else f"HTTP {response.status_code}"
                    logger.warning(f"Server error during upload (attempt {attempt + 1}/{max_retries}): {audio_file.name} - {error_detail}")
                    if attempt < max_retries - 1:
                        backoff = min(3 * (2 ** attempt), 45)
                        logger.info(f"Retrying after server error in {backoff}s...")
                        await asyncio.sleep(backoff)
                        
                else:  # Client errors (4xx)
                    error_detail = f"HTTP {response.status_code}"
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', response.text)
                    except:
                        error_detail = response.text[:500] if response.text else f"HTTP {response.status_code}"
                    
                    logger.warning(f"Upload failed (attempt {attempt + 1}/{max_retries}): {audio_file.name} - {error_detail}")
                    
                    if attempt < max_retries - 1 and response.status_code != 400:  # Don't retry bad requests
                        backoff = min(2 ** attempt, 30)
                        jitter = random.uniform(0, backoff * 0.1) 
                        sleep_time = backoff + jitter
                        logger.debug(f"Retrying upload in {sleep_time:.1f} seconds...")
                        await asyncio.sleep(sleep_time)
                    
            except (ConnectError, TimeoutException) as e:
                error_msg = str(e)
                logger.error(f"Connection/timeout error (attempt {attempt + 1}/{max_retries}): {audio_file.name} - {error_msg}")
                
                if attempt < max_retries - 1:
                    backoff = min(5 * (2 ** attempt), 60)
                    jitter = random.uniform(0, backoff * 0.2)
                    sleep_time = backoff + jitter
                    logger.info(f"Retrying after connection error in {sleep_time:.1f}s...")
                    await asyncio.sleep(sleep_time)
            except RequestError as e:
                error_msg = str(e)
                logger.error(f"Streaming upload network error (attempt {attempt + 1}/{max_retries}): {audio_file.name} - {error_msg}")
                
                if attempt < max_retries - 1:
                    backoff = min(3 * (2 ** attempt), 45)
                    jitter = random.uniform(0, backoff * 0.15)
                    sleep_time = backoff + jitter
                    logger.debug(f"Retrying after streaming network error in {sleep_time:.1f} seconds...")
                    await asyncio.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Unexpected streaming upload error: {audio_file.name} - {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                if attempt < max_retries - 1:
                    backoff = min(2 ** attempt, 30)
                    await asyncio.sleep(backoff)
        
        total_duration = time.time() - start_time
        logger.error(f"Streaming upload failed after {max_retries} attempts: {audio_file.name} (total time: {total_duration:.1f}s)")
        return None
    
    async def upload_file_multi_chunk(self, audio_file: Path, model: str = "turbo", max_retries: int = 3) -> Optional[str]:
        """Upload a large file using multi-chunk upload strategy."""
        file_size = audio_file.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"Starting multi-chunk upload: {audio_file.name} ({file_size_mb:.1f} MB)")
        start_time = time.time()
        
        try:
            # Step 1: Start upload session
            logger.debug(f"Starting upload session for {audio_file.name}")
            
            upload_data = {
                "filename": audio_file.name,
                "total_size": file_size,
                "chunk_size": self.chunk_size,
                "model": model
            }
            
            response = await self.http_client.post(
                f"{self.server_url}/upload-start",
                json=upload_data,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to start upload session: {response.status_code} - {response.text}")
                return None
            
            session_data = response.json()
            upload_id = session_data["upload_id"]
            total_chunks = session_data["total_chunks"]
            
            logger.info(f"Upload session started: {upload_id} ({total_chunks} chunks)")
            
            # Step 2: Upload chunks
            successful_chunks = 0
            
            with open(audio_file, 'rb') as f:
                for chunk_index in range(total_chunks):
                    chunk_start = chunk_index * self.chunk_size
                    f.seek(chunk_start)
                    chunk_data = f.read(self.chunk_size)
                    
                    if not chunk_data:
                        break
                    
                    # Upload this chunk with retries
                    chunk_success = False
                    for attempt in range(max_retries):
                        try:
                            logger.debug(f"Uploading chunk {chunk_index + 1}/{total_chunks} (attempt {attempt + 1})")
                            
                            chunk_response = await self.http_client.put(
                                f"{self.server_url}/upload-chunk/{upload_id}/{chunk_index}",
                                content=chunk_data,
                                headers=self.headers,
                                timeout=90  # 90 second timeout for chunk upload
                            )
                            
                            if chunk_response.status_code == 200:
                                successful_chunks += 1
                                chunk_success = True
                                logger.debug(f"Chunk {chunk_index + 1}/{total_chunks} uploaded successfully")
                                break
                            else:
                                logger.warning(f"Chunk upload failed (attempt {attempt + 1}): {chunk_response.status_code}")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                    
                        except Exception as e:
                            logger.warning(f"Chunk upload error (attempt {attempt + 1}): {str(e)}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                    
                    if not chunk_success:
                        logger.error(f"Failed to upload chunk {chunk_index + 1} after {max_retries} attempts")
                        return None
            
            logger.info(f"All {total_chunks} chunks uploaded successfully")
            
            # Step 3: Complete upload
            logger.debug(f"Completing upload session: {upload_id}")
            
            complete_response = await self.http_client.post(
                f"{self.server_url}/upload-complete/{upload_id}",
                headers=self.headers,
                timeout=60
            )
            
            if complete_response.status_code != 200:
                logger.error(f"Failed to complete upload: {complete_response.status_code} - {complete_response.text}")
                return None
            
            completion_data = complete_response.json()
            job_id = completion_data["job_id"]
            
            total_duration = time.time() - start_time
            logger.info(f"Multi-chunk upload completed: {audio_file.name} → job_id={job_id} (took {total_duration:.1f}s)")
            
            return job_id
            
        except Exception as e:
            total_duration = time.time() - start_time
            logger.error(f"Multi-chunk upload failed: {audio_file.name} - {str(e)} (total time: {total_duration:.1f}s)")
            return None
    
    async def upload_file_smart(self, audio_file: Path, model: str = "turbo") -> Optional[str]:
        """Smart upload that chooses single-chunk or multi-chunk based on file size."""
        file_size = audio_file.stat().st_size
        
        if file_size > self.large_file_threshold:
            logger.debug(f"Using multi-chunk upload for large file: {audio_file.name} ({file_size / (1024*1024):.1f} MB)")
            return await self.upload_file_multi_chunk(audio_file, model)
        else:
            logger.debug(f"Using single-chunk upload for small file: {audio_file.name} ({file_size / (1024*1024):.1f} MB)")
            return await self.upload_file(audio_file, model)
    
    async def start_processing(self, job_id: str, max_retries: int = 5) -> tuple[bool, bool]:
        """Start processing an uploaded file."""
        logger.info(f"Starting processing for job_id={job_id}")
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Process start attempt {attempt + 1}/{max_retries} for job_id={job_id}")
                
                response = await self.http_client.post(
                    f"{self.server_url}/process/{job_id}",
                    headers=self.headers,
                    timeout=30
                )
                
                logger.debug(f"Process start response: status={response.status_code} for job_id={job_id}")
                
                if response.status_code == 200:
                    # Check if this is a new job being queued or an idempotent response
                    response_data = response.json()
                    message = response_data.get('message', '')
                    if 'already' in message:
                        logger.info(f"Job already in progress: job_id={job_id}, status={response_data.get('status')}")
                        return True, False  # Already processing/queued
                    else:
                        logger.info(f"Processing started successfully for job_id={job_id}")
                        return True, False  # Newly queued
                        
                elif response.status_code == 409:
                    # Handle conflict responses (job already completed/failed)
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', response.text)
                        
                        if "already been completed" in error_detail:
                            logger.info(f"Job already completed: job_id={job_id}")
                            return True, True  # Already completed
                        elif "has failed" in error_detail:
                            logger.error(f"Job already failed: job_id={job_id} - {error_detail}")
                            return False, False  # Already failed
                        else:
                            logger.warning(f"Unexpected 409 conflict: job_id={job_id} - {error_detail}")
                            return False, False
                            
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Failed to parse 409 response: {e}")
                        logger.debug(f"Raw 409 response: {response.text[:500]}")
                        return False, False
                        
                elif response.status_code == 404:
                    # Job not found
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', 'Job not found')
                    except:
                        error_detail = 'Job not found'
                    logger.error(f"Job not found: job_id={job_id} - {error_detail}")
                    return False, False  # Don't retry for 404
                    
                elif response.status_code == 401:
                    # Authentication failed
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', 'Authentication failed')
                    except:
                        error_detail = 'Authentication failed'
                    logger.error(f"Authentication failed for process start: job_id={job_id} - {error_detail}")
                    return False, False  # Don't retry for auth errors
                    
                elif response.status_code == 403:
                    # Access forbidden
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', 'Access forbidden')
                    except:
                        error_detail = 'Access forbidden'
                    logger.error(f"Access forbidden for process start: job_id={job_id} - {error_detail}")
                    return False, False  # Don't retry for permission errors
                    
                else:
                    # Handle other error responses
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', response.text)
                    except:
                        error_detail = response.text[:500] if response.text else f"HTTP {response.status_code}"
                    
                    logger.warning(f"Process start failed: job_id={job_id}, status={response.status_code}, detail={error_detail}")
                    
                    # Only retry for server errors (5xx) and some client errors
                    if response.status_code >= 500 or response.status_code in [408, 429]:  # Server errors, timeout, rate limit
                        if attempt < max_retries - 1:
                            backoff = min(2 ** attempt, 30)
                            logger.debug(f"Retrying process start in {backoff} seconds...")
                            await asyncio.sleep(backoff)
                        continue
                    else:
                        # Don't retry other 4xx errors
                        logger.error(f"Non-retriable error for job_id={job_id}: {response.status_code}")
                        break
                        
            except (RequestError, TimeoutException) as e:
                error_type = "timeout" if isinstance(e, TimeoutException) else "network"
                logger.error(f"{error_type.title()} error starting process (attempt {attempt + 1}/{max_retries}): job_id={job_id} - {str(e)}")
                
                if attempt < max_retries - 1:
                    # Check job status before retry to handle idempotency
                    # This prevents duplicate processing when server processed request but response timed out
                    logger.debug(f"Checking job status before retry due to {error_type} error: job_id={job_id}")
                    try:
                        status = await self.check_status(job_id)
                        if status:
                            current_status = status.get('status')
                            logger.info(f"Job status after {error_type} error: job_id={job_id}, status={current_status}")
                            
                            if current_status == 'queued':
                                logger.info(f"Job already queued after {error_type} error: job_id={job_id}")
                                return True, False  # Already queued
                            elif current_status == 'processing':
                                logger.info(f"Job already processing after {error_type} error: job_id={job_id}")
                                return True, False  # Already processing
                            elif current_status == 'completed':
                                logger.info(f"Job already completed after {error_type} error: job_id={job_id}")
                                return True, True  # Already completed
                            elif current_status == 'failed':
                                logger.error(f"Job already failed after {error_type} error: job_id={job_id}")
                                return False, False  # Already failed
                            # If status is still 'uploaded', continue with retry
                            
                    except Exception as status_check_error:
                        logger.warning(f"Could not check status after {error_type} error: {status_check_error}")
                        # Continue with retry if status check fails
                    
                    backoff = min(3 * (2 ** attempt), 45)
                    jitter = random.uniform(0, backoff * 0.1)
                    sleep_time = backoff + jitter
                    logger.debug(f"Retrying after {error_type} error in {sleep_time:.1f} seconds...")
                    await asyncio.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Unexpected error starting process: job_id={job_id} - {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                if attempt < max_retries - 1:
                    backoff = min(2 ** attempt, 30)
                    await asyncio.sleep(backoff)
        
        total_duration = time.time() - start_time
        logger.error(f"Failed to start processing after {max_retries} attempts: job_id={job_id} (total time: {total_duration:.1f}s)")
        return False, False
    
    async def check_status(self, job_id: str, max_retries: int = 3) -> Optional[dict]:
        """Check job status with retry logic for temporary failures."""
        for attempt in range(max_retries):
            try:
                response = await self.http_client.get(
                    f"{self.server_url}/status/{job_id}",
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    status_data = response.json()
                    logger.debug(f"Status check: job_id={job_id}, status={status_data.get('status')}")
                    return status_data
                    
                elif response.status_code == 404:
                    logger.error(f"Job not found: job_id={job_id}")
                    return None  # Don't retry for 404
                    
                elif response.status_code == 401:
                    logger.error(f"Authentication failed for status check: job_id={job_id}")
                    return None  # Don't retry for auth errors
                    
                elif response.status_code == 403:
                    logger.error(f"Access forbidden for status check: job_id={job_id}")
                    return None  # Don't retry for permission errors
                    
                elif response.status_code >= 500 or response.status_code in [408, 429]:
                    # Server errors or rate limiting - retry
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', f"HTTP {response.status_code}")
                    except:
                        error_detail = f"HTTP {response.status_code}"
                        
                    logger.warning(f"Temporary error checking status (attempt {attempt + 1}/{max_retries}): job_id={job_id} - {error_detail}")
                    
                    if attempt < max_retries - 1:
                        backoff = min(2 ** attempt, 15)  # Shorter backoff for status checks
                        logger.debug(f"Retrying status check in {backoff} seconds...")
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        logger.error(f"Status check failed after {max_retries} attempts: job_id={job_id}")
                        return None
                        
                else:
                    # Other 4xx errors
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', f"HTTP {response.status_code}")
                    except:
                        error_detail = f"HTTP {response.status_code}"
                        
                    logger.warning(f"Status check error: job_id={job_id} - {error_detail}")
                    return None  # Don't retry for other 4xx errors
                    
            except (RequestError, TimeoutException) as e:
                error_type = "timeout" if isinstance(e, TimeoutException) else "network"
                logger.warning(f"{error_type.title()} error checking status (attempt {attempt + 1}/{max_retries}): job_id={job_id} - {str(e)}")
                
                if attempt < max_retries - 1:
                    backoff = min(2 ** attempt, 15)
                    logger.debug(f"Retrying status check after {error_type} error in {backoff} seconds...")
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"Status check failed after {max_retries} {error_type} errors: job_id={job_id}")
                    return None
                    
            except Exception as e:
                logger.error(f"Unexpected error checking status: job_id={job_id} - {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                return None
    
    async def wait_for_completion(self, job_id: str, check_interval: int = 3) -> bool:
        """Wait for job to complete."""
        logger.info(f"Waiting for completion: job_id={job_id}")
        start_time = time.time()
        check_count = 0
        
        while True:
            check_count += 1
            status = await self.check_status(job_id)
            
            if not status:
                logger.debug(f"Status check {check_count} returned None for job_id={job_id}, retrying...")
                await asyncio.sleep(check_interval)
                continue
            
            current_status = status.get('status')
            elapsed = time.time() - start_time
            
            if current_status == 'completed':
                logger.info(f"Job completed: job_id={job_id} (waited {elapsed:.1f}s, {check_count} checks)")
                return True
            elif current_status == 'failed':
                error_msg = status.get('error', 'No error message provided')
                logger.error(f"Job failed: job_id={job_id}, error={error_msg} (after {elapsed:.1f}s)")
                return False
            
            logger.debug(f"Job still processing: job_id={job_id}, status={current_status}, elapsed={elapsed:.1f}s")
            await asyncio.sleep(check_interval)
    
    async def download_results(self, job_id: str, output_dir: Path, max_retries: int = 3) -> bool:
        """Download transcription results with retry logic for temporary failures."""
        logger.info(f"Downloading results: job_id={job_id}")
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                response = await self.http_client.get(
                    f"{self.server_url}/download/{job_id}",
                    headers=self.headers,
                    timeout=120
                )
                
                logger.debug(f"Download response: status={response.status_code} for job_id={job_id} (attempt {attempt + 1})")
                
                if response.status_code == 200:
                    # Save zip file
                    output_dir.mkdir(parents=True, exist_ok=True)
                    zip_path = output_dir / f"transcripts_{job_id}.zip"
                    
                    # Download with progress tracking
                    bytes_downloaded = 0
                    with open(zip_path, 'wb') as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            f.write(chunk)
                            bytes_downloaded += len(chunk)
                    
                    download_duration = time.time() - start_time
                    size_mb = bytes_downloaded / (1024 * 1024)
                    logger.info(f"Downloaded {size_mb:.1f} MB in {download_duration:.1f}s for job_id={job_id}")
                    
                    # Extract zip file
                    logger.debug(f"Extracting results to {output_dir}")
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            extracted_files = zip_ref.namelist()
                            zip_ref.extractall(output_dir)
                            logger.debug(f"Extracted {len(extracted_files)} files: {extracted_files}")
                    except Exception as e:
                        logger.error(f"Failed to extract zip: job_id={job_id} - {str(e)}")
                        return False
                    
                    # Remove zip file after extraction
                    zip_path.unlink()
                    
                    total_duration = time.time() - start_time
                    logger.info(f"Download complete: job_id={job_id} (total time: {total_duration:.1f}s)")
                    return True
                    
                elif response.status_code == 404:
                    # Job or results not found
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', 'Results not found')
                    except:
                        error_detail = 'Results not found'
                    logger.error(f"Results not found: job_id={job_id} - {error_detail}")
                    return False  # Don't retry for 404
                    
                elif response.status_code == 401:
                    # Authentication failed
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', 'Authentication failed')
                    except:
                        error_detail = 'Authentication failed'
                    logger.error(f"Authentication failed for download: job_id={job_id} - {error_detail}")
                    return False  # Don't retry for auth errors
                    
                elif response.status_code == 403:
                    # Access forbidden
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', 'Access forbidden')
                    except:
                        error_detail = 'Access forbidden'
                    logger.error(f"Access forbidden for download: job_id={job_id} - {error_detail}")
                    return False  # Don't retry for permission errors
                    
                elif response.status_code >= 500 or response.status_code in [408, 429]:
                    # Server errors or rate limiting - retry
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', f"HTTP {response.status_code}")
                    except:
                        error_detail = f"HTTP {response.status_code}"
                        
                    logger.warning(f"Temporary error downloading results (attempt {attempt + 1}/{max_retries}): job_id={job_id} - {error_detail}")
                    
                    if attempt < max_retries - 1:
                        backoff = min(2 ** attempt, 30)
                        logger.debug(f"Retrying download in {backoff} seconds...")
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        logger.error(f"Download failed after {max_retries} attempts: job_id={job_id}")
                        return False
                        
                else:
                    # Other 4xx errors
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', response.text[:500])
                    except:
                        error_detail = response.text[:500] if response.text else f"HTTP {response.status_code}"
                    
                    logger.error(f"Download failed: job_id={job_id} - {error_detail}")
                    return False  # Don't retry for other 4xx errors
                    
            except (RequestError, TimeoutException) as e:
                error_type = "timeout" if isinstance(e, TimeoutException) else "network"
                logger.warning(f"{error_type.title()} error downloading results (attempt {attempt + 1}/{max_retries}): job_id={job_id} - {str(e)}")
                
                if attempt < max_retries - 1:
                    backoff = min(2 ** attempt, 30)
                    logger.debug(f"Retrying download after {error_type} error in {backoff} seconds...")
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"Download failed after {max_retries} {error_type} errors: job_id={job_id}")
                    return False
                    
            except Exception as e:
                logger.error(f"Unexpected error downloading results: job_id={job_id} - {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                return False
                
        return False
    
    async def cleanup_job(self, job_id: str):
        """Delete job from server with proper error logging."""
        try:
            response = await self.http_client.delete(
                f"{self.server_url}/job/{job_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.debug(f"Successfully cleaned up job: job_id={job_id}")
            elif response.status_code == 404:
                logger.debug(f"Job already deleted or not found during cleanup: job_id={job_id}")
            elif response.status_code == 401:
                logger.warning(f"Authentication failed during cleanup: job_id={job_id}")
            elif response.status_code == 403:
                logger.warning(f"Access forbidden during cleanup: job_id={job_id}")
            else:
                try:
                    error_data = response.json()
                    error_detail = error_data.get('detail', f"HTTP {response.status_code}")
                except:
                    error_detail = f"HTTP {response.status_code}"
                logger.warning(f"Cleanup failed: job_id={job_id} - {error_detail}")
                
        except (RequestError, TimeoutException) as e:
            error_type = "timeout" if isinstance(e, TimeoutException) else "network"
            logger.debug(f"{error_type.title()} error during cleanup (ignoring): job_id={job_id} - {str(e)}")
        except Exception as e:
            logger.debug(f"Unexpected error during cleanup (ignoring): job_id={job_id} - {str(e)}")
            # Don't log full traceback for cleanup errors as they're non-critical

# Core async processing functions
async def process_single_file(client: TranscriptionClient, audio_file: Path, 
                            output_dir: Path, model: str, progress_mgr: ProgressManager,
                            no_cleanup: bool = False) -> bool:
    """Process a single file with progress tracking."""
    job_id = None
    file_name = audio_file.name
    start_time = time.time()
    current_stage = "initialization"
    
    try:
        logger.info(f"Starting to process: {file_name}")
        
        # Upload with progress (using smart upload strategy)
        current_stage = "upload"
        file_size = audio_file.stat().st_size
        upload_status = "chunked_uploading" if file_size > client.large_file_threshold else "uploading"
        await progress_mgr.update_status("temp", file_name, upload_status)
        job_id = await client.upload_file_smart(audio_file, model)
        if not job_id:
            error_msg = f"Upload failed - no job_id returned"
            logger.error(f"{file_name}: {error_msg}")
            raise Exception(error_msg)
        
        logger.info(f"Upload complete for {file_name}: job_id={job_id}")
        
        # Start processing
        current_stage = "start_processing"
        await progress_mgr.update_status(job_id, file_name, "starting")
        processing_started, job_already_completed = await client.start_processing(job_id)
        if not processing_started:
            error_msg = f"Failed to start processing for job_id={job_id}"
            logger.error(f"{file_name}: {error_msg}")
            raise Exception(error_msg)
        
        # Wait for completion (skip if already completed)
        if not job_already_completed:
            current_stage = "wait_completion"
            await progress_mgr.update_status(job_id, file_name, "processing")
            if not await client.wait_for_completion(job_id):
                error_msg = f"Processing failed on server for job_id={job_id}"
                logger.error(f"{file_name}: {error_msg}")
                raise Exception(error_msg)
        else:
            logger.info(f"Job was already completed: {file_name} (job_id={job_id})")
        
        # Download results
        current_stage = "download"
        await progress_mgr.update_status(job_id, file_name, "downloading")
        if not await client.download_results(job_id, output_dir):
            error_msg = f"Download failed for job_id={job_id}"
            logger.error(f"{file_name}: {error_msg}")
            raise Exception(error_msg)
        
        # Cleanup (optional)
        if not no_cleanup:
            current_stage = "cleanup"
            await client.cleanup_job(job_id)
            logger.debug(f"Cleaned up job: {job_id}")
        
        total_duration = time.time() - start_time
        logger.info(f"Successfully processed {file_name} in {total_duration:.1f}s (job_id={job_id})")
        await progress_mgr.mark_completed(job_id, success=True)
        return True
        
    except Exception as e:
        total_duration = time.time() - start_time
        error_msg = f"Failed at stage '{current_stage}' after {total_duration:.1f}s: {str(e)}"
        
        logger.error(f"Processing failed for {file_name}: {error_msg}")
        logger.debug(f"Full traceback for {file_name}:\n{traceback.format_exc()}")
        
        if job_id:
            await progress_mgr.mark_completed(job_id, success=False, error_msg=error_msg)
        else:
            await progress_mgr.mark_completed(f"failed-{file_name}", success=False, error_msg=error_msg)
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
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for troubleshooting")
    
    args = parser.parse_args()
    
    # Setup logging based on debug flag
    setup_logging(args.debug)
    
    # Log startup information if debug mode
    if args.debug:
        logger.info("=== Transcription Client Starting ===")
        logger.info(f"Debug mode: ENABLED")
        logger.info(f"Input path: {args.input}")
        logger.info(f"Output path: {args.output}")
        logger.info(f"Server URL: {args.server}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Max concurrent: {args.max_concurrent}")
    
    # Validate input
    if not args.input.exists():
        print(f"❌ Input path does not exist: {args.input}")
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Find audio files
    audio_files = find_audio_files(args.input)
    if not audio_files:
        print(f"❌ No audio files found in: {args.input}")
        print("   Supported formats: MP3, WAV, M4A, FLAC, OGG")
        logger.error(f"No audio files found in: {args.input}")
        sys.exit(1)
    
    logger.info(f"Found {len(audio_files)} audio files")
    
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
    
    # Create async HTTP client with robust configuration
    limits = httpx.Limits(
        max_keepalive_connections=10,
        max_connections=20
    )
    
    timeout = httpx.Timeout(
        timeout=60.0,  # Overall timeout
        connect=60.0,  # Connection timeout
        read=1800.0,   # Read timeout (30 minutes for large files)
        write=60.0     # Write timeout
    )
    
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        follow_redirects=True
    ) as http_client:
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
        logger.info(f"Starting concurrent processing of {len(files_to_process)} files")
        results = await process_files_concurrently(
            client, files_to_process, args.output, args.model, 
            args.max_concurrent, args.no_cleanup
        )
        logger.info(f"Concurrent processing complete")
        
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