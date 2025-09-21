#!/usr/bin/env python3
"""
Test script to simulate multiple concurrent clients accessing the FastAPI server.
"""

import asyncio
import aiohttp
import sys
import time
from pathlib import Path
import argparse
from typing import List, Dict, Any

# Server configuration
SERVER_URL = "http://localhost:8080"
API_KEY = "mv_mtvG2X4U_dqRgdWMvSEoFtpMjRJkL4zlkwEXYH2I"

async def upload_and_process(session: aiohttp.ClientSession, client_id: int, audio_file: Path) -> Dict[str, Any]:
    """Simulate a single client uploading and processing a file."""
    
    print(f"[Client {client_id}] Starting upload of {audio_file.name}")
    start_time = time.time()
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    try:
        # 1. Upload file
        with open(audio_file, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=audio_file.name)
            data.add_field('model', 'turbo')
            
            async with session.post(f"{SERVER_URL}/upload-single", headers=headers, data=data) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"[Client {client_id}] Upload failed: {error_text}")
                    return {"client_id": client_id, "status": "upload_failed", "error": error_text}
                
                upload_result = await resp.json()
                job_id = upload_result["job_id"]
                
        print(f"[Client {client_id}] File uploaded, job_id: {job_id}")
        
        # 2. Start processing
        async with session.post(f"{SERVER_URL}/process/{job_id}", headers=headers) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                print(f"[Client {client_id}] Process start failed: {error_text}")
                return {"client_id": client_id, "status": "process_failed", "error": error_text}
            
            process_result = await resp.json()
            print(f"[Client {client_id}] Processing started, queue position: {process_result.get('queue_size', 'unknown')}")
        
        # 3. Poll status until complete
        while True:
            async with session.get(f"{SERVER_URL}/status/{job_id}", headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"[Client {client_id}] Status check failed: {error_text}")
                    return {"client_id": client_id, "status": "status_failed", "error": error_text}
                
                status = await resp.json()
                current_status = status["status"]
                
                if current_status == "completed":
                    elapsed_time = time.time() - start_time
                    print(f"[Client {client_id}] ✓ Completed in {elapsed_time:.2f} seconds")
                    return {
                        "client_id": client_id,
                        "status": "completed",
                        "job_id": job_id,
                        "elapsed_time": elapsed_time,
                        "processed_count": status["processed_count"],
                        "failed_count": status["failed_count"]
                    }
                elif current_status == "failed":
                    print(f"[Client {client_id}] ✗ Failed: {status.get('error', 'Unknown error')}")
                    return {
                        "client_id": client_id,
                        "status": "failed",
                        "job_id": job_id,
                        "error": status.get("error")
                    }
                else:
                    print(f"[Client {client_id}] Status: {current_status}")
                    await asyncio.sleep(2)  # Poll every 2 seconds
                    
    except Exception as e:
        print(f"[Client {client_id}] Exception: {e}")
        return {"client_id": client_id, "status": "exception", "error": str(e)}

async def check_server_health(session: aiohttp.ClientSession):
    """Check server health and system status."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    try:
        # Check health
        async with session.get(f"{SERVER_URL}/health") as resp:
            if resp.status == 200:
                health = await resp.json()
                print("\n=== Server Health ===")
                print(f"Status: {health['status']}")
                print(f"Device: {health['device']}")
                print(f"Model Instances: {health['model_instances']}")
                print(f"Active Workers: {health['active_workers']}")
                print(f"Queue Size: {health['queue_size']}")
        
        # Check system status
        async with session.get(f"{SERVER_URL}/system", headers=headers) as resp:
            if resp.status == 200:
                system = await resp.json()
                print("\n=== System Status ===")
                print(f"Workers: {system['workers']['active']}/{system['workers']['configured']}")
                print(f"Model Instances: {system['model']['instances_loaded']}")
                print(f"Jobs in Queue: {system['queue']['size']}")
                print(f"Jobs Processing: {system['jobs']['processing']}")
                print(f"Jobs Completed: {system['jobs']['completed']}")
                print("=" * 20 + "\n")
                
    except Exception as e:
        print(f"Failed to check server status: {e}")

async def run_concurrent_test(audio_files: List[Path], num_clients: int = 4):
    """Run concurrent client simulation."""
    
    print(f"\n{'='*60}")
    print(f"Starting concurrent test with {num_clients} clients")
    print(f"Audio files: {[f.name for f in audio_files[:num_clients]]}")
    print(f"{'='*60}\n")
    
    async with aiohttp.ClientSession() as session:
        # Check server health first
        await check_server_health(session)
        
        # Create tasks for concurrent clients
        tasks = []
        for i in range(num_clients):
            # Use modulo to cycle through files if we have fewer files than clients
            audio_file = audio_files[i % len(audio_files)]
            task = upload_and_process(session, i, audio_file)
            tasks.append(task)
        
        # Run all clients concurrently
        print(f"Launching {num_clients} concurrent clients...\n")
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Concurrent clients: {num_clients}")
        
        successful = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] in ["failed", "upload_failed", "process_failed", "exception"])
        
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if successful > 0:
            avg_time = sum(r["elapsed_time"] for r in results if r["status"] == "completed") / successful
            print(f"Average processing time per client: {avg_time:.2f} seconds")
            print(f"Speedup vs sequential: {(avg_time * num_clients / total_time):.2f}x")
        
        # Check final server state
        await check_server_health(session)
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Test concurrent client access to FastAPI transcription server")
    parser.add_argument("audio_files", nargs="+", type=Path, help="Audio files to upload")
    parser.add_argument("--clients", "-c", type=int, default=4, help="Number of concurrent clients (default: 4)")
    parser.add_argument("--server", "-s", type=str, default="http://localhost:8080", help="Server URL")
    parser.add_argument("--api-key", "-k", type=str, default="mv_mtvG2X4U_dqRgdWMvSEoFtpMjRJkL4zlkwEXYH2I", help="API key")
    
    args = parser.parse_args()
    
    global SERVER_URL, API_KEY
    SERVER_URL = args.server
    API_KEY = args.api_key
    
    # Validate audio files exist
    audio_files = []
    for file_path in args.audio_files:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        audio_files.append(file_path)
    
    if not audio_files:
        print("Error: No valid audio files provided")
        sys.exit(1)
    
    # Run the test
    asyncio.run(run_concurrent_test(audio_files, args.clients))

if __name__ == "__main__":
    main()