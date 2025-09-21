# Multi-Client Concurrent Processing Setup

## Overview
The FastAPI transcription server has been upgraded to support multiple concurrent clients. With an RTX 4090 (24GB VRAM), the server can now process 4 audio files simultaneously using dedicated GPU model instances.

## Key Changes

### 1. Multiple Worker Architecture
- **4 concurrent workers** process jobs from a shared queue
- Each worker has a **dedicated Whisper model instance** on GPU
- True parallel processing without model sharing bottlenecks

### 2. Thread-Safe Operations
- All `jobs_db` operations protected with `asyncio.Lock()`
- Prevents race conditions during concurrent access
- Safe status updates from multiple workers

### 3. Configuration
Environment variables to control concurrency:
```bash
WORKER_COUNT=4        # Number of concurrent workers (default: 4)
MODEL_INSTANCES=4     # Number of GPU model instances (default: 4)
MODEL_NAME=turbo      # Whisper model to use (default: turbo)
```

## Performance Expectations

### Before (Single Worker)
- 1 file processes at a time
- Other clients wait in queue
- ~22% GPU utilization

### After (4 Workers)
- 4 files process simultaneously
- Linear scaling up to 4 clients
- ~80-90% GPU utilization expected
- 4x throughput improvement

## Usage

### Starting the Server
```bash
# Default configuration (4 workers, 4 models)
python runpod_fastapi_server.py

# Custom configuration
WORKER_COUNT=3 MODEL_INSTANCES=3 python runpod_fastapi_server.py
```

### Testing Concurrent Clients
```bash
# Test with 4 concurrent clients
python test_concurrent_clients.py audio1.mp3 audio2.mp3 audio3.mp3 audio4.mp3

# Test with 2 clients
python test_concurrent_clients.py audio1.mp3 audio2.mp3 --clients 2

# Test with custom server
python test_concurrent_clients.py *.mp3 --server http://your-server:8080 --api-key your-key
```

## New Endpoints

### `/system` - System Status
Get detailed information about workers, jobs, and processing status:
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8080/system
```

Response includes:
- Active workers and their status
- Queue size and processing jobs
- Model instances loaded
- Job statistics by status

### `/health` - Enhanced Health Check
Now includes:
- Number of model instances
- Active worker count
- Current queue size

## Architecture

```
Client 1 → Upload → Queue → Worker 0 (Model 0) → Processing
Client 2 → Upload → Queue → Worker 1 (Model 1) → Processing
Client 3 → Upload → Queue → Worker 2 (Model 2) → Processing
Client 4 → Upload → Queue → Worker 3 (Model 3) → Processing
```

Each worker:
1. Pulls job from shared queue
2. Uses dedicated GPU model instance
3. Processes independently
4. Updates job status atomically

## Resource Usage

### GPU (RTX 4090 - 24GB VRAM)
- Single model: ~5.3GB VRAM (22% usage)
- 4 models: ~21GB VRAM (87% usage)
- Safe margin with 24GB total

### CPU
- Typical: 20-30% usage
- Peak: 50-70% with 4 workers
- No bottleneck for audio preprocessing

### RAM
- Minimal usage: 7-10%
- No constraints

## Troubleshooting

### If workers fail to start
- Check GPU memory: `nvidia-smi`
- Reduce `MODEL_INSTANCES` if OOM
- Verify CUDA is available

### If processing seems slow
- Check `/system` endpoint for queue backlog
- Verify all workers are active
- Monitor GPU utilization with `nvidia-smi`

### For debugging
- Worker logs show which worker processes each job
- Job status includes `worker_id` internally
- `/system` endpoint shows real-time processing state