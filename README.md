# RunPod FastAPI Transcription Service

A simplified HTTP API-based solution for audio transcription using RunPod GPU pods. This replaces the complex SSH-based file transfer approach with simple HTTP requests.

## 🚀 Quick Start

### 1. Build and Deploy Container

```bash
# Use the automated deployment script
./deploy.sh your-docker-username

# Or manually:
docker build -t your-username/whisper-transcription .
docker push your-username/whisper-transcription

# Deploy to RunPod using the Docker image
# Set environment variable: RUNPOD_API_KEY=your-secret-api-key
```

### 2. Use the Client

```bash
# Install client dependencies
pip install requests

# Transcribe audio files
python transcribe_client.py /path/to/audio /path/to/output \
  --server https://your-pod-id.runpod.io:8080 \
  --api-key your-secret-api-key
```

## 📋 What's Included

### Files Included:
- `runpod_fastapi_server.py` - FastAPI server with HTTP endpoints
- `Dockerfile` - Optimized Dockerfile without SSH
- `transcribe_client.py` - Simple Python client for uploads/downloads

### Features:
- ✅ **Simple HTTP API** - No SSH keys or complex setup
- ✅ **Individual file uploads** - Upload files one-by-one with progress bars
- ✅ **Extended timeouts** - 30-minute timeouts + 3-attempt retry logic
- ✅ **Real-time progress** - tqdm progress bars and upload speed display
- ✅ **Multi-job processing** - Parallel processing with live status updates
- ✅ **API key authentication** - Simple security
- ✅ **Auto cleanup** - Server manages temporary files
- ✅ **Multiple formats** - Supports MP3, WAV, M4A, FLAC, OGG
- ✅ **Robust error handling** - Automatic retry and graceful failure recovery

## 🔧 API Endpoints

### Upload Files
```bash
curl -X POST "https://your-pod.runpod.io:8080/upload" \
  -H "Authorization: Bearer your-api-key" \
  -F "files=@audio1.mp3" \
  -F "files=@audio2.mp3" \
  -F "model=turbo"
```

### Check Status
```bash
curl "https://your-pod.runpod.io:8080/status/job-id" \
  -H "Authorization: Bearer your-api-key"
```

### Download Results
```bash
curl "https://your-pod.runpod.io:8080/download/job-id" \
  -H "Authorization: Bearer your-api-key" \
  -o transcripts.zip
```

### Health Check
```bash
curl "https://your-pod.runpod.io:8080/health" \
  -H "Authorization: Bearer your-api-key"
```

## 🎯 Usage Examples

### Basic Usage
```bash
# Transcribe a directory of MP3s
python transcribe_client.py ~/podcasts ~/transcripts

# Transcribe a single file
python transcribe_client.py ~/audio/meeting.mp3 ~/transcripts
```

### Advanced Usage
```bash
# Use different model
python transcribe_client.py ~/audio ~/output --model large-v3

# Custom server and API key
python transcribe_client.py ~/audio ~/output \
  --server https://abc123.runpod.io:8080 \
  --api-key your-secret-key

# Keep job on server (don't auto-cleanup)
python transcribe_client.py ~/audio ~/output --no-cleanup
```

### Environment Variables
```bash
# Set default server and API key
export RUNPOD_SERVER_URL=https://your-pod.runpod.io:8080
export RUNPOD_API_KEY=your-secret-api-key

# Now you can use simple commands
python transcribe_client.py ~/audio ~/output
```

## 🏗️ Deployment Steps

### 1. RunPod Setup
1. Create RunPod account and get API key
2. Build and push Docker image to registry
3. Create pod with your Docker image
4. Set environment variable `RUNPOD_API_KEY=your-secret-key`
5. Expose port 8080

### 2. Local Client Setup
```bash
# Install Python dependencies
pip install requests

# Make client executable
chmod +x transcribe_client.py

# Test connection
python transcribe_client.py --help
```

## 🔒 Security

- **API Key Authentication**: All endpoints require Bearer token
- **Input Validation**: File type and model validation
- **Temporary Storage**: Files auto-deleted after processing
- **Error Handling**: Graceful error responses

## 🎚️ Model Options

- `tiny` - Fastest, lowest quality
- `base` - Fast, basic quality  
- `small` - Good balance
- `medium` - Better quality
- `large` - Best quality, slower
- `large-v2` - Enhanced large model
- `large-v3` - Latest large model
- `turbo` - **Default** - 8x faster than large with similar quality

## 📊 Response Formats

### Upload Response
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "files_count": 5,
  "message": "Files uploaded successfully. Processing started.",
  "status_url": "/status/uuid-string"
}
```

### Status Response
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "created_at": "2024-01-01T12:00:00",
  "updated_at": "2024-01-01T12:05:00",
  "files_count": 5,
  "processed_count": 3,
  "failed_count": 0,
  "error": null,
  "download_url": "/download/uuid-string"
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **Connection Failed**
   ```bash
   # Check if pod is running
   curl https://your-pod.runpod.io:8080/health
   ```

2. **Authentication Failed**
   ```bash
   # Verify API key matches pod environment variable
   echo $RUNPOD_API_KEY
   ```

3. **Upload Timeout**
   ```bash
   # Check file sizes and network connection
   # Large files may take several minutes to upload
   ```

4. **No Audio Files Found**
   ```bash
   # Verify file extensions are supported
   ls -la /path/to/audio/*.{mp3,wav,m4a,flac,ogg}
   ```

### Performance Tips

- Use `turbo` model for best speed/quality balance
- Upload files in batches rather than one-by-one
- Use RunPod's high-bandwidth regions
- Monitor GPU utilization in RunPod console

## 🆚 Comparison: Old vs New

| Feature | SSH Method | FastAPI Method |
|---------|------------|----------------|
| **Setup** | Complex SSH keys | Simple HTTP |
| **File Transfer** | SCP + zip files | Direct HTTP upload |
| **Authentication** | SSH keys | API key |
| **Progress** | Manual checking | Real-time API |
| **Error Handling** | Manual retry | Automatic retry |
| **Client Code** | 800+ lines bash | 200 lines Python |
| **Dependencies** | ssh, scp, zip | requests |

## 🎉 Benefits

1. **Dramatically Simpler**: No SSH setup, just HTTP requests
2. **More Reliable**: Better error handling and retries
3. **Real-time Progress**: Monitor job status via API
4. **Cross-platform**: Works on Windows, Mac, Linux
5. **Easier Debugging**: Clear HTTP error messages
6. **Scalable**: Easy to integrate into other applications

This FastAPI solution eliminates all the complexity of SSH file transfers while providing a much more robust and user-friendly experience!

## 📚 Documentation

- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Complete deployment instructions
- **[deploy.sh](deploy.sh)** - Automated deployment script

## 🚀 Quick Reference

### Deploy
```bash
./deploy.sh your-docker-username
```

### Use
```bash
export RUNPOD_SERVER_URL=https://your-pod-url
export RUNPOD_API_KEY=your-secret-key
python transcribe_client.py ~/audio ~/transcripts
```

### Monitor
```bash
# Check pod status
curl https://your-pod-url/health

# View processing costs in RunPod console
# Typical cost: $0.50/hour for RTX A5000
```