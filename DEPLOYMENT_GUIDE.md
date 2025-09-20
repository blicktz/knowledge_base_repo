# 🚀 RunPod FastAPI Transcription Deployment Guide

This guide will walk you through building and deploying your FastAPI transcription service to RunPod.

## 📋 Prerequisites

- Docker installed on your machine
- Docker Hub account (or other container registry)
- RunPod account with API access
- Your transcription files ready

## 🏗️ Step 1: Build the Docker Image

### 1.1 Build locally
```bash
# Navigate to your project directory
cd /Users/blickt/Documents/src/pdf_2_text

# Build the Docker image
docker build -t your-username/whisper-transcription:latest .

# Test the image locally (optional)
docker run -p 8080:8080 -e RUNPOD_API_KEY=test-key your-username/whisper-transcription:latest
```

### 1.2 Push to Docker Hub
```bash
# Login to Docker Hub
docker login

# Push the image
docker push your-username/whisper-transcription:latest
```

## 🌐 Step 2: Deploy to RunPod

### 2.1 Using RunPod Web Interface

1. **Login to RunPod Console**: https://www.runpod.io/console
2. **Go to Pods > Deploy**
3. **Configure your pod**:
   - **Template**: Custom
   - **Image**: `your-username/whisper-transcription:latest`
   - **GPU**: Choose based on your needs:
     - RTX A4000: $0.30/hr (good for small batches)
     - RTX A5000: $0.50/hr (recommended)
     - RTX A6000: $0.80/hr (fast processing)
   - **Disk**: 20GB minimum
   - **Ports**: Expose port 8080 as HTTP

4. **Environment Variables**:
   ```
   RUNPOD_API_KEY=your-secret-api-key-here
   ```

5. **Click Deploy**

### 2.2 Using RunPodCTL (Command Line)

```bash
# Install runpodctl
pip install runpodctl

# Configure API key
runpodctl config --api-key YOUR_RUNPOD_API_KEY

# Deploy pod
runpodctl create pod \
  --name "whisper-transcription" \
  --imageName "your-username/whisper-transcription:latest" \
  --gpuType "NVIDIA RTX A5000" \
  --gpuCount 1 \
  --volumeSize 20 \
  --containerDiskSize 10 \
  --ports "8080/http" \
  --env "RUNPOD_API_KEY=your-secret-api-key"
```

## 🔧 Step 3: Get Your Server URL

After deployment, you'll get a URL like:
```
https://abc123def456-8080.proxy.runpod.net
```

Test your deployment:
```bash
curl https://your-pod-url/health
```

## 🎵 Step 4: Use Your Transcription Service

### 4.1 Set Environment Variables (Recommended)
```bash
export RUNPOD_SERVER_URL=https://your-pod-url
export RUNPOD_API_KEY=your-secret-api-key
```

### 4.2 Transcribe Audio Files
```bash
# Single file
python transcribe_client.py audio.mp3 ./transcripts

# Multiple files in directory
python transcribe_client.py ~/podcast-files ~/transcripts

# With custom settings
python transcribe_client.py ~/audio ~/output \
  --model large-v3 \
  --server https://your-pod-url \
  --api-key your-api-key
```

## 💰 Step 5: Cost Management

### 5.1 Automatic Shutdown
Your pod will auto-shutdown when idle to save costs. The default idle timeout is usually 5-10 minutes.

### 5.2 Manual Management
```bash
# List your pods
runpodctl get pods

# Stop a pod
runpodctl stop pod POD_ID

# Start a pod
runpodctl start pod POD_ID

# Delete a pod
runpodctl remove pod POD_ID
```

### 5.3 Cost Estimates
- **RTX A4000**: ~$0.30/hour (~$1.50 for 30min of audio)
- **RTX A5000**: ~$0.50/hour (~$2.50 for 30min of audio)
- **Turbo model**: ~8x faster than large model

## 🛠️ Troubleshooting

### Common Issues

#### 1. Connection Failed
```bash
# Check if pod is running
curl https://your-pod-url/health

# If 404/503, pod might be starting up (wait 2-3 minutes)
# If timeout, check pod logs in RunPod console
```

#### 2. Authentication Failed
```bash
# Verify your API key matches the pod environment
echo $RUNPOD_API_KEY

# Check pod environment in RunPod console
```

#### 3. Upload Timeout
```bash
# The client now has 30-minute timeouts and retry logic
# If still failing, check:
# - File size (very large files >1GB may need special handling)
# - Network connection stability
# - Pod GPU memory (restart pod if out of memory)
```

#### 4. Out of GPU Memory
```bash
# Restart the pod to clear GPU memory
runpodctl restart pod POD_ID

# Or use a smaller model
python transcribe_client.py ... --model small
```

## 🔒 Security Best Practices

### 1. API Key Management
```bash
# Use environment variables (never hardcode keys)
export RUNPOD_API_KEY=your-secret-key

# Generate a strong random API key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. Network Security
- RunPod provides HTTPS by default
- Your API requires authentication for all endpoints
- Files are automatically cleaned up after processing

## 📊 Monitoring & Logs

### 1. Check Pod Status
```bash
# Via CLI
runpodctl get pod POD_ID

# Via API
curl -H "Authorization: Bearer $RUNPOD_API_KEY" \
  "https://api.runpod.io/v2/pods/POD_ID"
```

### 2. View Logs
- **RunPod Console**: Go to your pod → Logs tab
- **Via CLI**: `runpodctl logs POD_ID`

### 3. Monitor Usage
- **GPU Utilization**: Available in RunPod console
- **Processing Stats**: Client shows detailed progress
- **Cost Tracking**: RunPod dashboard shows hourly costs

## 🚀 Optimization Tips

### 1. Model Selection
- **turbo**: Fastest, good quality (recommended)
- **large-v3**: Best quality, slower
- **small**: Very fast, lower quality

### 2. Batch Processing
- Process multiple files in one session to maximize GPU utilization
- Larger files are more cost-efficient (fixed startup time)

### 3. Regional Selection
- Choose RunPod regions close to your location
- Some regions have better pricing/availability

## 🔄 Updates & Maintenance

### 1. Update Your Service
```bash
# Build new image
docker build -t your-username/whisper-transcription:v2 .
docker push your-username/whisper-transcription:v2

# Update pod with new image
# (Currently requires creating new pod in RunPod)
```

### 2. Backup Important Data
- Download transcripts immediately after processing
- Keep local copies of important audio files
- Export job logs if needed for billing/tracking

## 📞 Support

### 1. Client Issues
- Check this deployment guide
- Verify environment variables
- Test with health endpoint first

### 2. Server Issues
- Check RunPod console logs
- Restart pod if GPU memory issues
- Try different GPU types if performance issues

### 3. RunPod Platform Issues
- RunPod Discord: https://discord.gg/runpod
- RunPod Documentation: https://docs.runpod.io/
- Support tickets via RunPod console

---

## 🎉 Quick Start Summary

```bash
# 1. Build and push image
docker build -t username/whisper-transcription .
docker push username/whisper-transcription

# 2. Deploy to RunPod (web console or CLI)
# 3. Set environment variables
export RUNPOD_SERVER_URL=https://your-pod-url
export RUNPOD_API_KEY=your-secret-key

# 4. Transcribe files
python transcribe_client.py ~/audio ~/transcripts

# 5. Manage costs
runpodctl stop pod POD_ID  # when done
```

That's it! You now have a production-ready, scalable audio transcription service running on GPU infrastructure. 🚀