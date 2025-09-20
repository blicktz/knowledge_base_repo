# 🔧 Makefile Reference for RunPod Transcription

Quick reference for all the new Makefile targets for Docker and RunPod deployment.

## 🚀 Quick Deployment Workflow

```bash
# 1. Build and push Docker image (uses blickt123 by default)
make docker-deploy

# 3. Set RunPod API key (get from https://runpod.io/console/user/settings)
export RUNPOD_API_KEY=your-runpod-api-key

# 4. Create RunPod instance
make runpod-create

# 5. Get connection info
make runpod-info

# 6. Transcribe audio
make runpod-transcribe INPUT=audio.mp3
```

## 🐳 Docker Commands

| Command | Description | Example |
|---------|-------------|---------|
| `make docker-build` | Build Docker image | `make docker-build` (uses blickt123) |
| `make docker-push` | Build and push to Docker Hub | `make docker-push` (uses blickt123) |
| `make docker-deploy` | Complete build + push workflow | `make docker-deploy` (uses blickt123) |

## ☁️ RunPod Management Commands

### Deployment & Setup
| Command | Description | Notes |
|---------|-------------|-------|
| `make runpod-create` | Create new RunPod instance | Requires Docker image and RUNPOD_API_KEY |
| `make runpod-info` | Get status and connection URL | Shows pod details and health |
| `make runpod-test` | Test health endpoint | Quick connectivity check |

### Usage
| Command | Description | Example |
|---------|-------------|---------|
| `make runpod-transcribe` | Transcribe single file | `make runpod-transcribe INPUT=audio.mp3` |
| `make runpod-batch` | Transcribe directory | `make runpod-batch INPUT_DIR=./mp3s` |

### Control
| Command | Description | Use Case |
|---------|-------------|----------|
| `make runpod-status` | Quick status check | Check if pod is running |
| `make runpod-start` | Start stopped pod | Resume after stopping |
| `make runpod-stop` | Stop running pod | Save money when not in use |
| `make runpod-delete` | Delete pod permanently | Clean up when done |

### Monitoring
| Command | Description | Purpose |
|---------|-------------|---------|
| `make runpod-logs` | View container logs | Debug issues |
| `make runpod-url` | Get just the connection URL | For scripts/automation |

## 🔧 Configuration Variables

### Docker Variables
```bash
DOCKER_USERNAME=blickt123           # Docker Hub username (default: blickt123)
DOCKER_TAG=latest                  # Image tag (default: latest)
```

### RunPod Variables
```bash
RUNPOD_API_KEY=your-api-key        # Required: Your RunPod API key
RUNPOD_GPU_TYPE="NVIDIA RTX A5000" # GPU type (default: A5000)
```

## 💡 Usage Examples

### Complete First-Time Setup
```bash
# Set RunPod API key
export RUNPOD_API_KEY=rp_abc123...

# Deploy everything (uses blickt123 by default)
make docker-deploy
make runpod-create
make runpod-info

# Transcribe files
make runpod-transcribe INPUT="test_mp3/podcast.mp3"
make runpod-batch INPUT_DIR="~/podcasts" OUTPUT_DIR="~/transcripts"
```

### Daily Usage (Pod Already Created)
```bash
# Check if pod is running
make runpod-status

# Start if stopped
make runpod-start

# Transcribe files
make runpod-transcribe INPUT=audio.mp3

# Stop to save money
make runpod-stop
```

### Cost Management
```bash
# Quick status check
make runpod-status

# Stop pod when done (saves ~$0.50/hour)
make runpod-stop

# Restart when needed
make runpod-start

# Delete permanently when project is done
make runpod-delete
```

## 🎯 Common Workflows

### Development/Testing
```bash
# Test locally first
python transcribe_client.py test.mp3 ./output --server http://localhost:8080

# Deploy to RunPod
make docker-deploy
make runpod-create
make runpod-test
```

### Production Use
```bash
# Create production pod
make runpod-create

# Process large batch
make runpod-batch INPUT_DIR="/path/to/hundreds/of/mp3s" OUTPUT_DIR="./all_transcripts"

# Monitor progress
make runpod-logs

# Stop when done
make runpod-stop
```

### Troubleshooting
```bash
# Check pod status
make runpod-info

# View logs for errors
make runpod-logs

# Test connectivity
make runpod-test

# Get just the URL for manual testing
make runpod-url
```

## 📊 Cost Estimates

| GPU Type | Cost/Hour | Typical Use Case |
|----------|-----------|------------------|
| RTX A4000 | ~$0.30 | Light usage, small files |
| RTX A5000 | ~$0.50 | **Recommended** - Good balance |
| RTX A6000 | ~$0.80 | Heavy usage, large batches |

### Example Costs:
- **30 minutes of audio**: ~$2.50 on A5000
- **2 hours of audio**: ~$5.00 on A5000  
- **10 hours of audio**: ~$15.00 on A5000

## 🔍 File Tracking

The Makefile automatically creates and manages:
- `.runpod_pod_id` - Stores your current pod ID (auto-created)
- This file is ignored by git for security

## 🆘 Help Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make runpod-help` | Show only RunPod-related commands |

## ⚠️ Important Notes

1. **API Key Security**: Never commit your RUNPOD_API_KEY to version control
2. **Pod Management**: Pods auto-stop after ~5-10 minutes of inactivity
3. **File Persistence**: Upload/output files are temporary - download results promptly
4. **Cost Control**: Always stop pods when done to avoid unexpected charges
5. **Docker Hub**: Make sure your image is public or properly authenticated

## 🔗 Related Documentation

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Complete deployment instructions
- [README.md](README.md) - Project overview and features
- [deploy.sh](deploy.sh) - Legacy deployment script (still works)