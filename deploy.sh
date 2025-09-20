#!/bin/bash
# Quick deployment script for RunPod FastAPI Transcription Service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}🚀${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

# Get Docker Hub username
if [ -z "$1" ]; then
    echo "Usage: $0 <docker-hub-username> [image-tag]"
    echo ""
    echo "Examples:"
    echo "  $0 myusername"
    echo "  $0 myusername v1.0"
    echo ""
    exit 1
fi

DOCKER_USERNAME="$1"
IMAGE_TAG="${2:-latest}"
IMAGE_NAME="$DOCKER_USERNAME/whisper-transcription:$IMAGE_TAG"

echo "🎵 RunPod FastAPI Transcription Deployment"
echo "==========================================="
echo ""
echo "📦 Image: $IMAGE_NAME"
echo "🐳 Docker Hub: https://hub.docker.com/r/$DOCKER_USERNAME/whisper-transcription"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_success "Docker is running"

# Build the image
print_status "Building Docker image..."
if docker build -t "$IMAGE_NAME" .; then
    print_success "Image built successfully"
else
    print_error "Failed to build image"
    exit 1
fi

# Check if user is logged into Docker Hub
if ! docker info | grep -q "Username"; then
    print_warning "Not logged into Docker Hub"
    print_status "Logging into Docker Hub..."
    docker login
fi

# Push the image
print_status "Pushing image to Docker Hub..."
if docker push "$IMAGE_NAME"; then
    print_success "Image pushed successfully"
else
    print_error "Failed to push image"
    exit 1
fi

# Show next steps
echo ""
print_success "🎉 Deployment preparation complete!"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. 🌐 Deploy to RunPod:"
echo "   - Go to: https://www.runpod.io/console"
echo "   - Create new pod with image: $IMAGE_NAME"
echo "   - Expose port 8080 as HTTP"
echo "   - Set environment variable: RUNPOD_API_KEY=your-secret-key"
echo ""
echo "2. 🔧 Get your server URL (example):"
echo "   https://abc123def456-8080.proxy.runpod.net"
echo ""
echo "3. 🎵 Test your deployment:"
echo "   curl https://your-pod-url/health"
echo ""
echo "4. 🎯 Use your transcription service:"
echo "   export RUNPOD_SERVER_URL=https://your-pod-url"
echo "   export RUNPOD_API_KEY=your-secret-key"
echo "   python transcribe_client.py ~/audio ~/transcripts"
echo ""
echo "📖 For detailed instructions, see: DEPLOYMENT_GUIDE.md"
echo ""

# Generate a sample API key
SAMPLE_API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))" 2>/dev/null || echo "your-secret-api-key-here")
echo "🔑 Sample API Key (use this in RunPod environment):"
echo "   RUNPOD_API_KEY=$SAMPLE_API_KEY"
echo ""

print_success "Ready for RunPod deployment! 🚀"