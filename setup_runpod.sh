#!/bin/bash
# RunPod Setup Script - One-time configuration for audio transcription

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration file
RUNPOD_CONFIG_FILE="$HOME/.runpod_config"

print_header() {
    echo -e "${BLUE}"
    echo "🚀 RunPod Audio Transcription Setup"
    echo "===================================="
    echo -e "${NC}"
}

print_status() {
    echo -e "${BLUE}ℹ️${NC} $1"
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

print_step() {
    echo -e "${PURPLE}👉${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ️${NC} $1"
}

check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is required but not installed."
        print_step "Install Docker from: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        print_error "Docker is installed but not running."
        print_step "Please start Docker Desktop and try again."
        exit 1
    fi
    
    print_success "Docker is installed and running"
    
    # Check jq for JSON processing
    if ! command -v jq &> /dev/null; then
        print_error "jq is required for API response processing"
        print_step "Install jq:"
        echo "  macOS: brew install jq"
        echo "  Ubuntu: sudo apt-get install jq"
        echo "  Windows: Download from https://stedolan.github.io/jq/download/"
        exit 1
    else
        print_success "jq is installed"
    fi
    
    # Check runpodctl
    if ! command -v runpodctl &> /dev/null; then
        print_warning "runpodctl is not installed (needed for file transfers)"
        print_step "Install runpodctl:"
        echo "  pip install runpodctl"
        echo "  OR download from: https://github.com/runpod/runpodctl/releases"
        echo ""
        read -p "Install runpodctl now via pip? (Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            if command -v pip3 &> /dev/null; then
                print_status "Installing runpodctl via pip3..."
                if pip3 install runpodctl; then
                    print_success "runpodctl installed successfully"
                else
                    print_error "Failed to install runpodctl via pip3"
                    exit 1
                fi
            elif command -v pip &> /dev/null; then
                print_status "Installing runpodctl via pip..."
                if pip install runpodctl; then
                    print_success "runpodctl installed successfully"
                else
                    print_error "Failed to install runpodctl via pip"
                    exit 1
                fi
            else
                print_error "pip/pip3 not found. Please install runpodctl manually."
                exit 1
            fi
        else
            print_warning "Please install runpodctl manually before proceeding"
            exit 1
        fi
    else
        print_success "runpodctl is installed"
    fi
    
    # Check curl
    if ! command -v curl &> /dev/null; then
        print_error "curl is required but not installed."
        exit 1
    fi
    
    # Check unzip
    if ! command -v unzip &> /dev/null; then
        print_error "unzip is required but not installed."
        exit 1
    fi
}

get_runpod_api_key() {
    echo "" >&2
    print_step "Step 1: Get your RunPod API Key" >&2
    echo "" >&2
    echo "1. Visit: ${CYAN}https://www.runpod.io/console/user/settings${NC}" >&2
    echo "2. Log in to your RunPod account (or create one)" >&2
    echo "3. Go to 'Settings' → 'API Keys'" >&2
    echo "4. Click 'Create API Key'" >&2
    echo "5. Give it a name like 'Audio Transcription'" >&2
    echo "6. Copy the generated API key" >&2
    echo "" >&2
    
    read -p "Press Enter when you have your API key ready..." >&2
    echo "" >&2
    
    while true; do
        read -p "Paste your RunPod API Key: " api_key
        
        # Trim whitespace from the key
        api_key=$(echo "$api_key" | tr -d '[:space:]')
        
        if [[ -z "$api_key" ]]; then
            print_error "API key cannot be empty" >&2
            continue
        fi
        
        if [[ ${#api_key} -lt 20 ]]; then
            print_error "API key seems too short (${#api_key} characters). Please check and try again." >&2
            continue
        fi
        
        # More flexible validation - allow common API key characters
        if [[ ! "$api_key" =~ ^[A-Za-z0-9._-]+$ ]]; then
            print_error "API key contains invalid characters. Only letters, numbers, dots, underscores, and hyphens are allowed." >&2
            print_error "Your key contains: $(echo "$api_key" | sed 's/[A-Za-z0-9._-]//g' | fold -w1 | sort | uniq | tr '\n' ' ')" >&2
            continue
        fi
        
        break
    done
    
    echo "$api_key"
}

get_docker_hub_info() {
    # Print all user interaction to stderr so it's not captured
    echo "" >&2
    print_step "Step 2: Docker Hub Configuration" >&2
    echo "" >&2
    echo "We need to build and push your Docker image to Docker Hub." >&2
    echo "This will store your transcription container for RunPod to use." >&2
    echo "" >&2
    
    while true; do
        read -p "Enter your Docker Hub username: " docker_username
        
        # Clean the username - remove any color codes, spaces, newlines
        docker_username=$(echo "$docker_username" | tr -d '\033\n\r' | sed 's/\[[0-9;]*m//g' | tr -d '[:space:]')
        
        if [[ -z "$docker_username" ]]; then
            print_error "Docker Hub username is required" >&2
            continue
        fi
        
        # Validate username format (Docker Hub usernames are alphanumeric with hyphens/underscores)
        if [[ ! "$docker_username" =~ ^[a-zA-Z0-9_-]+$ ]]; then
            print_error "Invalid Docker Hub username. Only letters, numbers, hyphens, and underscores are allowed." >&2
            print_error "You entered: '$docker_username'" >&2
            continue
        fi
        
        break
    done
    
    echo "" >&2
    print_status "We'll create a Docker image named: ${docker_username}/whisper-transcriber" >&2
    echo "" >&2
    
    # Only echo the clean username to stdout (this gets captured)
    echo "$docker_username"
}

test_runpod_connection() {
    local api_key="$1"
    
    print_status "Testing RunPod API connection..."
    print_info "API key length: ${#api_key} characters"
    print_info "Configuring runpodctl with your API key..."
    
    # Configure runpodctl with the API key
    print_status "Configuring runpodctl..."
    local config_output=$(runpodctl config --apiKey="$api_key" 2>&1)
    
    # Check if configuration was successful by looking for success message
    if [[ "$config_output" == *"Configuration saved"* ]]; then
        print_success "runpodctl configuration saved successfully"
        
        # Check SSH key status
        if [[ "$config_output" == *"SSH key pair generated"* ]]; then
            print_success "SSH keys generated successfully"
        elif [[ "$config_output" == *"Existing local SSH key found"* ]]; then
            print_success "Using existing SSH keys"
        fi
    else
        print_error "Failed to save runpodctl configuration"
        print_error "Output: $config_output"
        return 1
    fi
    
    # Test the API key by trying to list pods (simple operation)
    print_status "Testing API key functionality..."
    local test_output=$(runpodctl get pod 2>&1)
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "API key validated successfully!"
        print_info "runpodctl is working correctly"
        
        # Check if we have any pods
        if [[ -n "$test_output" && "$test_output" != *"No pods found"* ]]; then
            print_info "Found existing pods in your account"
        else
            print_info "No existing pods found (this is normal for new accounts)"
        fi
        
        return 0
    else
        # Check for common error messages
        if [[ "$test_output" == *"unauthorized"* ]] || [[ "$test_output" == *"Unauthorized"* ]] || [[ "$test_output" == *"401"* ]]; then
            print_error "Invalid API key (unauthorized)"
            print_error "Please verify your API key from: https://www.runpod.io/console/user/settings"
        elif [[ "$test_output" == *"forbidden"* ]] || [[ "$test_output" == *"Forbidden"* ]] || [[ "$test_output" == *"403"* ]]; then
            print_error "API key lacks required permissions"
        elif [[ "$test_output" == *"network"* ]] || [[ "$test_output" == *"connection"* ]]; then
            print_error "Network connection failed"
        elif [[ "$test_output" == *"timeout"* ]]; then
            print_error "Connection timed out"
        else
            print_error "API validation failed"
            print_error "Error: $test_output"
        fi
        return 1
    fi
}

build_docker_image() {
    local docker_username="$1"
    local image_name="${docker_username}/whisper-transcriber:latest"
    
    echo "" >&2
    print_step "Step 3: Building Docker Image" >&2
    echo "" >&2
    print_status "Building Docker image: $image_name" >&2
    print_warning "This may take 10-15 minutes (downloading models and dependencies)..." >&2
    echo "" >&2
    
    # Clean the username again to be absolutely sure
    docker_username=$(echo "$docker_username" | tr -d '\033\n\r' | sed 's/\[[0-9;]*m//g' | tr -d '[:space:]')
    
    if ! docker build --platform linux/amd64 -t "$image_name" . >&2; then
        print_error "Docker build failed" >&2
        exit 1
    fi
    
    print_success "Docker image built successfully" >&2
    
    echo "" >&2
    print_status "Pushing image to Docker Hub..." >&2
    if ! docker push "$image_name" >&2; then
        print_error "Docker push failed - you may need to login first:" >&2
        print_step "Run: docker login" >&2
        print_step "Then re-run this script" >&2
        exit 1
    fi
    
    print_success "Image pushed to Docker Hub successfully" >&2
    
    # Only output the clean image name to stdout for capture
    echo "$image_name"
}

save_configuration() {
    local api_key="$1"
    local docker_image="$2"
    
    print_status "Saving configuration..."
    
    cat > "$RUNPOD_CONFIG_FILE" << EOF
# RunPod Audio Transcription Configuration
# Generated on $(date)

RUNPOD_API_KEY="$api_key"
DOCKER_IMAGE="$docker_image"

# GPU preferences (can be modified)
PREFERRED_GPU_TYPE="A10"
FALLBACK_GPU_TYPE="A4000"

# Default settings
DEFAULT_MODEL="turbo"
DEFAULT_VOLUME_SIZE="20"
DEFAULT_CONTAINER_DISK="20"
EOF
    
    chmod 600 "$RUNPOD_CONFIG_FILE"  # Secure the config file
    
    print_success "Configuration saved to: $RUNPOD_CONFIG_FILE"
}

show_next_steps() {
    echo ""
    print_success "🎉 Setup Complete!"
    echo ""
    print_step "Next steps:"
    echo ""
    echo "1. Test your setup:"
    echo "   ${CYAN}./runpod_transcribe.sh --help${NC}"
    echo ""
    echo "2. Transcribe your audio files:"
    echo "   ${CYAN}./runpod_transcribe.sh /path/to/mp3s /path/to/output${NC}"
    echo ""
    echo "3. Example with your 250 episodes:"
    echo "   ${CYAN}./runpod_transcribe.sh ~/podcasts ~/transcripts${NC}"
    echo ""
    print_warning "Estimated processing time: 1-2 hours (vs 62+ hours locally)"
    print_warning "Estimated cost: \$1-3 for 250 episodes"
    echo ""
    print_step "Pro tip: Use '--model turbo' for 8x faster processing with near large-v3 quality"
    echo ""
}

main() {
    print_header
    
    echo "This script will help you set up RunPod for audio transcription."
    echo "You'll need:"
    echo "  • RunPod account and API key"
    echo "  • Docker Hub account"
    echo "  • Internet connection for downloading models"
    echo ""
    
    read -p "Continue with setup? (Y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_status "Setup cancelled"
        exit 0
    fi
    
    # Check prerequisites
    check_prerequisites
    
    # Get RunPod API key
    local api_key=$(get_runpod_api_key)
    
    # Test API connection
    if ! test_runpod_connection "$api_key"; then
        print_error "API key validation failed. Please check your key and try again."
        exit 1
    fi
    
    # Get Docker Hub info
    local docker_username=$(get_docker_hub_info)
    
    # Build and push Docker image
    local docker_image=$(build_docker_image "$docker_username")
    
    # Save configuration
    save_configuration "$api_key" "$docker_image"
    
    # Show next steps
    show_next_steps
}

# Check if script is being run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi