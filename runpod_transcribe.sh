#!/bin/bash
# RunPod One-Click Audio Transcription Script
# Usage: ./runpod_transcribe.sh /path/to/mp3s /path/to/output

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
DOCKER_IMAGE="your-username/whisper-transcriber"  # Update this with your Docker Hub username
RUNPOD_CONFIG_FILE="$HOME/.runpod_config"
TEMP_DIR="/tmp/runpod_transcribe_$$"
PROGRESS_INTERVAL=30  # Check progress every 30 seconds

# Function to print colored output
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

print_info() {
    echo -e "${CYAN}ℹ️${NC} $1"
}

# Function to show usage
show_usage() {
    echo "RunPod One-Click Audio Transcription"
    echo "===================================="
    echo ""
    echo "Usage: $0 <input_folder> <output_folder> [options]"
    echo ""
    echo "Arguments:"
    echo "  input_folder   Directory containing MP3 files to transcribe"
    echo "  output_folder  Directory where transcripts will be saved"
    echo ""
    echo "Options:"
    echo "  --model MODEL  Whisper model to use (default: turbo)"
    echo "                 Options: tiny, base, small, medium, large, large-v2, large-v3, turbo"
    echo "  --gpu-type     GPU type preference (default: auto)"
    echo "                 Options: A4000, A5000, A6000, A100, H100"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 ~/podcasts ~/transcripts"
    echo "  $0 /path/to/mp3s /path/to/output --model large-v3"
    echo "  $0 ~/audio ~/text --gpu-type A100"
    echo ""
    echo "First-time setup:"
    echo "  ./setup_runpod.sh  # Configure API keys"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if jq is installed
    if ! command -v jq &> /dev/null; then
        print_error "jq is required but not installed."
        print_info "Install with: brew install jq (macOS) or apt-get install jq (Ubuntu)"
        exit 1
    fi
    
    # Check if curl is installed
    if ! command -v curl &> /dev/null; then
        print_error "curl is required but not installed."
        exit 1
    fi
    
    # Check if zip is installed
    if ! command -v zip &> /dev/null; then
        print_error "zip is required but not installed."
        exit 1
    fi
    
    # Check if unzip is installed
    if ! command -v unzip &> /dev/null; then
        print_error "unzip is required but not installed."
        exit 1
    fi
    
    # Check if runpodctl is installed
    if ! command -v runpodctl &> /dev/null; then
        print_error "runpodctl is required but not installed."
        print_info "Install with: pip install runpodctl"
        print_info "Or download from: https://github.com/runpod/runpodctl/releases"
        print_info "Then run setup: ./setup_runpod.sh"
        exit 1
    fi
    
    # Check if bc is available for cost calculations
    if ! command -v bc &> /dev/null; then
        print_warning "bc not found - cost estimates may not work"
        print_info "Install with: brew install bc (macOS) or apt-get install bc (Ubuntu)"
    fi
    
    # Check RunPod configuration
    if [[ ! -f "$RUNPOD_CONFIG_FILE" ]]; then
        print_error "RunPod configuration not found."
        print_info "Please run: ./setup_runpod.sh"
        exit 1
    fi
    
    source "$RUNPOD_CONFIG_FILE"
    
    if [[ -z "$RUNPOD_API_KEY" ]]; then
        print_error "RUNPOD_API_KEY not configured."
        print_info "Please run: ./setup_runpod.sh"
        exit 1
    fi
    
    if [[ -z "$DOCKER_IMAGE" ]]; then
        print_error "DOCKER_IMAGE not configured."
        print_info "Please run: ./setup_runpod.sh"
        exit 1
    fi
    
    print_success "Prerequisites OK"
}

# Function to validate input directory
validate_input() {
    local input_dir="$1"
    local output_dir="$2"
    
    if [[ ! -d "$input_dir" ]]; then
        print_error "Input directory does not exist: $input_dir"
        exit 1
    fi
    
    # Count MP3 files
    local mp3_count=$(find "$input_dir" -name "*.mp3" -type f | wc -l | tr -d ' ')
    
    if [[ $mp3_count -eq 0 ]]; then
        print_error "No MP3 files found in: $input_dir"
        exit 1
    fi
    
    print_info "Found $mp3_count MP3 files in input directory"
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    print_success "Input validation complete"
}

# Function to create zip archive of MP3 files
create_audio_archive() {
    local input_dir="$1"
    local archive_path="$TEMP_DIR/audio_files.zip"
    
    print_status "Creating audio archive..."
    
    mkdir -p "$TEMP_DIR"
    
    cd "$input_dir"
    zip -r "$archive_path" . -i "*.mp3" -x "*/.*" >/dev/null 2>&1
    
    local archive_size=$(du -h "$archive_path" | cut -f1)
    print_success "Archive created: $archive_size"
    
    echo "$archive_path"
}

# Function to launch RunPod container
launch_runpod_container() {
    local gpu_type="$1"
    
    print_status "Launching RunPod container..."
    
    # GPU type selection
    local gpu_query=""
    if [[ "$gpu_type" != "auto" ]]; then
        gpu_query="&gpuTypeId=$gpu_type"
    fi
    
    # Use runpodctl to create the pod
    local timestamp=$(date +%s)
    
    # Set GPU type - use a common one if auto
    local gpu_type_name=""
    if [[ "$gpu_type" == "auto" ]]; then
        gpu_type_name="NVIDIA RTX A4000"
    else
        gpu_type_name="$gpu_type"
    fi
    
    local create_output=$(runpodctl create pod \
        --name "whisper-transcription-$timestamp" \
        --imageName "$DOCKER_IMAGE" \
        --gpuCount 1 \
        --gpuType "$gpu_type_name" \
        --volumeSize 20 \
        --containerDiskSize 20 \
        --vcpu 2 \
        --mem 8 \
        --ports "8888/http" 2>&1)
    local create_exit_code=$?
    
    # Extract pod ID from runpodctl output
    local pod_id=$(echo "$create_output" | grep -oE '"[a-z0-9]+"' | head -1 | tr -d '"')
    
    if [[ $create_exit_code -ne 0 || -z "$pod_id" ]]; then
        print_error "Failed to launch RunPod container"
        print_error "runpodctl output: $create_output"
        print_error "Exit code: $create_exit_code"
        
        return 1
    fi
    
    print_success "Container launched: $pod_id"
    
    # Wait for container to be ready
    print_status "Waiting for container to start..."
    local ready=false
    local attempts=0
    local max_attempts=60  # 5 minutes
    
    while [[ $ready == false && $attempts -lt $max_attempts ]]; do
        sleep 5
        ((attempts++))
        
        local status_response=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
            "https://api.runpod.io/v2/pods/$pod_id")
        
        local status=$(echo "$status_response" | jq -r '.desiredStatus // empty')
        
        if [[ "$status" == "RUNNING" ]]; then
            ready=true
            print_success "Container is ready"
        else
            printf "."
        fi
    done
    
    if [[ $ready == false ]]; then
        print_error "Container failed to start within 5 minutes"
        exit 1
    fi
    
    echo "$pod_id"
}

# Function to upload files to container
upload_files() {
    local pod_id="$1"
    local archive_path="$2"
    
    print_status "Uploading audio files..."
    
    # Check if runpodctl is available
    if ! command -v runpodctl &> /dev/null; then
        print_error "runpodctl is required but not installed."
        print_step "Install runpodctl:"
        print_step "  pip install runpodctl"
        print_step "  OR download from: https://github.com/runpod/runpodctl/releases"
        exit 1
    fi
    
    # Get archive size for progress display
    local archive_size=$(du -h "$archive_path" | cut -f1)
    print_info "Archive size: $archive_size"
    
    print_status "Sending file to RunPod (this may take several minutes)..."
    
    # Use runpodctl send to upload the file
    local transfer_output=$(runpodctl send "$archive_path" 2>&1)
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        print_error "File upload failed"
        print_error "runpodctl output: $transfer_output"
        exit 1
    fi
    
    # Extract transfer code from output
    local transfer_code=$(echo "$transfer_output" | grep -o '[0-9]\{4\}-[a-z]\+-[a-z]\+-[a-z]\+' | head -1)
    
    if [[ -z "$transfer_code" ]]; then
        print_error "Could not extract transfer code from runpodctl output"
        print_error "Output was: $transfer_output"
        exit 1
    fi
    
    print_success "File uploaded successfully!"
    print_info "Transfer code: ${transfer_code}"
    
    # Now connect to the pod and receive the file
    print_status "Downloading file to container..."
    
    # Execute receive command on the pod
    local receive_cmd="cd /workspace && runpodctl receive $transfer_code"
    local exec_response=$(curl -s -X POST \
        "https://api.runpod.io/v2/pods/$pod_id/exec" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"command\": \"$receive_cmd\",
            \"workingDirectory\": \"/workspace\"
        }")
    
    local exec_id=$(echo "$exec_response" | jq -r '.id // empty')
    
    if [[ -z "$exec_id" || "$exec_id" == "null" ]]; then
        print_error "Failed to execute receive command on container"
        print_error "Response: $exec_response"
        exit 1
    fi
    
    # Wait for receive to complete
    print_status "Waiting for file transfer to container..."
    local completed=false
    local attempts=0
    local max_attempts=60  # 5 minutes max
    
    while [[ $completed == false && $attempts -lt $max_attempts ]]; do
        sleep 5
        ((attempts++))
        
        local exec_status=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
            "https://api.runpod.io/v2/pods/$pod_id/exec/$exec_id")
        
        local status=$(echo "$exec_status" | jq -r '.status // empty')
        
        if [[ "$status" == "COMPLETED" ]]; then
            completed=true
            print_success "File transfer to container completed"
        elif [[ "$status" == "FAILED" ]]; then
            print_error "File transfer to container failed"
            local output=$(echo "$exec_status" | jq -r '.output // empty')
            print_error "Error output: $output"
            exit 1
        else
            printf "."
        fi
    done
    
    if [[ $completed == false ]]; then
        print_error "File transfer to container timed out"
        exit 1
    fi
    
    # Rename the received file to expected name if needed
    local rename_cmd="cd /workspace && if [ -f \"$(basename "$archive_path")\" ]; then mv \"$(basename "$archive_path")\" audio_files.zip; fi"
    curl -s -X POST \
        "https://api.runpod.io/v2/pods/$pod_id/exec" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"command\": \"$rename_cmd\",
            \"workingDirectory\": \"/workspace\"
        }" >/dev/null
    
    print_success "Audio files ready in container"
}

# Function to start transcription job
start_transcription() {
    local pod_id="$1"
    local model="$2"
    
    print_status "Starting transcription job..."
    
    # Execute transcription command via RunPod API
    local exec_response=$(curl -s -X POST \
        "https://api.runpod.io/v2/pods/$pod_id/exec" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"command\": \"python runpod_batch_transcribe.py --model $model\",
            \"workingDirectory\": \"/workspace\"
        }")
    
    local exec_id=$(echo "$exec_response" | jq -r '.id // empty')
    
    if [[ -z "$exec_id" || "$exec_id" == "null" ]]; then
        print_error "Failed to start transcription job"
        exit 1
    fi
    
    print_success "Transcription job started"
    echo "$exec_id"
}

# Function to monitor progress
monitor_progress() {
    local pod_id="$1"
    local exec_id="$2"
    
    print_status "Monitoring transcription progress..."
    
    local completed=false
    local last_progress=""
    
    while [[ $completed == false ]]; do
        sleep $PROGRESS_INTERVAL
        
        # Check execution status
        local exec_status=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
            "https://api.runpod.io/v2/pods/$pod_id/exec/$exec_id")
        
        local status=$(echo "$exec_status" | jq -r '.status // empty')
        
        if [[ "$status" == "COMPLETED" ]]; then
            completed=true
            print_success "Transcription completed!"
            break
        elif [[ "$status" == "FAILED" ]]; then
            print_error "Transcription failed"
            exit 1
        fi
        
        # Try to read progress file (simplified - would need actual file access)
        print_info "Transcription in progress... (checking every ${PROGRESS_INTERVAL}s)"
    done
}

# Function to download results
download_results() {
    local pod_id="$1"
    local output_dir="$2"
    
    print_status "Downloading transcription results..."
    
    # First, create the output zip in the container
    print_status "Creating results archive in container..."
    local create_zip_cmd="cd /workspace && python -c \"
import zipfile
from pathlib import Path
output_dir = Path('/workspace/output')
if output_dir.exists():
    with zipfile.ZipFile('/workspace/transcripts.zip', 'w') as zipf:
        for txt_file in output_dir.glob('*.txt'):
            zipf.write(txt_file, txt_file.name)
    print('Archive created successfully')
else:
    print('No output directory found')
\""
    
    local zip_response=$(curl -s -X POST \
        "https://api.runpod.io/v2/pods/$pod_id/exec" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"command\": \"$create_zip_cmd\",
            \"workingDirectory\": \"/workspace\"
        }")
    
    local zip_exec_id=$(echo "$zip_response" | jq -r '.id // empty')
    
    if [[ -z "$zip_exec_id" || "$zip_exec_id" == "null" ]]; then
        print_error "Failed to create results archive"
        exit 1
    fi
    
    # Wait for zip creation to complete
    local zip_completed=false
    local zip_attempts=0
    local max_zip_attempts=30
    
    while [[ $zip_completed == false && $zip_attempts -lt $max_zip_attempts ]]; do
        sleep 2
        ((zip_attempts++))
        
        local zip_status=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
            "https://api.runpod.io/v2/pods/$pod_id/exec/$zip_exec_id")
        
        local status=$(echo "$zip_status" | jq -r '.status // empty')
        
        if [[ "$status" == "COMPLETED" ]]; then
            zip_completed=true
            print_success "Results archive created"
        elif [[ "$status" == "FAILED" ]]; then
            print_error "Failed to create results archive"
            exit 1
        fi
    done
    
    if [[ $zip_completed == false ]]; then
        print_error "Archive creation timed out"
        exit 1
    fi
    
    # Now send the results file from container
    print_status "Sending results from container..."
    local send_cmd="cd /workspace && runpodctl send transcripts.zip"
    
    local send_response=$(curl -s -X POST \
        "https://api.runpod.io/v2/pods/$pod_id/exec" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"command\": \"$send_cmd\",
            \"workingDirectory\": \"/workspace\"
        }")
    
    local send_exec_id=$(echo "$send_response" | jq -r '.id // empty')
    
    if [[ -z "$send_exec_id" || "$send_exec_id" == "null" ]]; then
        print_error "Failed to start file send from container"
        exit 1
    fi
    
    # Wait for send to complete and get transfer code
    print_status "Waiting for transfer code..."
    local send_completed=false
    local send_attempts=0
    local max_send_attempts=60
    local transfer_code=""
    
    while [[ $send_completed == false && $send_attempts -lt $max_send_attempts ]]; do
        sleep 3
        ((send_attempts++))
        
        local send_status=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
            "https://api.runpod.io/v2/pods/$pod_id/exec/$send_exec_id")
        
        local status=$(echo "$send_status" | jq -r '.status // empty')
        local output=$(echo "$send_status" | jq -r '.output // empty')
        
        if [[ "$status" == "COMPLETED" ]]; then
            send_completed=true
            # Extract transfer code from output
            transfer_code=$(echo "$output" | grep -o '[0-9]\{4\}-[a-z]\+-[a-z]\+-[a-z]\+' | head -1)
            if [[ -n "$transfer_code" ]]; then
                print_success "Transfer code received: $transfer_code"
            else
                print_error "Could not extract transfer code from container output"
                print_error "Output was: $output"
                exit 1
            fi
        elif [[ "$status" == "FAILED" ]]; then
            print_error "File send from container failed"
            print_error "Output: $output"
            exit 1
        else
            printf "."
        fi
    done
    
    if [[ $send_completed == false ]]; then
        print_error "File send from container timed out"
        exit 1
    fi
    
    # Now receive the file locally
    print_status "Receiving results file..."
    
    # Create output directory
    mkdir -p "$output_dir"
    cd "$output_dir"
    
    # Receive the file
    if ! runpodctl receive "$transfer_code"; then
        print_error "Failed to receive results file"
        exit 1
    fi
    
    # Extract the received zip file
    print_status "Extracting transcripts..."
    
    if [[ -f "transcripts.zip" ]]; then
        if unzip -q transcripts.zip; then
            rm transcripts.zip  # Clean up zip file
            print_success "Transcripts extracted successfully"
            
            # Count extracted files
            local transcript_count=$(find . -name "*.txt" -type f | wc -l | tr -d ' ')
            print_info "📄 Extracted $transcript_count transcript files"
        else
            print_error "Failed to extract transcripts.zip"
            exit 1
        fi
    else
        print_error "transcripts.zip not found after download"
        exit 1
    fi
    
    print_success "Results downloaded to: $output_dir"
}

# Function to cleanup
cleanup() {
    local pod_id="$1"
    
    print_status "Cleaning up..."
    
    # Stop and delete the pod
    if [[ -n "$pod_id" ]]; then
        curl -s -X DELETE \
            -H "Authorization: Bearer $RUNPOD_API_KEY" \
            "https://api.runpod.io/v2/pods/$pod_id" >/dev/null
        
        print_success "RunPod container deleted"
    fi
    
    # Remove temporary files
    if [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
        print_success "Temporary files cleaned up"
    fi
}

# Function to estimate cost
estimate_cost() {
    local mp3_count="$1"
    local model="$2"
    
    # Rough estimates based on model speed and GPU costs
    local minutes_per_file=0.5  # turbo model estimate
    case "$model" in
        "tiny"|"base") minutes_per_file=0.2 ;;
        "small") minutes_per_file=0.3 ;;
        "medium") minutes_per_file=0.4 ;;
        "large"|"large-v2"|"large-v3") minutes_per_file=1.0 ;;
        "turbo") minutes_per_file=0.5 ;;
    esac
    
    if command -v bc &> /dev/null; then
        local total_minutes=$(echo "$mp3_count * $minutes_per_file" | bc -l 2>/dev/null || echo "0")
        local hours=$(echo "$total_minutes / 60" | bc -l 2>/dev/null || echo "0")
        local cost=$(echo "$hours * 0.3" | bc -l 2>/dev/null || echo "0")  # Estimate $0.30/hour for A10
        printf "%.1f" "$cost"
    else
        # Fallback calculation without bc
        local total_minutes=$(( mp3_count * minutes_per_file ))
        local hours=$(( (total_minutes + 30) / 60 ))  # Round up
        local cost=$(( hours * 30 / 100 ))  # $0.30/hour in cents, then convert
        if [[ $cost -eq 0 ]]; then
            echo "1"  # Minimum $1
        else
            echo "$cost"
        fi
    fi
}

# Main function
main() {
    local input_dir=""
    local output_dir=""
    local model="turbo"
    local gpu_type="auto"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                show_usage
                exit 0
                ;;
            --model)
                model="$2"
                shift 2
                ;;
            --gpu-type)
                gpu_type="$2"
                shift 2
                ;;
            *)
                if [[ -z "$input_dir" ]]; then
                    input_dir="$1"
                elif [[ -z "$output_dir" ]]; then
                    output_dir="$1"
                else
                    print_error "Unknown argument: $1"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$input_dir" || -z "$output_dir" ]]; then
        print_error "Both input and output directories are required"
        show_usage
        exit 1
    fi
    
    # Convert to absolute paths
    input_dir=$(realpath "$input_dir")
    output_dir=$(realpath "$output_dir")
    
    echo "🎵 RunPod One-Click Audio Transcription"
    echo "======================================="
    echo ""
    echo "📁 Input:  $input_dir"
    echo "📁 Output: $output_dir"
    echo "🤖 Model:  $model"
    echo "🔧 GPU:    $gpu_type"
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Validate input
    validate_input "$input_dir" "$output_dir"
    
    # Count files and estimate cost
    local mp3_count=$(find "$input_dir" -name "*.mp3" -type f | wc -l | tr -d ' ')
    local estimated_cost=$(estimate_cost "$mp3_count" "$model")
    
    print_info "Estimated cost: \$${estimated_cost}"
    print_info "Processing will take approximately $(echo "$mp3_count * 0.5" | bc) minutes"
    echo ""
    
    # Confirm before proceeding
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Operation cancelled"
        exit 0
    fi
    
    local pod_id=""
    local archive_path=""
    
    # Set up cleanup trap
    trap 'cleanup "$pod_id"' EXIT
    
    # Create archive
    archive_path=$(create_audio_archive "$input_dir")
    
    # Launch container
    pod_id=$(launch_runpod_container "$gpu_type")
    
    # Upload files
    upload_files "$pod_id" "$archive_path"
    
    # Start transcription
    local exec_id=$(start_transcription "$pod_id" "$model")
    
    # Monitor progress
    monitor_progress "$pod_id" "$exec_id"
    
    # Download results
    download_results "$pod_id" "$output_dir"
    
    # Show final summary
    echo ""
    print_success "🎉 Transcription complete!"
    print_info "📊 Processed: $mp3_count files"
    print_info "📁 Results: $output_dir"
    print_info "💰 Estimated cost: \$${estimated_cost}"
    
    # Cleanup happens automatically via trap
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi