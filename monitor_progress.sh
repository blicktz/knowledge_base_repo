#!/bin/bash
# RunPod Progress Monitor - Real-time transcription progress tracking

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
RUNPOD_CONFIG_FILE="$HOME/.runpod_config"
UPDATE_INTERVAL=15  # seconds

print_header() {
    clear
    echo -e "${BLUE}"
    echo "🎵 RunPod Transcription Progress Monitor"
    echo "======================================="
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

print_progress() {
    echo -e "${PURPLE}🎵${NC} $1"
}

format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    
    if [[ $hours -gt 0 ]]; then
        printf "%dh %dm %ds" $hours $minutes $secs
    elif [[ $minutes -gt 0 ]]; then
        printf "%dm %ds" $minutes $secs
    else
        printf "%ds" $secs
    fi
}

draw_progress_bar() {
    local progress=$1
    local width=50
    local filled=$((progress * width / 100))
    local empty=$((width - filled))
    
    printf "["
    printf "%*s" $filled | tr ' ' '='
    if [[ $filled -lt $width ]]; then
        printf ">"
        ((empty--))
    fi
    printf "%*s" $empty | tr ' ' '-'
    printf "] %3d%%" $progress
}

get_pod_progress() {
    local pod_id="$1"
    
    # Execute command to read progress file from container
    local exec_response=$(curl -s -X POST \
        "https://api.runpod.io/v2/pods/$pod_id/exec" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"command": "cat /workspace/progress.json 2>/dev/null || echo \"{}\"", "workingDirectory": "/workspace"}' 2>/dev/null)
    
    local exec_id=$(echo "$exec_response" | jq -r '.id // empty' 2>/dev/null)
    
    if [[ -z "$exec_id" || "$exec_id" == "null" ]]; then
        echo "{}"
        return
    fi
    
    # Wait a moment for command to complete
    sleep 2
    
    # Get the execution result
    local exec_status=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
        "https://api.runpod.io/v2/pods/$pod_id/exec/$exec_id" 2>/dev/null)
    
    local status=$(echo "$exec_status" | jq -r '.status // empty' 2>/dev/null)
    local output=$(echo "$exec_status" | jq -r '.output // empty' 2>/dev/null)
    
    if [[ "$status" == "COMPLETED" && -n "$output" && "$output" != "empty" && "$output" != "{}" ]]; then
        # Try to parse as JSON
        if echo "$output" | jq . >/dev/null 2>&1; then
            echo "$output"
        else
            echo "{}"
        fi
    else
        echo "{}"
    fi
}

monitor_pod() {
    local pod_id="$1"
    
    if [[ -z "$pod_id" ]]; then
        print_error "Pod ID is required"
        echo "Usage: $0 <pod_id>"
        exit 1
    fi
    
    source "$RUNPOD_CONFIG_FILE" 2>/dev/null || {
        print_error "RunPod configuration not found. Run: ./setup_runpod.sh"
        exit 1
    }
    
    local start_time=$(date +%s)
    local last_progress=0
    local eta_displayed=false
    
    print_status "Monitoring pod: $pod_id"
    print_status "Updates every ${UPDATE_INTERVAL} seconds (press Ctrl+C to exit)"
    echo ""
    
    while true; do
        print_header
        echo "Pod ID: ${CYAN}$pod_id${NC}"
        echo "Started: $(date -r $start_time '+%Y-%m-%d %H:%M:%S')"
        echo "Elapsed: $(format_time $(($(date +%s) - start_time)))"
        echo ""
        
        # Get pod status
        local pod_status=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
            "https://api.runpod.io/v2/pods/$pod_id" 2>/dev/null)
        
        if command -v jq &> /dev/null && [[ -n "$pod_status" ]]; then
            local status=$(echo "$pod_status" | jq -r '.desiredStatus // "UNKNOWN"')
            local gpu_type=$(echo "$pod_status" | jq -r '.machine.gpuDisplayName // "Unknown"')
            
            echo "Status: $status"
            echo "GPU: $gpu_type"
            echo ""
        fi
        
        # Try to get transcription progress
        local progress_data=$(get_pod_progress "$pod_id")
        
        if command -v jq &> /dev/null && [[ -n "$progress_data" ]] && [[ "$progress_data" != "{}" ]]; then
            local current_file=$(echo "$progress_data" | jq -r '.current_file // 0')
            local total_files=$(echo "$progress_data" | jq -r '.total_files // 0')
            local processed=$(echo "$progress_data" | jq -r '.processed // 0')
            local failed=$(echo "$progress_data" | jq -r '.failed // 0')
            local current_filename=$(echo "$progress_data" | jq -r '.current_filename // "Unknown"')
            local percent_complete=$(echo "$progress_data" | jq -r '.percent_complete // 0')
            local eta_minutes=$(echo "$progress_data" | jq -r '.eta_minutes // 0')
            local error_msg=$(echo "$progress_data" | jq -r '.error // empty')
            
            if [[ "$total_files" -gt 0 ]]; then
                print_progress "Processing file $current_file of $total_files"
                echo "Current: ${CYAN}$current_filename${NC}"
                echo ""
                
                # Progress bar
                local progress_int=$(printf "%.0f" "$percent_complete")
                echo -n "Progress: "
                draw_progress_bar $progress_int
                echo ""
                echo ""
                
                # Statistics
                echo "📊 Statistics:"
                echo "  ✅ Completed: $processed"
                if [[ "$failed" -gt 0 ]]; then
                    echo "  ❌ Failed: $failed"
                fi
                echo "  📁 Remaining: $((total_files - current_file))"
                
                # ETA
                if [[ "$eta_minutes" != "0" && "$eta_minutes" != "null" ]]; then
                    local eta_formatted=$(format_time $(echo "$eta_minutes * 60" | bc 2>/dev/null || echo "0"))
                    echo "  ⏱️  ETA: $eta_formatted"
                    eta_displayed=true
                fi
                
                # Error display
                if [[ -n "$error_msg" ]]; then
                    echo ""
                    print_error "Last error: $error_msg"
                fi
                
                # Check if completed
                if [[ "$current_file" -eq "$total_files" ]] && [[ "$processed" -gt 0 ]]; then
                    echo ""
                    print_success "🎉 Transcription completed!"
                    print_success "📊 Final stats: $processed processed, $failed failed"
                    
                    # Estimate cost
                    local total_time=$(($(date +%s) - start_time))
                    local hours=$(echo "scale=2; $total_time / 3600" | bc 2>/dev/null || echo "0")
                    local estimated_cost=$(echo "scale=2; $hours * 0.3" | bc 2>/dev/null || echo "0")
                    
                    print_status "💰 Estimated cost: \$${estimated_cost}"
                    print_status "📁 Download your results from the RunPod console"
                    break
                fi
            fi
        else
            # No progress data available yet
            print_warning "Waiting for transcription to start..."
            echo ""
            echo "This usually takes 1-2 minutes while the container:"
            echo "  • Loads the Whisper model"
            echo "  • Extracts audio files"
            echo "  • Begins processing"
            
            if [[ ! $eta_displayed ]]; then
                echo ""
                print_status "📊 Initial processing may take a few minutes to show progress"
            fi
        fi
        
        echo ""
        echo "Last updated: $(date '+%H:%M:%S')"
        echo "Press Ctrl+C to exit monitor (transcription will continue)"
        
        sleep $UPDATE_INTERVAL
    done
}

show_usage() {
    echo "RunPod Progress Monitor"
    echo "======================"
    echo ""
    echo "Usage: $0 <pod_id>"
    echo ""
    echo "Monitor the progress of a RunPod transcription job in real-time."
    echo ""
    echo "Examples:"
    echo "  $0 abc123def456  # Monitor pod with ID abc123def456"
    echo ""
    echo "The pod ID is displayed when you start a transcription job."
    echo "You can also find it in the RunPod console."
}

main() {
    if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        show_usage
        exit 0
    fi
    
    local pod_id="$1"
    
    # Validate pod ID format (basic check)
    if [[ ! "$pod_id" =~ ^[a-zA-Z0-9]+$ ]] || [[ ${#pod_id} -lt 8 ]]; then
        print_error "Invalid pod ID format: $pod_id"
        echo ""
        show_usage
        exit 1
    fi
    
    # Check if config exists
    if [[ ! -f "$RUNPOD_CONFIG_FILE" ]]; then
        print_error "RunPod configuration not found"
        print_status "Please run: ./setup_runpod.sh"
        exit 1
    fi
    
    monitor_pod "$pod_id"
}

# Handle Ctrl+C gracefully
trap 'echo ""; print_status "Monitoring stopped. Transcription continues in the background."; exit 0' INT

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi