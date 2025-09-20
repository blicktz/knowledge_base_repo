#!/bin/bash
# Test script for RunPod transcription workflow
# Creates sample audio files and tests the complete pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

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

print_test() {
    echo -e "${PURPLE}🧪${NC} $1"
}

# Test directories
TEST_DIR="./test_audio"
TEST_OUTPUT="./test_transcripts"
SAMPLE_COUNT=3

cleanup_test() {
    print_status "Cleaning up test files..."
    rm -rf "$TEST_DIR" "$TEST_OUTPUT"
    print_success "Test cleanup complete"
}

create_test_audio() {
    print_status "Creating test audio files..."
    
    mkdir -p "$TEST_DIR"
    
    # Check if we can create test audio files
    if command -v say &> /dev/null; then
        # macOS text-to-speech
        print_status "Using macOS 'say' command to create test audio..."
        
        echo "This is test episode one. This is a sample podcast transcript for testing the RunPod audio transcription system." | say -o "$TEST_DIR/episode_001.mp3" --data-format=mp3 2>/dev/null || {
            print_warning "Failed to create MP3 with say command"
            return 1
        }
        
        echo "This is test episode two. We are testing the automatic speech recognition capabilities with multiple files in a batch." | say -o "$TEST_DIR/episode_002.mp3" --data-format=mp3 2>/dev/null || {
            print_warning "Failed to create second MP3"
            return 1
        }
        
        echo "This is the third and final test episode. The system should process all episodes and return transcribed text files." | say -o "$TEST_DIR/episode_003.mp3" --data-format=mp3 2>/dev/null || {
            print_warning "Failed to create third MP3"
            return 1
        }
        
        print_success "Created 3 test MP3 files"
        return 0
        
    elif command -v espeak &> /dev/null && command -v ffmpeg &> /dev/null; then
        # Linux text-to-speech with espeak and ffmpeg
        print_status "Using espeak + ffmpeg to create test audio..."
        
        espeak "This is test episode one. This is a sample podcast transcript for testing the RunPod audio transcription system." -w "$TEST_DIR/episode_001.wav"
        ffmpeg -i "$TEST_DIR/episode_001.wav" -codec:a mp3 "$TEST_DIR/episode_001.mp3" -y &>/dev/null
        rm "$TEST_DIR/episode_001.wav"
        
        espeak "This is test episode two. We are testing the automatic speech recognition capabilities with multiple files in a batch." -w "$TEST_DIR/episode_002.wav"
        ffmpeg -i "$TEST_DIR/episode_002.wav" -codec:a mp3 "$TEST_DIR/episode_002.mp3" -y &>/dev/null
        rm "$TEST_DIR/episode_002.wav"
        
        espeak "This is the third and final test episode. The system should process all episodes and return transcribed text files." -w "$TEST_DIR/episode_003.wav"
        ffmpeg -i "$TEST_DIR/episode_003.wav" -codec:a mp3 "$TEST_DIR/episode_003.mp3" -y &>/dev/null
        rm "$TEST_DIR/episode_003.wav"
        
        print_success "Created 3 test MP3 files"
        return 0
    else
        print_error "Cannot create test audio files automatically"
        print_warning "Please manually place 2-3 MP3 files in: $TEST_DIR"
        print_warning "Then run this test script again"
        return 1
    fi
}

test_prerequisites() {
    print_test "Testing prerequisites..."
    
    # Test each required command
    local missing_deps=()
    
    if ! command -v jq &> /dev/null; then
        missing_deps+=("jq")
    fi
    
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi
    
    if ! command -v zip &> /dev/null; then
        missing_deps+=("zip")
    fi
    
    if ! command -v unzip &> /dev/null; then
        missing_deps+=("unzip")
    fi
    
    if ! command -v runpodctl &> /dev/null; then
        missing_deps+=("runpodctl")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_status "Please install missing dependencies and run setup:"
        print_status "  ./setup_runpod.sh"
        return 1
    fi
    
    # Check configuration
    if [[ ! -f "$HOME/.runpod_config" ]]; then
        print_error "RunPod configuration not found"
        print_status "Please run setup first: ./setup_runpod.sh"
        return 1
    fi
    
    print_success "All prerequisites OK"
    return 0
}

test_dry_run() {
    print_test "Testing dry run (validation only)..."
    
    # Test script execution without actually running
    if ./runpod_transcribe.sh --help >/dev/null 2>&1; then
        print_success "Help command works"
    else
        print_error "Help command failed"
        return 1
    fi
    
    return 0
}

run_full_test() {
    print_test "Running full transcription test..."
    
    print_warning "This will actually launch a RunPod container and incur costs (~$0.10-0.50)"
    read -p "Continue with full test? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Full test skipped"
        return 0
    fi
    
    print_status "Starting full transcription test..."
    print_status "Input: $TEST_DIR"
    print_status "Output: $TEST_OUTPUT"
    
    # Run the actual transcription
    if ./runpod_transcribe.sh "$TEST_DIR" "$TEST_OUTPUT" --model tiny; then
        print_success "Transcription completed successfully!"
        
        # Verify output
        if [[ -d "$TEST_OUTPUT" ]]; then
            local transcript_count=$(find "$TEST_OUTPUT" -name "*.txt" -type f | wc -l | tr -d ' ')
            if [[ $transcript_count -gt 0 ]]; then
                print_success "Found $transcript_count transcript files"
                
                # Show sample content
                print_status "Sample transcript content:"
                local first_transcript=$(find "$TEST_OUTPUT" -name "*.txt" -type f | head -1)
                if [[ -f "$first_transcript" ]]; then
                    echo "---"
                    head -3 "$first_transcript"
                    echo "---"
                fi
                
                return 0
            else
                print_error "No transcript files found in output"
                return 1
            fi
        else
            print_error "Output directory not created"
            return 1
        fi
    else
        print_error "Transcription failed"
        return 1
    fi
}

show_usage() {
    echo "RunPod Transcription Test Suite"
    echo "==============================="
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --create-audio    Create test audio files only"
    echo "  --check-deps      Check dependencies only"
    echo "  --dry-run         Test script validation only"
    echo "  --full-test       Run complete transcription test (costs money!)"
    echo "  --cleanup         Remove test files"
    echo "  --help            Show this help"
    echo ""
    echo "Default: Run all tests except full transcription test"
}

main() {
    echo "🧪 RunPod Transcription Test Suite"
    echo "==================================="
    echo ""
    
    case "${1:-}" in
        --create-audio)
            create_test_audio
            exit $?
            ;;
        --check-deps)
            test_prerequisites
            exit $?
            ;;
        --dry-run)
            test_dry_run
            exit $?
            ;;
        --full-test)
            if ! test_prerequisites; then exit 1; fi
            if ! create_test_audio; then exit 1; fi
            run_full_test
            exit $?
            ;;
        --cleanup)
            cleanup_test
            exit 0
            ;;
        --help)
            show_usage
            exit 0
            ;;
        "")
            # Default: run all tests except full test
            print_status "Running standard test suite..."
            echo ""
            
            if ! test_prerequisites; then
                print_error "Prerequisites test failed"
                exit 1
            fi
            echo ""
            
            if ! test_dry_run; then
                print_error "Dry run test failed"
                exit 1
            fi
            echo ""
            
            if ! create_test_audio; then
                print_error "Audio creation failed"
                exit 1
            fi
            echo ""
            
            print_success "🎉 All tests passed!"
            print_status "Your RunPod transcription system is ready to use"
            echo ""
            print_warning "To test with actual RunPod (costs ~$0.10-0.50):"
            print_status "  $0 --full-test"
            echo ""
            print_status "To clean up test files:"
            print_status "  $0 --cleanup"
            echo ""
            print_status "To transcribe your real audio files:"
            print_status "  ./runpod_transcribe.sh /path/to/your/mp3s /path/to/output"
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Set up cleanup trap
trap 'echo ""; print_status "Test interrupted"' INT

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi