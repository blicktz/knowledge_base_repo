.PHONY: help install convert test clean example batch batch-custom setup-dirs docker-build docker-deploy runpod-create runpod-info runpod-stop runpod-start runpod-delete runpod-logs
.DEFAULT_GOAL := help

# Variables
PYTHON := poetry run python
SCRIPT := pdf2text/pdf_to_markdown.py
VENV := .venv

# Default directories for batch processing
PDF_INPUT := /Volumes/J15/copy-writing/dk_books_pdf
MD_OUTPUT := ./dk_books

# Docker and RunPod variables
DOCKER_USERNAME ?= blickt123
DOCKER_IMAGE_NAME := whisper-transcription
DOCKER_TAG ?= latest
DOCKER_FULL_NAME := $(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)
RUNPOD_POD_NAME ?= whisper-transcription-$(shell date +%s)
RUNPOD_GPU_TYPE ?= NVIDIA RTX A5000
TRANSCRIBE_API_KEY_ENV := mv_mtvG2X4U_dqRgdWMvSEoFtpMjRJkL4zlkwEXYH2I

help: ## Show this help message
	@echo "PDF to Markdown Converter"
	@echo "=========================="
	@echo ""
	@echo "Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Single file examples:"
	@echo "  make convert INPUT=document.pdf"
	@echo "  make convert INPUT=document.pdf OUTPUT=output.md"
	@echo "  make convert INPUT=document.pdf PAGES=0,2-4"
	@echo "  make convert INPUT=document.pdf IMAGES=true"
	@echo ""
	@echo "Batch processing examples:"
	@echo "  make setup-dirs    # Create default directories"
	@echo "  make batch         # Convert all PDFs in $(PDF_INPUT)/"
	@echo "  make batch-custom INPUT_DIR=my_pdfs OUTPUT_DIR=my_md"
	@echo ""
	@echo "Audio transcription examples:"
	@echo "  make audio-setup   # Verify audio setup"
	@echo "  make audio-transcribe INPUT=podcast.mp3"
	@echo "  make audio-batch INPUT_DIR=podcasts OUTPUT_DIR=transcripts"
	@echo ""
	@echo "RunPod deployment examples:"
	@echo "  make docker-deploy DOCKER_USERNAME=myuser  # Build and push to Docker Hub"
	@echo "  make runpod-create    # Create and deploy RunPod instance"
	@echo "  make runpod-info      # Get pod status and connection URL"

install: ## Install dependencies using Poetry
	@echo "Installing dependencies..."
	poetry install
	@echo "✓ Dependencies installed!"

setup: install ## Complete setup (install dependencies)
	@echo "✓ Setup complete! You can now convert PDFs to Markdown."
	@echo ""
	@echo "Try: make convert INPUT=your_document.pdf"

convert: ## Convert PDF to Markdown (requires INPUT=file.pdf)
ifndef INPUT
	@echo "Error: INPUT parameter required"
	@echo "Usage: make convert INPUT=document.pdf [OUTPUT=output.md] [PAGES=0,2-4] [IMAGES=true]"
	@exit 1
endif
	@echo "Converting $(INPUT) to Markdown..."
	@cmd="$(PYTHON) $(SCRIPT) '$(INPUT)'"; \
	if [ -n "$(OUTPUT)" ]; then cmd="$$cmd -o '$(OUTPUT)'"; fi; \
	if [ -n "$(PAGES)" ]; then cmd="$$cmd -p '$(PAGES)'"; fi; \
	if [ "$(IMAGES)" = "true" ]; then cmd="$$cmd -i"; fi; \
	eval $$cmd

quick: ## Quick convert - just specify filename (e.g., make quick FILE=doc.pdf)
ifndef FILE
	@echo "Error: FILE parameter required"
	@echo "Usage: make quick FILE=document.pdf"
	@exit 1
endif
	@echo "Converting $(FILE) to Markdown..."
	$(PYTHON) $(SCRIPT) "$(FILE)"

save: ## Convert and save to .md file (e.g., make save FILE=doc.pdf)
ifndef FILE
	@echo "Error: FILE parameter required"
	@echo "Usage: make save FILE=document.pdf"
	@exit 1
endif
	@output_file="$$(basename '$(FILE)' .pdf).md"; \
	echo "Converting $(FILE) to $$output_file..."; \
	$(PYTHON) $(SCRIPT) "$(FILE)" -o "$$output_file"

test: install ## Test the script with help command
	@echo "Testing PDF to Markdown converter..."
	$(PYTHON) $(SCRIPT) --help
	@echo ""
	@echo "✓ Script is working! Ready to convert PDFs."

clean: ## Clean up generated files
	@echo "Cleaning up..."
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	rm -rf dist/
	rm -rf build/
	rm -rf images/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "✓ Cleaned up!"

example: ## Show usage examples
	@echo "PDF to Markdown Converter - Usage Examples"
	@echo "=========================================="
	@echo ""
	@echo "1. Convert PDF and print to terminal:"
	@echo "   make convert INPUT=document.pdf"
	@echo ""
	@echo "2. Convert PDF and save to file:"
	@echo "   make convert INPUT=document.pdf OUTPUT=result.md"
	@echo ""
	@echo "3. Convert specific pages:"
	@echo "   make convert INPUT=document.pdf PAGES=0,2-4,7"
	@echo ""
	@echo "4. Convert with image extraction:"
	@echo "   make convert INPUT=document.pdf IMAGES=true"
	@echo ""
	@echo "5. Quick convert (print to terminal):"
	@echo "   make quick FILE=document.pdf"
	@echo ""
	@echo "6. Quick save (auto-generate .md filename):"
	@echo "   make save FILE=document.pdf"
	@echo ""
	@echo "7. Direct Python usage:"
	@echo "   poetry run python pdf2text/pdf_to_markdown.py document.pdf -o output.md"

setup-dirs: ## Create default input/output directories
	@echo "Creating default directories..."
	mkdir -p $(PDF_INPUT)
	mkdir -p $(MD_OUTPUT)
	@echo "✓ Created directories:"
	@echo "  Input:  $(PDF_INPUT)/"
	@echo "  Output: $(MD_OUTPUT)/"
	@echo ""
	@echo "You can now place PDF files in $(PDF_INPUT)/ and run 'make batch'"

batch: setup-dirs ## Batch convert all PDFs in default input folder
	@echo "Converting all PDFs from $(PDF_INPUT)/ to $(MD_OUTPUT)/"
	$(PYTHON) $(SCRIPT) --batch --input-dir $(PDF_INPUT) --output-dir $(MD_OUTPUT)

batch-custom: ## Batch convert with custom folders (INPUT_DIR=... OUTPUT_DIR=...)
ifndef INPUT_DIR
	@echo "Error: INPUT_DIR parameter required"
	@echo "Usage: make batch-custom INPUT_DIR=my_pdfs OUTPUT_DIR=my_markdown"
	@exit 1
endif
ifndef OUTPUT_DIR
	@echo "Error: OUTPUT_DIR parameter required"
	@echo "Usage: make batch-custom INPUT_DIR=my_pdfs OUTPUT_DIR=my_markdown"
	@exit 1
endif
	@echo "Converting all PDFs from $(INPUT_DIR)/ to $(OUTPUT_DIR)/"
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) $(SCRIPT) --batch --input-dir $(INPUT_DIR) --output-dir $(OUTPUT_DIR)

env-info: ## Show environment information
	@echo "Environment Information"
	@echo "======================"
	@echo "Python version: $$(poetry run python --version)"
	@echo "Poetry version: $$(poetry --version)"
	@echo "Default directories:"
	@echo "  PDF Input:  $(PDF_INPUT)/"
	@echo "  MD Output:  $(MD_OUTPUT)/"
	@echo "Project dependencies:"
	@poetry show --tree

dk-setup: ## Setup DK RAG copywriting assistant
	@echo "Setting up DK RAG Copywriting Assistant..."
	@echo "=========================================="
	@if [ ! -f ".env" ]; then \
		echo "⚠️  No .env file found. Creating from template..."; \
		cp .env.example .env; \
		echo "📝 Please edit .env file with your API keys:"; \
		echo "   - OPENROUTER_API_KEY (get from https://openrouter.ai/keys)"; \
		echo "   - OPENAI_API_KEY (get from https://platform.openai.com/api-keys)"; \
		echo ""; \
		echo "After setting up API keys, run 'make dk-setup' again."; \
		exit 1; \
	fi
	@if [ ! -d "$(MD_OUTPUT)" ]; then echo "Error: MD_OUTPUT directory not found: $(MD_OUTPUT)"; exit 1; fi
	@echo "✓ Found markdown documents directory"
	@if [ -z "$$(grep '^OPENROUTER_API_KEY=' .env | cut -d= -f2)" ]; then \
		echo "❌ OPENROUTER_API_KEY not set in .env file"; \
		exit 1; \
	fi
	@if [ -z "$$(grep '^OPENAI_API_KEY=' .env | cut -d= -f2)" ]; then \
		echo "❌ OPENAI_API_KEY not set in .env file"; \
		exit 1; \
	fi
	@echo "✓ API keys configured"
	$(PYTHON) dk_rag/generate_copy.py --setup-only --documents-dir $(MD_OUTPUT)
	@echo "✓ DK RAG setup complete!"

dk-generate: ## Generate copy with DK assistant (TASK="your task here")
ifndef TASK
	@echo "Error: TASK parameter required"
	@echo "Usage: make dk-generate TASK=\"Write an email for a webinar about marketing\""
	@exit 1
endif
	@echo "Generating copy with DK assistant..."
	$(PYTHON) dk_rag/generate_copy.py "$(TASK)" --documents-dir $(MD_OUTPUT)

dk-interactive: ## Run DK assistant in interactive mode
	@echo "Starting DK assistant in interactive mode..."
	$(PYTHON) dk_rag/generate_copy.py --interactive --documents-dir $(MD_OUTPUT)

dk-rebuild: ## Rebuild DK knowledge base
	@echo "Rebuilding DK knowledge base..."
	$(PYTHON) dk_rag/generate_copy.py --setup-only --rebuild-kb --documents-dir $(MD_OUTPUT)

dk-test: ## Test DK assistant with sample task
	@echo "Testing DK assistant with sample task..."
	$(PYTHON) dk_rag/generate_copy.py "Write a short email promoting a marketing webinar for developers" --documents-dir $(MD_OUTPUT)

dk-debug: ## Test with debug mode enabled (TASK="your task here")
ifndef TASK
	@echo "Error: TASK parameter required"
	@echo "Usage: make dk-debug TASK=\"Write an email for a webinar about marketing\""
	@exit 1
endif
	@echo "Generating copy with DK assistant (DEBUG MODE)..."
	$(PYTHON) dk_rag/generate_copy.py "$(TASK)" --documents-dir $(MD_OUTPUT) --debug

# Audio Transcription Targets
# ============================

audio-setup: ## Verify audio transcription setup
	@echo "Checking audio transcription setup..."
	@echo "=========================================="
	@if ! command -v ffmpeg >/dev/null 2>&1; then \
		echo "❌ ffmpeg not found. Install with: brew install ffmpeg"; \
		exit 1; \
	fi
	@echo "✓ ffmpeg installed"
	@echo "Testing Whisper model loading..."
	$(PYTHON) audio2text/audio_to_text.py --setup-only
	@echo "✓ Audio transcription setup complete!"

audio-transcribe: ## Transcribe single audio file (INPUT=file.mp3 [OUTPUT=file.txt])
ifndef INPUT
	@echo "Error: INPUT parameter required"
	@echo "Usage: make audio-transcribe INPUT=podcast.mp3 [OUTPUT=transcript.txt]"
	@exit 1
endif
	@echo "Transcribing $(INPUT)..."
	@cmd="$(PYTHON) audio2text/audio_to_text.py '$(INPUT)'"; \
	if [ -n "$(OUTPUT)" ]; then cmd="$$cmd -o '$(OUTPUT)'"; fi; \
	if [ -n "$(MODEL)" ]; then cmd="$$cmd --model '$(MODEL)'"; fi; \
	eval $$cmd

audio-quick: ## Quick transcribe to auto-named file (FILE=audio.mp3)
ifndef FILE
	@echo "Error: FILE parameter required"
	@echo "Usage: make audio-quick FILE=podcast.mp3"
	@exit 1
endif
	@output_file="$$(basename '$(FILE)' .mp3).txt"; \
	echo "Transcribing $(FILE) to $$output_file..."; \
	$(PYTHON) audio2text/audio_to_text.py "$(FILE)" -o "$$output_file"

audio-batch: ## Batch transcribe MP3 files (INPUT_DIR=... OUTPUT_DIR=...)
ifndef INPUT_DIR
	@echo "Error: INPUT_DIR parameter required"
	@echo "Usage: make audio-batch INPUT_DIR=/path/to/mp3s OUTPUT_DIR=/path/to/texts"
	@exit 1
endif
ifndef OUTPUT_DIR
	@echo "Error: OUTPUT_DIR parameter required"
	@echo "Usage: make audio-batch INPUT_DIR=/path/to/mp3s OUTPUT_DIR=/path/to/texts"
	@exit 1
endif
	@echo "Batch transcribing MP3 files..."
	@echo "Input:  $(INPUT_DIR)/"
	@echo "Output: $(OUTPUT_DIR)/"
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) audio2text/audio_to_text.py --batch --input-dir "$(INPUT_DIR)" --output-dir "$(OUTPUT_DIR)"

audio-batch-model: ## Batch transcribe with specific model (INPUT_DIR=... OUTPUT_DIR=... MODEL=...)
ifndef INPUT_DIR
	@echo "Error: INPUT_DIR parameter required"
	@echo "Usage: make audio-batch-model INPUT_DIR=/path/to/mp3s OUTPUT_DIR=/path/to/texts MODEL=medium"
	@exit 1
endif
ifndef OUTPUT_DIR
	@echo "Error: OUTPUT_DIR parameter required"
	@echo "Usage: make audio-batch-model INPUT_DIR=/path/to/mp3s OUTPUT_DIR=/path/to/texts MODEL=medium"
	@exit 1
endif
ifndef MODEL
	@echo "Error: MODEL parameter required"
	@echo "Available models: tiny, base, small, medium, large, large-v2, large-v3, turbo, large-v3-turbo"
	@exit 1
endif
	@echo "Batch transcribing with model: $(MODEL)"
	@echo "Input:  $(INPUT_DIR)/"
	@echo "Output: $(OUTPUT_DIR)/"
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) audio2text/audio_to_text.py --batch --input-dir "$(INPUT_DIR)" --output-dir "$(OUTPUT_DIR)" --model "$(MODEL)"

audio-test: ## Test transcription with sample audio
	@echo "Running audio transcription test..."
	@if [ -f "test_audio.mp3" ]; then \
		$(PYTHON) audio2text/audio_to_text.py test_audio.mp3 -o test_transcript.txt; \
		echo "✓ Test complete! Check test_transcript.txt"; \
	else \
		echo "Please provide a test_audio.mp3 file"; \
		echo "You can download a sample podcast or create a short recording"; \
	fi

audio-info: ## Show audio transcription settings and model info
	@echo "Audio Transcription Configuration"
	@echo "================================="
	@echo "Script: audio2text/audio_to_text.py"
	@echo "Default model: large-v3"
	@echo "Available models:"
	@echo "  tiny:     ~39 MB, fastest, lowest accuracy"
	@echo "  base:     ~74 MB"
	@echo "  small:    ~244 MB"
	@echo "  medium:   ~769 MB"
	@echo "  large:    ~1550 MB"
	@echo "  large-v2: ~1550 MB, better accuracy"
	@echo "  large-v3: ~1550 MB, best accuracy (default)
  turbo:    ~1550 MB, 8x faster than large-v3, near same quality"
	@echo ""
	@echo "Device support:"
	@$(PYTHON) -c "import torch; print(f'  MPS (Metal): {torch.backends.mps.is_available()}')"
	@$(PYTHON) -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}')"
	@echo ""
	@echo "Example usage:"
	@echo "  make audio-batch INPUT_DIR=/path/to/podcasts OUTPUT_DIR=/path/to/transcripts"

audio-examples: ## Show audio transcription examples
	@echo "Audio Transcription - Usage Examples"
	@echo "===================================="
	@echo ""
	@echo "1. Transcribe single file to terminal:"
	@echo "   make audio-transcribe INPUT=podcast.mp3"
	@echo ""
	@echo "2. Transcribe single file to specific output:"
	@echo "   make audio-transcribe INPUT=episode1.mp3 OUTPUT=episode1.txt"
	@echo ""
	@echo "3. Quick transcribe with auto-generated filename:"
	@echo "   make audio-quick FILE=podcast.mp3"
	@echo ""
	@echo "4. Batch transcribe entire directory:"
	@echo "   make audio-batch INPUT_DIR=/Users/you/podcasts OUTPUT_DIR=/Users/you/transcripts"
	@echo ""
	@echo "5. Batch with specific model (faster but less accurate):"
	@echo "   make audio-batch-model INPUT_DIR=./mp3s OUTPUT_DIR=./texts MODEL=medium"
	@echo ""
	@echo "6. Format existing transcript files:"
	@echo "   make format-text INPUT=transcript.txt OUTPUT=formatted.txt"
	@echo "   make format-batch INPUT_DIR=./transcripts OUTPUT_DIR=./formatted"
	@echo ""
	@echo "7. Direct Python usage:"
	@echo "   poetry run python audio2text/audio_to_text.py --batch \\"
	@echo "     --input-dir /path/to/mp3s --output-dir /path/to/texts"

# Text Formatting Targets
# ========================

format-text: ## Format single text file with semantic paragraphs (INPUT=file.txt [OUTPUT=formatted.txt])
ifndef INPUT
	@echo "Error: INPUT parameter required"
	@echo "Usage: make format-text INPUT=transcript.txt [OUTPUT=formatted.txt]"
	@exit 1
endif
	@echo "Formatting text file: $(INPUT)"
	@cmd="$(PYTHON) audio2text/format_text.py '$(INPUT)'"; \
	if [ -n "$(OUTPUT)" ]; then cmd="$$cmd -o '$(OUTPUT)'"; fi; \
	if [ -n "$(DEVICE)" ]; then cmd="$$cmd --device '$(DEVICE)'"; fi; \
	eval $$cmd

format-batch: ## Batch format text files (INPUT_DIR=... OUTPUT_DIR=...)
ifndef INPUT_DIR
	@echo "Error: INPUT_DIR parameter required"
	@echo "Usage: make format-batch INPUT_DIR=/path/to/texts OUTPUT_DIR=/path/to/formatted"
	@exit 1
endif
ifndef OUTPUT_DIR
	@echo "Error: OUTPUT_DIR parameter required"
	@echo "Usage: make format-batch INPUT_DIR=/path/to/texts OUTPUT_DIR=/path/to/formatted"
	@exit 1
endif
	@echo "Batch formatting text files..."
	@echo "Input:  $(INPUT_DIR)/"
	@echo "Output: $(OUTPUT_DIR)/"
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) audio2text/format_text.py --batch --input-dir "$(INPUT_DIR)" --output-dir "$(OUTPUT_DIR)"

format-inplace: ## Format text file in place (overwrites original) (INPUT=file.txt)
ifndef INPUT
	@echo "Error: INPUT parameter required"
	@echo "Usage: make format-inplace INPUT=transcript.txt"
	@exit 1
endif
	@echo "Formatting text file in place: $(INPUT)"
	$(PYTHON) audio2text/format_text.py "$(INPUT)"

format-test: ## Test formatting on sample text file
	@if [ -f "1.txt" ]; then \
		echo "Testing text formatting on 1.txt..."; \
		$(PYTHON) audio2text/format_text.py 1.txt -o 1_formatted.txt; \
		echo "✓ Test complete! Check 1_formatted.txt"; \
	else \
		echo "Please provide a 1.txt file to test formatting"; \
	fi

# Safe Audio Processing Targets (handles problematic filenames)
# ==============================================================


audio-batch-safe: ## Batch transcribe with safe filenames (INPUT_DIR=... OUTPUT_DIR=...)
ifndef INPUT_DIR
	@echo "Error: INPUT_DIR parameter required"
	@echo "Usage: make audio-batch-safe INPUT_DIR=/path/to/mp3s OUTPUT_DIR=/path/to/texts"
	@exit 1
endif
ifndef OUTPUT_DIR
	@echo "Error: OUTPUT_DIR parameter required"
	@echo "Usage: make audio-batch-safe INPUT_DIR=/path/to/mp3s OUTPUT_DIR=/path/to/texts"
	@exit 1
endif
	@echo "Batch transcribing with safe filename generation..."
	@echo "Input:  $(INPUT_DIR)/"
	@echo "Output: $(OUTPUT_DIR)/"
	@echo "Note: Hidden files will be excluded, filenames will be sanitized"
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) audio2text/audio_to_text.py --batch --input-dir "$(INPUT_DIR)" --output-dir "$(OUTPUT_DIR)"

filename-test: ## Test filename slugification
	@echo "Testing filename slugification..."
	@$(PYTHON) -c "\
from audio2text.audio_to_text import slugify_filename; \
test_names = [ \
    'ChatGPT Operator Built a \$$500⧸Day Business in 30 Minutes (tutorial)', \
    'Episode #123: How to Make \$$1000⧸Month Online! (AMAZING)', \
    'The Best Marketing Tips & Tricks [2024] - MUST WATCH!!!', \
    'File with spaces and (parentheses) & symbols', \
    '._hidden_file_with_weird_chars' \
]; \
print('Filename Slugification Test:'); \
print('=' * 60); \
[print(f'Original: {name}\\nSafe:     {slugify_filename(name)}.txt\\n{\"=\"*40}') for name in test_names] \
"

audio-help: ## Show all audio-related commands
	@echo "Audio Transcription Commands"
	@echo "============================="
	@echo ""
	@echo "Basic Commands:"
	@echo "  make audio-transcribe INPUT=file.mp3 OUTPUT=transcript.txt"
	@echo "  make audio-batch INPUT_DIR=./mp3s OUTPUT_DIR=./texts"
	@echo ""
	@echo "Safe Commands (handles problematic filenames):"
	@echo "  make audio-batch-safe INPUT_DIR=./mp3s OUTPUT_DIR=./texts"
	@echo ""
	@echo "Text Formatting:"
	@echo "  make format-text INPUT=transcript.txt OUTPUT=formatted.txt"
	@echo "  make format-batch INPUT_DIR=./texts OUTPUT_DIR=./formatted"
	@echo ""
	@echo "Testing & Info:"
	@echo "  make audio-setup     # Verify setup"
	@echo "  make audio-info      # Show configuration"
	@echo "  make filename-test   # Test filename sanitization"
	@echo ""
	@echo "Recommended for 250 podcasts with problematic filenames:"
	@echo "  make audio-batch-safe INPUT_DIR=\"/path/to/podcasts\" OUTPUT_DIR=./transcripts"
	@echo ""
	@echo "Note: For single files with special characters, create a directory"
	@echo "      with just that file and use batch mode instead."

# Docker and RunPod Deployment Targets
# ====================================

docker-build: ## Build Docker image for RunPod deployment
	@echo "Building Docker image: $(DOCKER_FULL_NAME)"
	@echo "========================================"
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "❌ Docker not found. Please install Docker first."; \
		exit 1; \
	fi
	@if ! docker info >/dev/null 2>&1; then \
		echo "❌ Docker is not running. Please start Docker."; \
		exit 1; \
	fi
	@echo "✓ Docker is running"
	@echo "Building image..."
	docker build -t $(DOCKER_FULL_NAME) .
	@echo "✅ Docker image built successfully: $(DOCKER_FULL_NAME)"

docker-push: docker-build ## Push Docker image to Docker Hub
	@echo "Pushing Docker image: $(DOCKER_FULL_NAME)"
	@echo "========================================="
	@if ! docker info | grep -q "Username"; then \
		echo "🔑 Logging into Docker Hub..."; \
		docker login; \
	fi
	@echo "📤 Pushing image..."
	docker push $(DOCKER_FULL_NAME)
	@echo "✅ Image pushed successfully to Docker Hub"

docker-deploy: docker-push ## Build and push Docker image (shortcut)
	@echo "🎉 Docker deployment complete!"
	@echo ""
	@echo "📋 Next steps:"
	@echo "1. Create RunPod instance: make runpod-create"
	@echo "2. Get connection URL: make runpod-info"
	@echo "3. Start transcribing: make runpod-transcribe INPUT=audio.mp3"
	@echo ""
	@echo "🐳 Your image: $(DOCKER_FULL_NAME)"
	@echo "🌐 Docker Hub: https://hub.docker.com/r/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME)"

runpod-check: ## Check RunPod CLI installation
	@echo "Checking RunPod CLI installation..."
	@if ! command -v runpodctl >/dev/null 2>&1; then \
		echo "❌ runpodctl not found. Installing..."; \
		pip install runpodctl; \
	fi
	@echo "✓ runpodctl installed"
	@if [ -z "$$RUNPOD_API_KEY" ]; then \
		echo "⚠️  RUNPOD_API_KEY environment variable not set"; \
		echo "Please run: export RUNPOD_API_KEY=your-api-key"; \
		echo "Get your API key from: https://www.runpod.io/console/user/settings"; \
		exit 1; \
	fi
	@echo "✓ RUNPOD_API_KEY configured"

runpod-create: runpod-check ## Create and deploy RunPod instance
	@echo "Creating RunPod instance..."
	@echo "==========================="
	@echo "📦 Image: $(DOCKER_FULL_NAME)"
	@echo "🔧 GPU: $(RUNPOD_GPU_TYPE)"
	@echo "🏷️  Name: $(RUNPOD_POD_NAME)"
	@echo ""
	@read -p "Continue? (y/N): " confirm; \
	if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
		echo "Cancelled."; \
		exit 1; \
	fi
	@echo "🚀 Creating pod..."
	@pod_output=$$(runpodctl create pod \
		--name "$(RUNPOD_POD_NAME)" \
		--imageName "$(DOCKER_FULL_NAME)" \
		--gpuType "$(RUNPOD_GPU_TYPE)" \
		--gpuCount 1 \
		--volumeSize 20 \
		--containerDiskSize 10 \
		--ports "8080/http" \
		--env "TRANSCRIBE_API_KEY=$(TRANSCRIBE_API_KEY_ENV)" 2>&1); \
	if echo "$$pod_output" | grep -q "error\|Error\|ERROR"; then \
		echo "❌ Failed to create pod:"; \
		echo "$$pod_output"; \
		exit 1; \
	fi; \
	pod_id=$$(echo "$$pod_output" | grep -oE '"[a-z0-9-]+"' | head -1 | tr -d '"'); \
	if [ -n "$$pod_id" ]; then \
		echo "$$pod_id" > .runpod_pod_id; \
		echo "✅ Pod created successfully!"; \
		echo "🆔 Pod ID: $$pod_id"; \
		echo "💾 Pod ID saved to .runpod_pod_id"; \
		echo ""; \
		echo "⏳ Pod is starting up (this may take 2-3 minutes)..."; \
		echo "🔍 Check status: make runpod-info"; \
	else \
		echo "❌ Could not extract pod ID from output:"; \
		echo "$$pod_output"; \
	fi

runpod-info: ## Get RunPod instance status and connection URL
	@echo "RunPod Instance Information"
	@echo "=========================="
	@if [ ! -f ".runpod_pod_id" ]; then \
		echo "❌ No pod ID found. Run 'make runpod-create' first."; \
		exit 1; \
	fi
	@pod_id=$$(cat .runpod_pod_id); \
	echo "🆔 Pod ID: $$pod_id"; \
	echo ""; \
	echo "📊 Status:"; \
	pod_info=$$(runpodctl get pod $$pod_id 2>/dev/null); \
	if [ $$? -eq 0 ]; then \
		echo "$$pod_info" | grep -E "(Status|Runtime|GPU|Container)"; \
		echo ""; \
		if echo "$$pod_info" | grep -q "RUNNING"; then \
			echo "✅ Pod is running!"; \
			echo ""; \
			echo "🌐 Getting connection URL..."; \
			url_info=$$(runpodctl get pod $$pod_id | grep -E "Connect With|Mapped Port" || echo ""); \
			if [ -n "$$url_info" ]; then \
				echo "$$url_info"; \
			fi; \
			echo ""; \
			echo "🔗 Typical URL format: https://$$pod_id-8080.proxy.runpod.net"; \
			echo "🏥 Health check: curl https://$$pod_id-8080.proxy.runpod.net/health"; \
			echo ""; \
			echo "🎵 Ready for transcription!"; \
			echo "💡 Usage: make runpod-transcribe INPUT=audio.mp3"; \
		else \
			echo "⏳ Pod is not yet running. Status check:"; \
			echo "$$pod_info" | grep "Status"; \
		fi; \
	else \
		echo "❌ Could not get pod information. Pod may not exist."; \
	fi

runpod-url: ## Get just the RunPod connection URL
	@if [ ! -f ".runpod_pod_id" ]; then \
		echo "❌ No pod ID found. Run 'make runpod-create' first."; \
		exit 1; \
	fi
	@pod_id=$$(cat .runpod_pod_id); \
	echo "https://$$pod_id-8080.proxy.runpod.net"

runpod-test: ## Test RunPod instance health
	@echo "Testing RunPod instance..."
	@if [ ! -f ".runpod_pod_id" ]; then \
		echo "❌ No pod ID found. Run 'make runpod-create' first."; \
		exit 1; \
	fi
	@pod_id=$$(cat .runpod_pod_id); \
	url="https://$$pod_id-8080.proxy.runpod.net"; \
	echo "🔗 Testing: $$url/health"; \
	response=$$(curl -s -w "%{http_code}" "$$url/health" || echo "000"); \
	if echo "$$response" | grep -q "200"; then \
		echo "✅ Pod is healthy and ready!"; \
		echo "🌐 Server URL: $$url"; \
	else \
		echo "❌ Pod not responding (HTTP: $$response)"; \
		echo "💡 Pod might still be starting up. Try: make runpod-info"; \
	fi

runpod-transcribe: ## Transcribe audio using RunPod instance (INPUT=file.mp3 [OUTPUT=./transcripts])
ifndef INPUT
	@echo "Error: INPUT parameter required"
	@echo "Usage: make runpod-transcribe INPUT=audio.mp3 [OUTPUT=./transcripts]"
	@exit 1
endif
	@echo "Transcribing with RunPod..."
	@echo "=========================="
	@if [ ! -f ".runpod_pod_id" ]; then \
		echo "❌ No pod ID found. Run 'make runpod-create' first."; \
		exit 1; \
	fi
	@if [ ! -f "$(INPUT)" ]; then \
		echo "❌ Input file not found: $(INPUT)"; \
		exit 1; \
	fi
	@pod_id=$$(cat .runpod_pod_id); \
	url="https://$$pod_id-8080.proxy.runpod.net"; \
	output_dir="$(if $(OUTPUT),$(OUTPUT),./transcripts)"; \
	api_key="$(TRANSCRIBE_API_KEY_ENV)"; \
	echo "🎵 Input: $(INPUT)"; \
	echo "📁 Output: $$output_dir"; \
	echo "🌐 Server: $$url"; \
	echo ""; \
	export RUNPOD_SERVER_URL="$$url"; \
	export TRANSCRIBE_API_KEY="$$api_key"; \
	$(PYTHON) transcribe_client.py "$(INPUT)" "$$output_dir"

runpod-batch: ## Batch transcribe directory using RunPod (INPUT_DIR=... [OUTPUT_DIR=./transcripts])
ifndef INPUT_DIR
	@echo "Error: INPUT_DIR parameter required"
	@echo "Usage: make runpod-batch INPUT_DIR=/path/to/mp3s [OUTPUT_DIR=./transcripts]"
	@exit 1
endif
	@echo "Batch transcribing with RunPod..."
	@echo "================================"
	@if [ ! -f ".runpod_pod_id" ]; then \
		echo "❌ No pod ID found. Run 'make runpod-create' first."; \
		exit 1; \
	fi
	@if [ ! -d "$(INPUT_DIR)" ]; then \
		echo "❌ Input directory not found: $(INPUT_DIR)"; \
		exit 1; \
	fi
	@pod_id=$$(cat .runpod_pod_id); \
	url="https://$$pod_id-8080.proxy.runpod.net"; \
	output_dir="$(if $(OUTPUT_DIR),$(OUTPUT_DIR),./transcripts)"; \
	api_key="$(TRANSCRIBE_API_KEY_ENV)"; \
	echo "📁 Input: $(INPUT_DIR)"; \
	echo "📁 Output: $$output_dir"; \
	echo "🌐 Server: $$url"; \
	echo ""; \
	export RUNPOD_SERVER_URL="$$url"; \
	export TRANSCRIBE_API_KEY="$$api_key"; \
	$(PYTHON) transcribe_client.py "$(INPUT_DIR)" "$$output_dir"

runpod-logs: ## Show RunPod instance logs
	@echo "RunPod Instance Logs"
	@echo "==================="
	@if [ ! -f ".runpod_pod_id" ]; then \
		echo "❌ No pod ID found. Run 'make runpod-create' first."; \
		exit 1; \
	fi
	@pod_id=$$(cat .runpod_pod_id); \
	echo "🆔 Pod ID: $$pod_id"; \
	echo ""; \
	runpodctl logs $$pod_id

runpod-stop: ## Stop RunPod instance
	@echo "Stopping RunPod instance..."
	@if [ ! -f ".runpod_pod_id" ]; then \
		echo "❌ No pod ID found. Run 'make runpod-create' first."; \
		exit 1; \
	fi
	@pod_id=$$(cat .runpod_pod_id); \
	echo "🆔 Pod ID: $$pod_id"; \
	echo "🛑 Stopping pod..."; \
	runpodctl stop pod $$pod_id; \
	echo "✅ Pod stopped successfully"

runpod-start: ## Start RunPod instance
	@echo "Starting RunPod instance..."
	@if [ ! -f ".runpod_pod_id" ]; then \
		echo "❌ No pod ID found. Run 'make runpod-create' first."; \
		exit 1; \
	fi
	@pod_id=$$(cat .runpod_pod_id); \
	echo "🆔 Pod ID: $$pod_id"; \
	echo "🚀 Starting pod..."; \
	runpodctl start pod $$pod_id; \
	echo "✅ Pod started successfully"; \
	echo "⏳ Give it 1-2 minutes to become ready"; \
	echo "🔍 Check status: make runpod-info"

runpod-delete: ## Delete RunPod instance
	@echo "Deleting RunPod instance..."
	@if [ ! -f ".runpod_pod_id" ]; then \
		echo "❌ No pod ID found. Nothing to delete."; \
		exit 1; \
	fi
	@pod_id=$$(cat .runpod_pod_id); \
	echo "🆔 Pod ID: $$pod_id"; \
	echo "⚠️  This will permanently delete the pod and all its data."; \
	read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "🗑️  Deleting pod..."; \
		runpodctl remove pod $$pod_id; \
		rm -f .runpod_pod_id; \
		echo "✅ Pod deleted successfully"; \
	else \
		echo "Cancelled."; \
	fi

runpod-status: ## Quick status check of RunPod instance
	@if [ ! -f ".runpod_pod_id" ]; then \
		echo "❌ No pod found. Run 'make runpod-create' to deploy."; \
	else \
		pod_id=$$(cat .runpod_pod_id); \
		echo "🆔 Pod: $$pod_id"; \
		status=$$(runpodctl get pod $$pod_id 2>/dev/null | grep "Status" || echo "Status: Unknown"); \
		echo "📊 $$status"; \
		if echo "$$status" | grep -q "RUNNING"; then \
			url="https://$$pod_id-8080.proxy.runpod.net"; \
			echo "🌐 URL: $$url"; \
		fi; \
	fi

runpod-help: ## Show all RunPod-related commands
	@echo "RunPod Management Commands"
	@echo "=========================="
	@echo ""
	@echo "Deployment:"
	@echo "  make docker-deploy DOCKER_USERNAME=myuser  # Build and push Docker image"
	@echo "  make runpod-create                          # Create RunPod instance"
	@echo ""
	@echo "Usage:"
	@echo "  make runpod-transcribe INPUT=audio.mp3      # Transcribe single file"
	@echo "  make runpod-batch INPUT_DIR=./mp3s          # Transcribe directory"
	@echo ""
	@echo "Management:"
	@echo "  make runpod-info      # Get status and connection URL"
	@echo "  make runpod-status    # Quick status check"
	@echo "  make runpod-test      # Test health endpoint"
	@echo "  make runpod-logs      # View container logs"
	@echo ""
	@echo "Control:"
	@echo "  make runpod-stop      # Stop instance (saves money)"
	@echo "  make runpod-start     # Start stopped instance"
	@echo "  make runpod-delete    # Delete instance permanently"
	@echo ""
	@echo "💰 Cost Management:"
	@echo "  - Pods auto-stop when idle (~5-10 minutes)"
	@echo "  - Use 'make runpod-stop' when done to save costs"
	@echo "  - RTX A5000: ~$$0.50/hour (~$$2.50 for 30min of audio)"