.PHONY: help install convert test clean example batch batch-custom setup-dirs
.DEFAULT_GOAL := help

# Variables
PYTHON := poetry run python
SCRIPT := pdf2text/pdf_to_markdown.py
VENV := .venv

# Default directories for batch processing
PDF_INPUT := /Volumes/J15/copy-writing/dk_books_pdf
MD_OUTPUT := ./dk_books

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