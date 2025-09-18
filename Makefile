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