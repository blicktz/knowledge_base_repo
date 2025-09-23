# Makefile for Markdown Chapter Splitter
# 
# This Makefile provides convenient targets for running the chapter splitting script
# with different configurations and common use cases.

# Default configuration
BOOKS_DIR = ../content_repo/dk_books
OUTPUT_DIR = output_chapters
SCRIPT = split_markdown_chapters.py

# Chunk size configuration
TARGET_WORDS = 2000
MIN_WORDS = 1000
MAX_WORDS = 3000

# Python interpreter
PYTHON = python3

# Default target
.PHONY: help
help:
	@echo "Markdown Chapter Splitter - Available targets:"
	@echo ""
	@echo "  split-all       - Split all books in the default directory"
	@echo "  split-verbose   - Split all books with verbose output"
	@echo "  split-preview   - Preview what would be split (no files created)"
	@echo "  split-small     - Split with smaller chunks (1500 words target)"
	@echo "  split-large     - Split with larger chunks (2500 words target)"
	@echo "  clean           - Remove output directory"
	@echo "  test-single     - Test on a single book (Ultimate Marketing Plan)"
	@echo "  stats           - Show statistics of existing output"
	@echo "  help            - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  BOOKS_DIR       = $(BOOKS_DIR)"
	@echo "  OUTPUT_DIR      = $(OUTPUT_DIR)"
	@echo "  TARGET_WORDS    = $(TARGET_WORDS)"
	@echo ""
	@echo "Examples:"
	@echo "  make split-all"
	@echo "  make split-preview"
	@echo "  make BOOKS_DIR=/path/to/other/books split-all"
	@echo "  make TARGET_WORDS=1500 split-all"

# Split all books with default settings
.PHONY: split-all
split-all:
	@echo "Splitting all books in $(BOOKS_DIR)..."
	$(PYTHON) $(SCRIPT) $(BOOKS_DIR) -o $(OUTPUT_DIR) \
		--target-words $(TARGET_WORDS) \
		--min-words $(MIN_WORDS) \
		--max-words $(MAX_WORDS)
	@echo "✅ Complete! Check $(OUTPUT_DIR)/ for results"

# Split with verbose output for debugging
.PHONY: split-verbose
split-verbose:
	@echo "Splitting all books with verbose output..."
	$(PYTHON) $(SCRIPT) $(BOOKS_DIR) -o $(OUTPUT_DIR) \
		--target-words $(TARGET_WORDS) \
		--min-words $(MIN_WORDS) \
		--max-words $(MAX_WORDS) \
		--verbose

# Preview mode - show what would be done without creating files
.PHONY: split-preview
split-preview:
	@echo "Preview mode - showing what would be split..."
	$(PYTHON) $(SCRIPT) $(BOOKS_DIR) -o $(OUTPUT_DIR) \
		--target-words $(TARGET_WORDS) \
		--min-words $(MIN_WORDS) \
		--max-words $(MAX_WORDS) \
		--preview

# Split with smaller chunks (better for detailed analysis)
.PHONY: split-small
split-small:
	@echo "Splitting with smaller chunks (1500 words target)..."
	$(PYTHON) $(SCRIPT) $(BOOKS_DIR) -o $(OUTPUT_DIR) \
		--target-words 1500 \
		--min-words 800 \
		--max-words 2200 \
		--verbose

# Split with larger chunks (better for context preservation)
.PHONY: split-large
split-large:
	@echo "Splitting with larger chunks (2500 words target)..."
	$(PYTHON) $(SCRIPT) $(BOOKS_DIR) -o $(OUTPUT_DIR) \
		--target-words 2500 \
		--min-words 1500 \
		--max-words 3500 \
		--verbose

# Test on a single book
.PHONY: test-single
test-single:
	@echo "Testing on The Ultimate Marketing Plan..."
	$(PYTHON) $(SCRIPT) $(BOOKS_DIR)/The_Ultimate_Marketing_Plan_-_Dan_S_Kennedy.md \
		-o test_output --verbose

# Clean output directory
.PHONY: clean
clean:
	@echo "Cleaning output directories..."
	rm -rf $(OUTPUT_DIR)
	rm -rf test_output
	@echo "✅ Cleaned!"

# Show statistics of existing output
.PHONY: stats
stats:
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		echo "Output Statistics:"; \
		echo "=================="; \
		echo "Total chunks: $$(ls -1 $(OUTPUT_DIR)/*.md 2>/dev/null | wc -l)"; \
		echo ""; \
		echo "Chunks per book:"; \
		ls -1 $(OUTPUT_DIR)/*.md 2>/dev/null | sed 's/.*\///' | sed 's/_chunk_.*//g' | sort | uniq -c | while read count book; do \
			echo "  $$book: $$count chunks"; \
		done; \
	else \
		echo "No output directory found. Run 'make split-all' first."; \
	fi

# Development targets
.PHONY: check-deps
check-deps:
	@echo "Checking dependencies..."
	@$(PYTHON) --version
	@$(PYTHON) -c "import re, os, argparse; print('✅ Required modules available')"

# Run with custom book directory
split-custom:
	@if [ -z "$(CUSTOM_DIR)" ]; then \
		echo "Error: Please specify CUSTOM_DIR"; \
		echo "Usage: make split-custom CUSTOM_DIR=/path/to/books"; \
		exit 1; \
	fi
	@echo "Splitting books in $(CUSTOM_DIR)..."
	$(PYTHON) $(SCRIPT) $(CUSTOM_DIR) -o $(OUTPUT_DIR) \
		--target-words $(TARGET_WORDS) \
		--min-words $(MIN_WORDS) \
		--max-words $(MAX_WORDS) \
		--verbose

# Quality assurance - check for very small or large chunks
.PHONY: qa-check
qa-check:
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		echo "Quality Assurance Check:"; \
		echo "========================"; \
		echo "Checking for problematic chunks..."; \
		echo ""; \
		echo "Very small chunks (<500 words):"; \
		find $(OUTPUT_DIR) -name '*.md' -exec grep -l "word_count: [0-4][0-9][0-9]" {} \; | head -5; \
		echo ""; \
		echo "Very large chunks (>4000 words):"; \
		find $(OUTPUT_DIR) -name '*.md' -exec grep -l "word_count: [4-9][0-9][0-9][0-9]" {} \; | head -5; \
		echo ""; \
		echo "Sample chunk sizes:"; \
		find $(OUTPUT_DIR) -name '*.md' -exec grep "word_count:" {} \; | head -10; \
	else \
		echo "No output directory found. Run 'make split-all' first."; \
	fi

# Interactive mode - prompt for settings
.PHONY: interactive
interactive:
	@echo "Interactive Chapter Splitter"
	@echo "============================"
	@read -p "Books directory [$(BOOKS_DIR)]: " books_dir; \
	books_dir=$${books_dir:-$(BOOKS_DIR)}; \
	read -p "Output directory [$(OUTPUT_DIR)]: " output_dir; \
	output_dir=$${output_dir:-$(OUTPUT_DIR)}; \
	read -p "Target words per chunk [$(TARGET_WORDS)]: " target; \
	target=$${target:-$(TARGET_WORDS)}; \
	echo ""; \
	echo "Running with:"; \
	echo "  Books: $$books_dir"; \
	echo "  Output: $$output_dir"; \
	echo "  Target: $$target words"; \
	echo ""; \
	$(PYTHON) $(SCRIPT) "$$books_dir" -o "$$output_dir" --target-words "$$target" --verbose

# Show example usage
.PHONY: examples
examples:
	@echo "Example Usage:"
	@echo "=============="
	@echo ""
	@echo "# Basic usage:"
	@echo "make split-all"
	@echo ""
	@echo "# Preview before running:"
	@echo "make split-preview"
	@echo ""
	@echo "# Use different book directory:"
	@echo "make BOOKS_DIR=/path/to/other/books split-all"
	@echo ""
	@echo "# Create smaller chunks for detailed analysis:"
	@echo "make split-small"
	@echo ""
	@echo "# Create larger chunks for more context:"
	@echo "make split-large"
	@echo ""
	@echo "# Test on a single book:"
	@echo "make test-single"
	@echo ""
	@echo "# Check output statistics:"
	@echo "make stats"
	@echo ""
	@echo "# Clean up:"
	@echo "make clean"
	@echo ""
	@echo "# Interactive mode:"
	@echo "make interactive"