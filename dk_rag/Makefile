# Virtual Influencer Persona Agent - Phase 1 Makefile
# =======================================================

# Configuration Variables
# -----------------------
PYTHON := poetry run python
PYTEST := poetry run pytest
MODULE := dk_rag.cli.persona_builder

# Default paths - customize these for your setup
DOCS_DIR ?= /Users/blickt/Documents/src/pdf_2_text/content_repo/greg_startup
OUTPUT_DIR ?= ./output
CONFIG_FILE ?= ./config/persona_config.yaml
PATTERN ?= *.txt

# Default persona name for extraction
PERSONA_NAME ?= greg_startup
DK_PERSONA_NAME := dan_kennedy

# Search query for testing
SEARCH_QUERY ?= "mental models for productivity"

# Color output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# SETUP TARGETS
# =============================================================================

.PHONY: install
install: ## Install Python dependencies via Poetry
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	@cd .. && poetry install
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

.PHONY: setup-spacy
setup-spacy: ## Download required spaCy language models
	@echo "$(BLUE)Downloading spaCy models...$(NC)"
	$(PYTHON) -m spacy download en_core_web_sm
	@echo "$(GREEN)✓ spaCy models downloaded$(NC)"

.PHONY: setup-env
setup-env: ## Create .env file template
	@echo "$(BLUE)Creating .env template...$(NC)"
	@if [ ! -f ../.env ]; then \
		echo "# OpenRouter API Configuration" > ../.env; \
		echo "OPENROUTER_API_KEY=your_api_key_here" >> ../.env; \
		echo "$(GREEN)✓ Created .env template at ../.env$(NC)"; \
		echo "$(YELLOW)⚠ Please edit ../.env and add your API key$(NC)"; \
	else \
		echo "$(YELLOW)⚠ .env already exists, skipping$(NC)"; \
	fi

.PHONY: setup
setup: install setup-spacy setup-env ## Complete setup (install + spacy + env)
	@echo "$(GREEN)✓ Complete setup finished$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Edit ../.env and add your OPENROUTER_API_KEY"
	@echo "  2. Run 'make build-kb' to build knowledge base"
	@echo "  3. Run 'make extract-persona' to extract a persona"

# =============================================================================
# KNOWLEDGE BASE TARGETS
# =============================================================================

.PHONY: build-kb
build-kb: ## Build knowledge base from documents
	@echo "$(BLUE)Building knowledge base from $(DOCS_DIR)...$(NC)"
	$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" build-kb \
		--documents-dir "$(DOCS_DIR)" \
		--pattern "$(PATTERN)"
	@echo "$(GREEN)✓ Knowledge base built$(NC)"

.PHONY: rebuild-kb
rebuild-kb: ## Clear and rebuild knowledge base from scratch
	@echo "$(YELLOW)⚠ This will delete the existing knowledge base!$(NC)"
	@read -p "Are you sure? [y/N]: " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "$(BLUE)Rebuilding knowledge base from scratch...$(NC)"; \
		$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" build-kb \
			--documents-dir "$(DOCS_DIR)" \
			--pattern "$(PATTERN)" \
			--rebuild; \
		echo "$(GREEN)✓ Knowledge base rebuilt$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled$(NC)"; \
	fi

.PHONY: update-kb
update-kb: ## Update knowledge base with new documents
	@echo "$(BLUE)Updating knowledge base with documents from $(DOCS_DIR)...$(NC)"
	$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" build-kb \
		--documents-dir "$(DOCS_DIR)" \
		--pattern "$(PATTERN)"
	@echo "$(GREEN)✓ Knowledge base updated$(NC)"

.PHONY: analyze-kb
analyze-kb: ## Analyze knowledge base statistics
	@echo "$(BLUE)Analyzing knowledge base...$(NC)"
	$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" analyze

# =============================================================================
# PERSONA EXTRACTION TARGETS
# =============================================================================

.PHONY: extract-persona
extract-persona: ## Extract persona from documents (use PERSONA_NAME=name)
	@echo "$(BLUE)Extracting persona '$(PERSONA_NAME)' from $(DOCS_DIR)...$(NC)"
	$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" --verbose extract-persona \
		--documents-dir "$(DOCS_DIR)" \
		--name "$(PERSONA_NAME)" \
		--pattern "$(PATTERN)"
	@echo "$(GREEN)✓ Persona '$(PERSONA_NAME)' extracted$(NC)"

.PHONY: extract-dk
extract-dk: ## Extract Dan Kennedy persona specifically
	@echo "$(BLUE)Extracting Dan Kennedy persona...$(NC)"
	$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" --verbose extract-persona \
		--documents-dir "$(DOCS_DIR)" \
		--name "$(DK_PERSONA_NAME)" \
		--pattern "$(PATTERN)"
	@echo "$(GREEN)✓ Dan Kennedy persona extracted$(NC)"

.PHONY: list-personas
list-personas: ## List all available personas
	@echo "$(BLUE)Available personas:$(NC)"
	@$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" --verbose list-personas

.PHONY: export-personas
export-personas: ## Export all personas to output directory
	@echo "$(BLUE)Exporting personas to $(OUTPUT_DIR)...$(NC)"
	@mkdir -p "$(OUTPUT_DIR)"
	$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" export \
		--output-dir "$(OUTPUT_DIR)"
	@echo "$(GREEN)✓ Personas exported to $(OUTPUT_DIR)$(NC)"

# =============================================================================
# SEARCH & QUERY TARGETS
# =============================================================================

.PHONY: search
search: ## Search knowledge base (use SEARCH_QUERY="your query")
	@echo "$(BLUE)Searching for: $(SEARCH_QUERY)$(NC)"
	@$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" search "$(SEARCH_QUERY)" \
		--n-results 10

.PHONY: search-mental-models
search-mental-models: ## Search for mental models and frameworks
	@echo "$(BLUE)Searching for mental models...$(NC)"
	@$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" search "mental models frameworks problem-solving" \
		--n-results 15

.PHONY: search-beliefs
search-beliefs: ## Search for core beliefs and principles
	@echo "$(BLUE)Searching for core beliefs...$(NC)"
	@$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" search "core beliefs principles values philosophy" \
		--n-results 15

.PHONY: search-interactive
search-interactive: ## Interactive search session
	@echo "$(BLUE)Starting interactive search...$(NC)"
	@echo "$(YELLOW)Enter your search queries (Ctrl+C to exit):$(NC)"
	@while true; do \
		read -p "Search> " query; \
		if [ -n "$$query" ]; then \
			$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" search "$$query" \
				--n-results 5; \
		fi; \
	done

# =============================================================================
# TESTING & VALIDATION TARGETS
# =============================================================================

.PHONY: test
test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	@cd .. && $(PYTEST) dk_rag/tests/
	@echo "$(GREEN)✓ All tests passed$(NC)"

.PHONY: test-verbose
test-verbose: ## Run tests with verbose output
	@echo "$(BLUE)Running tests (verbose)...$(NC)"
	@cd .. && $(PYTEST) dk_rag/tests/ -v

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@cd .. && $(PYTEST) dk_rag/tests/ --cov=dk_rag --cov-report=term-missing

.PHONY: validate
validate: ## Validate configuration and setup
	@echo "$(BLUE)Validating configuration and setup...$(NC)"
	@$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" validate

.PHONY: check-quality
check-quality: ## Check persona extraction quality
	@echo "$(BLUE)Checking extraction quality for available personas...$(NC)"
	@for persona in $$($(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" list-personas 2>/dev/null | grep "Name:" | cut -d: -f2); do \
		echo "Checking $$persona..."; \
		$(PYTHON) -c "from dk_rag.data.storage.artifacts import ArtifactManager; \
		              from dk_rag.config.settings import Settings; \
		              from dk_rag.utils.validation import validate_extraction_quality; \
		              settings = Settings.from_file('$(CONFIG_FILE)'); \
		              am = ArtifactManager(settings); \
		              persona = am.load_persona_constitution(name='$$persona'.strip()); \
		              scores = validate_extraction_quality(persona); \
		              print(f'Quality scores: {scores}')"; \
	done

# =============================================================================
# UTILITY TARGETS
# =============================================================================

.PHONY: clean
clean: ## Clean cache and temporary files
	@echo "$(BLUE)Cleaning cache and temporary files...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@rm -rf data/storage/cache/* 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(NC)"

.PHONY: clean-all
clean-all: clean ## Clean everything including vector DB (DESTRUCTIVE!)
	@echo "$(RED)⚠ WARNING: This will delete all data including vector DB and personas!$(NC)"
	@read -p "Are you sure? [y/N]: " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "$(BLUE)Removing all data...$(NC)"; \
		rm -rf data/storage/chroma_db/* 2>/dev/null || true; \
		rm -rf data/storage/artifacts/* 2>/dev/null || true; \
		rm -rf logs/* 2>/dev/null || true; \
		echo "$(GREEN)✓ All data removed$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled$(NC)"; \
	fi

.PHONY: backup
backup: ## Backup personas and knowledge base
	@echo "$(BLUE)Creating backup...$(NC)"
	@mkdir -p backups
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	tar -czf "backups/dk_rag_backup_$$timestamp.tar.gz" \
		data/storage/artifacts \
		data/storage/chroma_db \
		config/*.yaml \
		2>/dev/null || true
	@echo "$(GREEN)✓ Backup created in backups/$(NC)"

# =============================================================================
# DEVELOPMENT TARGETS
# =============================================================================

.PHONY: debug-extract
debug-extract: ## Extract persona with debug logging
	@echo "$(BLUE)Extracting persona with debug output...$(NC)"
	$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" --debug --verbose extract-persona \
		--documents-dir "$(DOCS_DIR)" \
		--name "$(PERSONA_NAME)_debug" \
		--pattern "$(PATTERN)"

.PHONY: shell
shell: ## Launch Python shell with modules loaded
	@echo "$(BLUE)Launching Python shell...$(NC)"
	@$(PYTHON) -c "from dk_rag.core import *; \
	               from dk_rag.data.models.persona_constitution import *; \
	               from dk_rag.config.settings import Settings; \
	               settings = Settings.from_file('$(CONFIG_FILE)'); \
	               print('Available: PersonaExtractor, KnowledgeIndexer, StatisticalAnalyzer'); \
	               print('Settings loaded as: settings'); \
	               import code; \
	               code.interact(local=locals())"

.PHONY: format
format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	@cd .. && poetry run black dk_rag/
	@cd .. && poetry run isort dk_rag/
	@echo "$(GREEN)✓ Code formatted$(NC)"

.PHONY: lint
lint: ## Run code linters
	@echo "$(BLUE)Running linters...$(NC)"
	@cd .. && poetry run flake8 dk_rag/ --max-line-length=120 || true
	@cd .. && poetry run mypy dk_rag/ --ignore-missing-imports || true
	@echo "$(GREEN)✓ Linting complete$(NC)"

# =============================================================================
# QUICK START TARGETS
# =============================================================================

.PHONY: quickstart
quickstart: setup build-kb extract-dk ## Run complete quickstart setup
	@echo "$(GREEN)✓ Quickstart complete!$(NC)"
	@echo "$(YELLOW)You can now:$(NC)"
	@echo "  - Search: make search SEARCH_QUERY='your query'"
	@echo "  - List personas: make list-personas"
	@echo "  - Export: make export-personas"

.PHONY: demo
demo: ## Run a demonstration of key features
	@echo "$(BLUE)Running Phase 1 demonstration...$(NC)"
	@echo "\n$(YELLOW)1. Checking configuration...$(NC)"
	@$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" validate
	@echo "\n$(YELLOW)2. Analyzing knowledge base...$(NC)"
	@$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" analyze | head -20
	@echo "\n$(YELLOW)3. Listing personas...$(NC)"
	@$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" list-personas
	@echo "\n$(YELLOW)4. Sample search...$(NC)"
	@$(PYTHON) -m $(MODULE) --config "$(CONFIG_FILE)" search "mental models" --n-results 3
	@echo "\n$(GREEN)✓ Demo complete$(NC)"

# =============================================================================
# HELP TARGET
# =============================================================================

.PHONY: help
help: ## Display this help message
	@echo "$(BLUE)Virtual Influencer Persona Agent - Phase 1 Makefile$(NC)"
	@echo "======================================================"
	@echo ""
	@echo "$(YELLOW)Usage:$(NC)"
	@echo "  make [target] [VARIABLE=value ...]"
	@echo ""
	@echo "$(YELLOW)Common Variables:$(NC)"
	@echo "  DOCS_DIR       - Directory containing documents (default: $(DOCS_DIR))"
	@echo "  PERSONA_NAME   - Name for persona extraction (default: $(PERSONA_NAME))"
	@echo "  SEARCH_QUERY   - Query for search operations (default: $(SEARCH_QUERY))"
	@echo "  PATTERN        - File pattern to match (default: $(PATTERN))"
	@echo "  CONFIG_FILE    - Configuration file path (default: $(CONFIG_FILE))"
	@echo ""
	@echo "$(YELLOW)Available Targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "; \
		    groups["SETUP"] = "$(GREEN)Setup:$(NC)"; \
		    groups["KNOWLEDGE"] = "$(GREEN)Knowledge Base:$(NC)"; \
		    groups["PERSONA"] = "$(GREEN)Persona Extraction:$(NC)"; \
		    groups["SEARCH"] = "$(GREEN)Search & Query:$(NC)"; \
		    groups["TEST"] = "$(GREEN)Testing & Validation:$(NC)"; \
		    groups["UTILITY"] = "$(GREEN)Utilities:$(NC)"; \
		    groups["DEV"] = "$(GREEN)Development:$(NC)"; \
		    groups["QUICK"] = "$(GREEN)Quick Start:$(NC)"}; \
		{if ($$1 ~ /^setup/) group = "SETUP"; \
		 else if ($$1 ~ /^(build|rebuild|update|analyze)-kb/) group = "KNOWLEDGE"; \
		 else if ($$1 ~ /^(extract|list|export)/) group = "PERSONA"; \
		 else if ($$1 ~ /^search/) group = "SEARCH"; \
		 else if ($$1 ~ /^(test|validate|check)/) group = "TEST"; \
		 else if ($$1 ~ /^(clean|backup)/) group = "UTILITY"; \
		 else if ($$1 ~ /^(debug|shell|format|lint)/) group = "DEV"; \
		 else if ($$1 ~ /^(quickstart|demo)/) group = "QUICK"; \
		 else group = "OTHER"; \
		 targets[group] = targets[group] sprintf("  %-20s %s\n", $$1, $$2)} \
		END {for (g in groups) if (targets[g]) printf "\n%s\n%s", groups[g], targets[g]}'
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make quickstart                    # Complete setup and extraction"
	@echo "  make build-kb DOCS_DIR=/my/docs   # Build KB from custom directory"
	@echo "  make extract-persona PERSONA_NAME=gary_vee  # Extract specific persona"
	@echo "  make search SEARCH_QUERY='productivity tips' # Search knowledge base"
	@echo ""
	@echo "$(YELLOW)For more information, see dk_rag/PHASE1_README.md$(NC)"

# Prevent make from printing directory changes
MAKEFLAGS += --no-print-directory