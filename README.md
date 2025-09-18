# PDF to Markdown Converter

A simple Python script to convert text-selectable PDFs to clean Markdown format, optimized for LLM analysis and processing.

## Features

- 🚀 **Simple & Fast**: Convert PDFs to Markdown in seconds
- 🤖 **LLM-Optimized**: Clean output perfect for AI analysis
- 📄 **Flexible**: Convert entire documents or specific pages
- 🖼️ **Image Support**: Extract images when needed
- 🛠️ **Easy Setup**: Poetry dependency management + Makefile commands

## Quick Start

1. **Install dependencies**:
   ```bash
   make install
   ```

2. **Convert a PDF**:
   ```bash
   make convert INPUT=your_document.pdf
   ```

3. **Save to file**:
   ```bash
   make save FILE=your_document.pdf
   ```

## Installation

### Prerequisites
- Python 3.10 or higher
- Poetry (install from [python-poetry.org](https://python-poetry.org/docs/#installation))

### Setup
```bash
# Clone or download this project
cd pdf_2_text

# Install dependencies
make install

# Test installation
make test
```

## Usage

### Using Makefile (Recommended)

```bash
# Convert PDF to stdout
make convert INPUT=document.pdf

# Convert and save to file
make convert INPUT=document.pdf OUTPUT=result.md

# Convert specific pages (0-based indexing)
make convert INPUT=document.pdf PAGES=0,2-4,7

# Convert with image extraction
make convert INPUT=document.pdf IMAGES=true

# Quick convert (print to terminal)
make quick FILE=document.pdf

# Quick save (auto-generates filename)
make save FILE=document.pdf
```

### Direct Python Usage

```bash
# Basic conversion
poetry run python pdf2text/pdf_to_markdown.py document.pdf

# Save to file
poetry run python pdf2text/pdf_to_markdown.py document.pdf -o output.md

# Convert specific pages
poetry run python pdf2text/pdf_to_markdown.py document.pdf -p 0,2-4 -o output.md

# Extract images
poetry run python pdf2text/pdf_to_markdown.py document.pdf -i --image-dir images

# See all options
poetry run python pdf2text/pdf_to_markdown.py --help
```

## Command Reference

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install` | Install dependencies |
| `make convert` | Convert PDF (see examples above) |
| `make quick FILE=x.pdf` | Quick convert to stdout |
| `make save FILE=x.pdf` | Quick save to .md file |
| `make test` | Test the installation |
| `make clean` | Clean up generated files |
| `make example` | Show usage examples |

## Examples

### Convert entire PDF
```bash
make convert INPUT=research_paper.pdf OUTPUT=paper.md
```

### Convert first 3 pages only
```bash
make convert INPUT=book.pdf PAGES=0-2 OUTPUT=chapter1.md
```

### Extract text and images
```bash
make convert INPUT=presentation.pdf IMAGES=true OUTPUT=slides.md
```

## Output Format

The converter produces clean Markdown with:
- ✅ Proper headers (detected by font size)
- ✅ **Bold** and *italic* text formatting
- ✅ Tables and lists
- ✅ Code blocks (monospaced text)
- ✅ Logical reading order
- ✅ GitHub-compatible markdown syntax

Perfect for feeding into LLMs like Claude, GPT, or local models!

## Why This Tool?

- **Built for LLMs**: PyMuPDF4LLM is specifically designed for AI/LLM consumption
- **No Heavy Dependencies**: No PyTorch or complex ML models required
- **Fast & Reliable**: Works great with text-selectable PDFs
- **Developer Friendly**: Poetry + Makefile = smooth workflow

## Troubleshooting

### Common Issues

1. **"pymupdf4llm not installed"**
   ```bash
   make install
   ```

2. **"Poetry not found"**
   - Install Poetry: https://python-poetry.org/docs/#installation

3. **Poor conversion quality**
   - Ensure your PDF has selectable text (not scanned images)
   - Try extracting images: `make convert INPUT=file.pdf IMAGES=true`

### Getting Help

```bash
make help          # Show available commands
make example       # Show usage examples
make env-info      # Show environment details
```

## License

MIT License - feel free to use and modify!