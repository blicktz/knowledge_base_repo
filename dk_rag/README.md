# DK AI Copywriting Assistant

A Retrieval-Augmented Generation (RAG) system that generates high-persuasion email copy using DK's proven direct response principles.

## Features

- **RAG-powered Context Retrieval**: Uses DK's book content as dynamic knowledge base
- **Smart Query Formulation**: Automatically selects relevant context based on your task
- **OpenRouter Integration**: Uses Google Gemini 2.0 Flash for high-quality generation
- **Local Vector Database**: Fast, local similarity search with scikit-learn
- **Interactive & Batch Modes**: CLI and interactive usage options

## Quick Setup

1. **Install dependencies** (from project root):
   ```bash
   poetry install
   ```

2. **Setup API keys**:
   ```bash
   make dk-setup
   ```
   This will create a `.env` file. Edit it with your API keys:
   - `OPENROUTER_API_KEY`: Get from [OpenRouter](https://openrouter.ai/keys)
   - `OPENAI_API_KEY`: Get from [OpenAI](https://platform.openai.com/api-keys)

3. **Initialize knowledge base**:
   ```bash
   make dk-setup
   ```
   (Run again after setting API keys)

## Usage

### Generate Copy (One-off)
```bash
make dk-generate TASK="Write an email for a webinar about marketing for developers"
```

### Interactive Mode
```bash
make dk-interactive
```

### Direct Python Usage
```bash
poetry run python dk_rag/generate_copy.py "Your copywriting task here"
```

### Other Commands
```bash
make dk-rebuild     # Rebuild knowledge base
make dk-test        # Test with sample task
```

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Task     │───▶│   Query Gen     │───▶│  Context Search │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Generated Copy │◀───│   OpenRouter    │◀───│ Prompt Builder  │
└─────────────────┘    │ (Gemini 2.0)    │    └─────────────────┘
                       └─────────────────┘
```

## Configuration

Edit `openrouter_config.yaml` to customize:
- LLM model (default: `google/gemini-2.0-flash-001`)
- Chunk size and overlap for document processing
- Similarity thresholds and max results
- API endpoints and headers

## Files Structure

```
dk_rag/
├── generate_copy.py      # Main CLI script
├── rag_system.py         # RAG implementation
├── prompts.py           # DK-style templates
├── openrouter_config.yaml # Configuration
└── db/                  # Vector database (auto-created)
```

## Troubleshooting

### No relevant context found
- Check that markdown files exist in MD_OUTPUT directory
- Try rebuilding the knowledge base: `make dk-rebuild`
- Verify chunk size and similarity thresholds in config

### API errors
- Verify API keys are set correctly in `.env`
- Check OpenRouter account has credits
- Ensure model `google/gemini-2.0-flash-001` is available

### Performance issues
- Reduce chunk size in config for faster processing
- Adjust max_results to return fewer context chunks
- Consider using a smaller model for testing