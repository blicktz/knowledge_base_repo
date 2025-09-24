# Phase 2 Testing Scripts

## test_phase2_interactive.py

Interactive testing script for Phase 2 advanced retrieval features including HyDE, Hybrid Search, and Cross-Encoder Reranking.

### Prerequisites

1. **Knowledge Base Built**: Run `make build-kb` first to ensure both vector store and BM25 indexes are built
2. **Phase 2 Enabled**: Ensure `retrieval.enabled: true` in your config file
3. **Persona Available**: Have a persona created and configured

### Usage

```bash
# Basic usage
python scripts/test_phase2_interactive.py --persona-id greg_startup

# With custom config file
python scripts/test_phase2_interactive.py --persona-id dan_kennedy --config ./config/persona_config.yaml
```

### Features

**Individual Feature Tests:**
1. **HyDE Retrieval** - Test hypothetical document generation and enhanced search
2. **Hybrid Search** - Compare BM25, Vector, and fused results with score breakdowns
3. **Cross-Encoder Reranking** - Before/after reranking comparison with detailed scores

**Interactive Mode:**
- **Interactive Search** - Real-time search interface using full Phase 2 pipeline
- **Phase 1 vs Phase 2** - Side-by-side comparison with performance metrics

### Sample Session

```
python scripts/test_phase2_interactive.py --persona-id greg_startup

Available Tests:
1. Test HyDE Retrieval
2. Test Hybrid Search (BM25+Vector)
3. Test Cross-Encoder Reranking
4. Interactive Search (Full Pipeline)
5. Compare Phase 1 vs Phase 2
6. Exit

Select option (1-6): 4

Interactive Search - Full Phase 2 Pipeline
Full Pipeline Includes:
• HyDE query expansion
• Hybrid search (BM25 + Vector)
• Cross-encoder reranking

Enter your search queries (type 'quit' to exit)

Search> productivity frameworks

Processing with full Phase 2 pipeline...
✓ Search completed in 1.23s

Phase 2 Pipeline Results
----------------------------------------
1. Productivity frameworks like GTD and PARA METHOD help organize...
   Score: 0.89
   Source: /path/to/document.txt
   Chunk: 15

[Additional results...]

Search> quit
```

### Troubleshooting

**"Persona not found"** - Run `make list-personas` to see available personas

**"Knowledge base is empty"** - Run `make build-kb PERSONA_NAME=your_persona`

**"Phase 2 pipeline not available"** - Check that `retrieval.enabled: true` in config and knowledge base was built after Phase 2 configuration

**Import errors** - Ensure you're running from the dk_rag root directory