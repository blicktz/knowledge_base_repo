# DK RAG - LangChain Persona Agent System

This directory contains the complete LangChain-native persona agent system.

## 🚀 Quick Start

### Run the Agent
```bash
# From this directory
python main.py

# Or from project root  
python -m dk_rag.main
```

### Test the Agent
```bash
# From this directory
python test_agent.py

# Or from project root
python -m dk_rag.test_agent  
```

### API Documentation
Once running, visit: http://localhost:8000/docs

## 🏗️ Architecture

- **`main.py`** - FastAPI application entry point
- **`test_agent.py`** - Comprehensive testing suite
- **`agent/persona_agent.py`** - LangChain ReAct agent with memory
- **`api/persona_api.py`** - FastAPI endpoints with conversation features
- **`tools/agent_tools.py`** - LangChain @tool functions
- **`chains/synthesis_chain.py`** - LCEL chains for response synthesis

## 📚 Full Documentation

See `/README_LANGCHAIN.md` in the project root for complete documentation, examples, and migration details.

---

This system uses LangChain's `create_react_agent` with built-in conversation memory, replacing the previous manual orchestration with a mature, maintainable framework.

## ⚙️ Configuration-Driven

All models and parameters are driven by `config/persona_config.yaml`:

- **Light Tasks**: `gemini/gemini-2.0-flash` (query analysis)
- **Heavy Tasks**: `gemini/gemini-2.5-pro` (response synthesis) 
- **Retrieval Counts**: Mental Models (k=3), Core Beliefs (k=5), Transcripts (k=5)
- **API Settings**: Host, port, CORS, rate limiting all configurable

No hardcoded values - change the YAML to adjust behavior!