# Chainlit UI for Virtual Influencer Persona Agent

Simple chat interface for internal testing of the LangChain ReAct persona agent.

## Quick Start

### 1. Run the Application

From the project root:

```bash
cd /Users/blickt/Documents/src/pdf_2_text/dk_rag
chainlit run chainlit/app.py --port 8000
```

Or from this directory:

```bash
chainlit run app.py --port 8000
```

### 2. Access the UI

Open your browser to: **http://localhost:8000**

### 3. Use the Chat Interface

1. Click the **settings icon (⚙️)** in the top right
2. Select your desired persona from the dropdown
3. Start chatting!

## Features

✅ **Persona Selector** - Choose from available influencers via dropdown  
✅ **Conversation Memory** - Chat history maintained within session  
✅ **Easy Persona Switching** - Change personas anytime (starts fresh conversation)  
✅ **Real-time Responses** - See agent thinking and responses in real-time  
✅ **No Database Required** - Simple in-memory session management  

## Architecture

- **Chainlit**: Chat UI framework with built-in LangChain integration
- **LangChain ReAct Agent**: Existing `LangChainPersonaAgent` from `dk_rag.agent.persona_agent`
- **Session Management**: UUID-based sessions, one per browser tab
- **Agent Caching**: Agents cached in memory per persona for performance

## Configuration

The app uses the default configuration from `dk_rag/config/persona_config.yaml`. No additional setup needed.

## Limitations (Current Version)

- ❌ No persistent thread history (refresh = new conversation)
- ❌ No ChatGPT-style sidebar with past conversations
- ❌ No authentication
- ❌ Sessions lost on server restart

## Future Enhancements

To add thread persistence with ChatGPT-style sidebar:
- Option 1: Integrate Literal AI (2-3 hours)
- Option 2: Build custom SQLite data layer (4 hours)

## Troubleshooting

**Port already in use:**
```bash
chainlit run app.py --port 8080  # Use different port
```

**Agent not found:**
- Verify personas exist in storage directory: `{base_storage_dir}/personas/`
- Check logs in `dk_rag/logs/`

**Slow responses:**
- Check LLM model configuration in `persona_config.yaml`
- Verify API keys in `.env` file

## Development

Run with auto-reload for development:
```bash
chainlit run app.py --port 8000 --watch
```

View logs:
```bash
tail -f ../logs/chainlit_*.log
```

## Comparison with Gradio UI

| Feature | Chainlit | Gradio |
|---------|----------|--------|
| Lines of code | ~150 | ~170 |
| LangChain integration | Native | Manual |
| Tool visualization | Yes | No |
| Setup complexity | Simple | Medium |
| Best for | Agents | General ML UIs |

---

*Built for internal testing. Not production-ready.*