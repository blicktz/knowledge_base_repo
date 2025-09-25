# LangChain Persona Agent System

This is a complete rewrite of the Phase 3 implementation using LangChain's native agent framework instead of manual orchestration.

## 🆕 What Changed

### Before (Manual Implementation)
- Custom `PersonaAgent` class with manual tool orchestration
- Direct `litellm.completion()` calls
- Manual synthesis engine with custom prompt building
- No conversation memory between queries
- Complex error handling and fallback logic

### After (LangChain Native)
- **ReAct Agent**: Uses LangGraph's `create_react_agent` with built-in reasoning
- **Native Tools**: Simple `@tool` decorated functions instead of complex inheritance
- **LCEL Chains**: Composable chains for synthesis instead of manual prompt assembly
- **Memory**: Built-in conversation context with `MemorySaver`
- **Streaming**: Real-time tool execution feedback

## 🏗️ Architecture

### Core Components

1. **Tools** (`dk_rag/tools/agent_tools.py`)
   - `@tool` decorated functions for each capability
   - `query_analyzer`: Extract core task and RAG query
   - `get_persona_data`: Load persona characteristics
   - `retrieve_mental_models`: Get relevant frameworks (k=3)
   - `retrieve_core_beliefs`: Get relevant principles (k=5)
   - `retrieve_transcripts`: Get relevant content (k=5)

2. **Agent** (`dk_rag/agent/persona_agent.py`)
   - `LangChainPersonaAgent` using `create_react_agent`
   - Built-in conversation memory with session management
   - Automatic tool selection and orchestration
   - System prompt that defines persona behavior

3. **Chains** (`dk_rag/chains/synthesis_chain.py`)
   - LCEL chains for response synthesis
   - Context aggregation and formatting
   - Composable pipeline architecture

4. **API** (`dk_rag/api/persona_api.py`)
   - FastAPI endpoints for the LangChain agent
   - Conversation history management
   - Session-based memory

5. **Entry Points**
   - `dk_rag/main.py`: Main FastAPI application launcher
   - `dk_rag/test_agent.py`: Comprehensive agent testing suite

## 🚀 Quick Start

### 1. Run the LangChain Agent
```bash
# Navigate to dk_rag directory and run the agent
cd dk_rag
python main.py

# Or from project root:
python -m dk_rag.main
```

### 2. Test the Agent
```bash
# From dk_rag directory
cd dk_rag
python test_agent.py

# Or from project root:
python -m dk_rag.test_agent
```

### 3. API Usage

#### Process a Query
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is your approach to problem solving?",
    "persona_id": "dan_kennedy",
    "session_id": "my_conversation_123"
  }'
```

#### Get Conversation History
```bash
curl "http://localhost:8000/api/v1/conversation/my_conversation_123?persona_id=dan_kennedy"
```

#### Check Health
```bash
curl "http://localhost:8000/api/v1/health"
```

## 🎯 Key Benefits

### 1. **Mature Framework**
- Leverages LangChain's battle-tested agent patterns
- Built-in error handling and retry logic
- Extensive ecosystem and community support

### 2. **Conversation Memory**
```python
# Automatic conversation context
agent.process_query("Hello, I'm John", session_id="user_123")
agent.process_query("Do you remember my name?", session_id="user_123")
# Agent remembers "John" from previous interaction
```

### 3. **Tool Streaming**
```python
# Real-time tool execution feedback
for step in agent_executor.stream({"messages": [message]}, config):
    # See each tool call as it happens
    print(f"Tool called: {step['messages'][-1]}")
```

### 4. **Simplified Tools**
```python
# Before: Complex inheritance
class QueryAnalyzerTool(BasePersonaTool):
    def execute(self, query, metadata):
        # 100+ lines of manual orchestration

# After: Simple function
@tool
def query_analyzer(query: str, persona_id: str) -> Dict[str, Any]:
    """Analyze user queries and extract core tasks"""
    # Direct, focused implementation
```

### 5. **Composable Chains**
```python
# LCEL chain composition
synthesis_chain = (
    context_aggregation
    | synthesis_prompt 
    | llm 
    | output_parser
)
```

## 🔧 Configuration

The system uses the same `persona_config.yaml` but leverages LangChain's LLM wrappers:

```yaml
agent:
  synthesis:
    llm_model: "gemini/gemini-2.5-pro"  # Used via ChatLiteLLM
    temperature: 0.7
    max_tokens: 4000
```

## 🧪 Testing

### Agent Test Suite
```bash
python test_langchain_agent.py
```

This tests:
- ✅ Agent creation and initialization
- ✅ Tool availability and configuration  
- ✅ Query processing with different types
- ✅ Conversation memory and history
- ✅ Session management

### API Test Suite
```bash
# Start the API first
python main_langchain.py

# Then test endpoints
curl -X GET "http://localhost:8000/api/v1/health"
curl -X GET "http://localhost:8000/api/v1/personas"
```

## 📊 Comparison

| Feature | Manual Implementation | LangChain Implementation |
|---------|----------------------|-------------------------|
| **Agent Pattern** | Custom orchestration | ReAct with create_react_agent |
| **Tools** | Complex inheritance | Simple @tool functions |
| **Memory** | None | Built-in MemorySaver |
| **Streaming** | Not supported | Real-time tool execution |
| **Error Handling** | Manual fallbacks | Framework built-ins |
| **Code Lines** | ~2000+ lines | ~1200 lines |
| **Maintainability** | Complex custom logic | Standard LangChain patterns |

## 🔄 Migration from Manual System

The new system has completely replaced the old manual implementation:

1. **Entry point**: `cd dk_rag && python main.py` (self-contained within dk_rag/)
2. **Clean file names**: No more "langchain" prefixes/suffixes
3. **Enhanced capabilities**: Memory, streaming, better error handling
4. **Same API interface**: All endpoints work the same way with additional conversation features

## 🛠️ Development

### Adding New Tools
```python
@tool
def my_new_tool(query: str, persona_id: str) -> Any:
    """Description of what this tool does"""
    # Implementation
    return result

# Add to PERSONA_TOOLS list in agent_tools.py
```

### Custom Chains
```python
from langchain_core.runnables import RunnablePassthrough
from dk_rag.chains.synthesis_chain import create_synthesis_chain

# Create custom chain
my_chain = (
    RunnablePassthrough.assign(custom_context=my_custom_retriever)
    | create_synthesis_chain(persona_id, settings)
)
```

## 🎯 Next Steps

1. **Performance Testing**: Compare response times with manual system
2. **Advanced Features**: Add tool result caching, parallel tool execution
3. **Monitoring**: Add LangSmith integration for observability
4. **Deployment**: Configure for production with proper scaling

## 📁 File Structure

The complete system is now self-contained within `dk_rag/`:

```
dk_rag/
├── main.py                    # 🚀 FastAPI entry point
├── test_agent.py             # 🧪 Agent test suite
├── agent/
│   └── persona_agent.py      # 🤖 LangChain ReAct agent
├── api/
│   └── persona_api.py        # 🌐 FastAPI endpoints
├── tools/
│   └── agent_tools.py        # 🛠️ @tool functions
├── chains/
│   └── synthesis_chain.py    # 🔗 LCEL chains
└── ... (other existing components)
```

This LangChain implementation provides the same functionality as the manual system but with significant improvements in maintainability, extensibility, and built-in capabilities. Everything is now organized cleanly within the `dk_rag` folder with intuitive file names.