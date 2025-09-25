# Interactive Chain Testing Scripts

This directory contains interactive testing tools for the LangChain Persona Agent system.

## Quick Start

```bash
# From project root
python dk_rag/scripts/test_interactive.py

# Or make it executable and run directly
chmod +x dk_rag/scripts/test_interactive.py
./dk_rag/scripts/test_interactive.py
```

## Main Script: `test_interactive.py`

An interactive menu-driven script that allows you to test each component of the LangChain persona agent system individually.

### Features

- **Step-by-step testing**: Test each component in isolation
- **Interactive prompts**: Enter custom queries and test data
- **Multiple personas**: Switch between available personas
- **Detailed output**: Shows inputs, outputs, timing, and errors
- **Configuration display**: Shows active models and settings
- **Error handling**: Graceful error handling with detailed messages

### Available Tests

1. **Query Analysis Test**
   - Tests the `query_analyzer` tool
   - Input: Custom user query
   - Output: Structured analysis (core_task, rag_query, intent_type, etc.)

2. **Persona Data Retrieval Test**
   - Tests the `get_persona_data` tool
   - Input: Selected persona
   - Output: Linguistic style, communication patterns, metadata

3. **Mental Models Retrieval Test**
   - Tests the `retrieve_mental_models` tool
   - Input: Custom search query
   - Output: Top-k relevant mental models with scores

4. **Core Beliefs Retrieval Test**
   - Tests the `retrieve_core_beliefs` tool
   - Input: Custom search query
   - Output: Top-k relevant core beliefs with scores

5. **Transcript Retrieval Test**
   - Tests the `retrieve_transcripts` tool (Phase 2 pipeline)
   - Input: Custom search query
   - Output: Top-k relevant transcript chunks with metadata

6. **Synthesis Chain Test**
   - Tests the LCEL synthesis chain
   - Input: User query + context
   - Output: Complete synthesized response using mock retrieval data

7. **End-to-End Agent Test**
   - Tests the complete ReAct agent with conversation memory
   - Input: User query
   - Output: Full agent response with tool usage

8. **Configuration Validation**
   - Validates all settings and dependencies
   - Shows detailed configuration information
   - Reports any issues or missing components

### Sample Session

```
🧪 LangChain Persona Agent - Interactive Step Tester
============================================================

📋 Configuration:
   Base Storage: /Volumes/J15/aicallgo_data/persona_data_base
   Query Analysis Model: gemini/gemini-2.0-flash (fast)
   Synthesis Model: gemini/gemini-2.5-pro (heavy)
   Retrieval Config: MM=3, CB=5, T=5

👥 Available Personas: greg_startup, dan_kennedy
   Selected Persona: greg_startup

🧪 Test Options:
   1. Query Analysis Test
   2. Persona Data Retrieval Test
   3. Mental Models Retrieval Test
   4. Core Beliefs Retrieval Test
   5. Transcript Retrieval Test
   6. Synthesis Chain Test
   7. End-to-End Agent Test
   8. Configuration Validation
   9. Select Different Persona
   0. Exit

Enter your choice (0-9): 1

🔍 Query Analysis Test
==============================
Enter your query [Write me a sales email for a new SaaS product]: Help me create a pricing strategy

⏱️  Running query analysis...
✅ Query Analysis Results:
{
  "core_task": "Develop a pricing strategy for a product or service",
  "rag_query": "pricing strategy product pricing models revenue optimization",
  "provided_context": "",
  "intent_type": "task"
}

⏱️  Execution time: 1.34s

Press Enter to continue...
```

### Usage Tips

1. **Start with Configuration Validation** - Run test #8 first to ensure everything is set up correctly

2. **Test Individual Components** - Test each component (1-6) individually to understand their outputs

3. **Try Different Queries** - Use various types of queries to see how the system responds

4. **Check End-to-End** - Run test #7 to see the complete agent in action

5. **Switch Personas** - Use test #9 to switch between different personas and see how responses differ

### Error Handling

The script includes comprehensive error handling:
- Shows detailed error messages
- Logs exceptions for debugging
- Allows you to continue testing after errors
- Graceful handling of missing dependencies or data

### Dependencies

All dependencies are handled automatically through the main project configuration. The script will inform you if any required components are missing.

## Development

To add new test cases:

1. Add a new test method to the `InteractiveChainTester` class
2. Add the option to the `display_menu()` method
3. Add the choice handling to the main loop in `run()`

The script follows the same configuration and logging patterns as the rest of the system.