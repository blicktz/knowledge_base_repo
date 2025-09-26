# Tool Calling Bug Fix Summary

## Problem
The Gemini 2.5 Pro agent was concatenating tool names like `retrieve_transcriptsretrieve_transcriptsretrieve_transcripts` instead of calling individual tools properly. This happened primarily in 2nd/3rd chat rounds but worked fine initially.

## Root Causes (from Web Research)
1. **Gemini API Parallel Function Call Issue**: Gemini models support parallel function calling, which can cause tool name formatting issues in multi-turn conversations
2. **ToolMessage.name Not Set Properly**: LangChain's tool message formatting wasn't consistently setting the `name` field required by Gemini API
3. **Multi-Turn State Corruption**: Agent was attempting to call multiple tools simultaneously when only one was needed

## Solutions Implemented

### Solution 3: Explicit Tool Name Handling (agent_tools.py)
**Changes made:**
- Modified all three tools to return JSON strings with explicit `tool_name` field
- Changed return type from `List[Dict[str, Any]]` to `str` (JSON)
- Each tool response now includes:
  ```json
  {
    "tool_name": "retrieve_transcripts",
    "results": [...],
    "count": 5
  }
  ```
- Added clearer docstrings with usage guidelines
- Improved error handling with explicit tool name in error responses

**Files modified:**
- `/Users/blickt/Documents/src/pdf_2_text/dk_rag/tools/agent_tools.py`

### Solution 4: Improved System Prompt Clarity (persona_agent.py)
**Changes made:**
- Added **CRITICAL TOOL USAGE RULES** section at the top of system prompt with 5 explicit rules:
  1. ONE TOOL AT A TIME - Never call multiple tools simultaneously
  2. WAIT FOR OBSERVATION - Wait for results before next action
  3. EXACT TOOL NAMES - Use exact tool names (with examples of what NOT to do)
  4. PROPER FORMAT - Single, non-repeated tool name strings
  5. NO CONCATENATION - Each tool name appears once per call

- Simplified reasoning process instructions:
  - Changed from complex multi-tool plans to **sequential tool calling**
  - Added "FIRST", "THEN", "FINALLY" keywords to emphasize order
  - Emphasized "ONE AT A TIME" throughout
  - Removed instructions that could be interpreted as "call multiple tools at once"

- Updated intent-specific instructions:
  - `instructional_inquiry`: FIRST call retrieve_mental_models, THEN retrieve_transcripts
  - `principled_inquiry`: FIRST call retrieve_core_beliefs, THEN retrieve_transcripts
  - `factual_inquiry`: FIRST call retrieve_transcripts (most relevant)
  - `creative_task`: Sequential calls in specific order

**Files modified:**
- `/Users/blickt/Documents/src/pdf_2_text/dk_rag/agent/persona_agent.py` (lines 189-328)

### Solution 5: Tool Call Validation (persona_agent.py)
**Changes made:**
- Added `_validate_tool_calls()` method to validate all tool calls before execution
- Validation checks:
  - Tool name must be exactly one of: `retrieve_mental_models`, `retrieve_core_beliefs`, `retrieve_transcripts`
  - Detects concatenated names (e.g., if tool name contains valid name but isn't exact match)
  - Validates required `args` parameter exists
  - Validates required `query` parameter exists
  - Logs detailed error messages for debugging

- Integrated validation into `process_query()`:
  - Validates tool calls during streaming
  - Stops execution if invalid tool call detected
  - Returns graceful error message to user
  - Logs all validation failures for debugging

- Added validation to `process_query_stream()`:
  - Validates tool calls in `on_tool_start` events
  - Stops streaming if invalid tool detected
  - Yields error message to user

**Files modified:**
- `/Users/blickt/Documents/src/pdf_2_text/dk_rag/agent/persona_agent.py` (lines 112-151, 572-603, 685-695)

## Expected Behavior After Fix

### Before Fix:
```
Tool: Error: retrieve_transcriptsretrieve_transcriptsretrieve_transcripts is not a valid tool
Tool: Error: retrieve_transcriptsretrieve_transcriptsretrieve_transcripts is not a valid tool
Tool: Error: -1 is not a valid tool
```

### After Fix:
```
Tool called: retrieve_transcripts
Tool calls validated successfully: ['retrieve_transcripts']
Retrieved 3 transcript chunks
[Returns proper JSON response with tool_name field]
```

## Additional Recommendations

### 1. Update LangChain Dependencies (Not Implemented)
Ensure you're using latest versions:
```bash
poetry add langchain@^0.3.76 langchain-google-genai@^0.2.0
```

### 2. Consider Disabling Streaming (If Issues Persist)
The parallel function call bug only affects streaming mode. If problems continue, set:
```python
# In llm_factory.py or settings
streaming=False
```

### 3. Monitor Logs
Watch for these log messages:
- `"INVALID TOOL CALL DETECTED:"` - Validation caught a bad tool call
- `"Tool calls validated successfully:"` - Tool calls are working correctly
- `"Tool name appears to be concatenated"` - Concatenation detected and blocked

## Testing Recommendations

1. **Test multi-turn conversations** - The bug occurred primarily in 2nd/3rd rounds
2. **Test different intent types** - Verify factual_inquiry, instructional_inquiry, etc.
3. **Check logs** - Monitor for validation messages
4. **Verify tool responses** - Ensure tools return proper JSON format
5. **Test error recovery** - Verify graceful error handling when validation fails

## Files Modified
1. `/Users/blickt/Documents/src/pdf_2_text/dk_rag/tools/agent_tools.py` - Tool return formats and explicit naming
2. `/Users/blickt/Documents/src/pdf_2_text/dk_rag/agent/persona_agent.py` - System prompt + validation logic

## Next Steps
1. Test the agent with multi-turn conversations
2. Monitor logs for validation messages
3. If issues persist, consider updating LangChain dependencies
4. Consider adding telemetry to track tool call patterns over time