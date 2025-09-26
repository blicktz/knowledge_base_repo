# Aggressive Tool Calling Implementation

## Problem Statement
After the first conversation round, the LLM was skipping tool calls and answering from general knowledge instead of using RAG retrieval. This was particularly problematic for `creative_task` intent type (e.g., "rewrite my email sequence").

### Example from Logs:
```
Round 1 (11:41:05): Cold email question → Tools called: retrieve_core_beliefs, retrieve_transcripts ✓
Round 2 (12:03:04): Email rewriting request → NO tools called ✗ (went straight to final answer)
```

## Root Cause Analysis

1. **Permissive System Prompt**: Original prompt allowed agent to skip tools without clear enforcement
2. **No Intent-Based Enforcement**: All intent types had the same level of tool requirement
3. **No Policy Validation**: No verification that tools were actually used when required

## Solution Implemented

### Part 1: Conditional Tool Enforcement Based on Intent Type

**File**: `persona_agent.py` - `_build_system_prompt()` method

**Implementation**:
```python
intent_type = query_analysis.get('intent_type', '') if query_analysis else ''

if intent_type != 'conversational_exchange':
    tool_enforcement = """
## ⚠️ MANDATORY TOOL USAGE FOR THIS QUERY ⚠️

**CRITICAL: This query requires tool usage.**

You MUST call at least ONE tool before providing your final answer.
DO NOT skip tools. Even if you think you know the answer from earlier in 
the conversation, you MUST retrieve fresh information from the tools.

If you attempt to provide a Final Answer without calling any tools first, 
the system will reject your response as a policy violation.
"""
else:
    tool_enforcement = """
## TOOL USAGE: OPTIONAL FOR THIS QUERY

This query appears to be simple conversational exchange.
You may skip tools ONLY if this is genuinely just small talk.
"""
```

**Result**: System prompt now dynamically enforces tool usage based on intent classification.

### Part 2: Updated Intent-Specific Instructions

**File**: `persona_agent.py` - `_build_system_prompt()` method

**Changes**:

#### instructional_inquiry - [TOOLS MANDATORY]
```
⚠️ REQUIRED: You MUST call tools for this intent type
- FIRST, call retrieve_mental_models - MANDATORY
- THEN, call retrieve_transcripts - MANDATORY
```

#### principled_inquiry - [TOOLS MANDATORY]
```
⚠️ REQUIRED: You MUST call tools for this intent type
- FIRST, call retrieve_core_beliefs - MANDATORY
- THEN, call retrieve_transcripts - MANDATORY
```

#### factual_inquiry - [TOOLS MANDATORY]
```
⚠️ REQUIRED: You MUST call tools for this intent type
- FIRST, call retrieve_transcripts - MANDATORY
```

#### creative_task - [TOOLS MANDATORY - ALL THREE]
```
⚠️ REQUIRED: You MUST call ALL THREE tools for this intent type
- FIRST, call retrieve_mental_models - MANDATORY
- THEN, call retrieve_core_beliefs - MANDATORY
- FINALLY, call retrieve_transcripts - MANDATORY
- DO NOT SKIP - All three tools are required for authentic creative output

Example: "I will FIRST call retrieve_mental_models for email writing frameworks, 
THEN retrieve_core_beliefs for my philosophy on direct response copy, 
FINALLY retrieve_transcripts for proven email examples"
```

#### conversational_exchange - [TOOLS OPTIONAL]
```
This is the ONLY case where you may skip tools
Examples: "hi", "hello", "thanks", "ok", "got it"
If there is ANY substantive question or request, treat it as a different intent type
```

### Part 3: Tool Usage Policy Validation

**File**: `persona_agent.py`

**New Method**: `_check_tool_usage_policy()`
```python
def _check_tool_usage_policy(self, messages_history: List, intent_type: str) -> bool:
    """
    Check if tools were used when required based on intent type.
    Returns True if policy is satisfied, False if violated.
    """
    if intent_type == 'conversational_exchange':
        return True  # No tools required
    
    # For all other intents, check if at least one tool was called
    tools_called = []
    for msg in messages_history:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get('name', '')
                if tool_name in {'retrieve_mental_models', 'retrieve_core_beliefs', 'retrieve_transcripts'}:
                    tools_called.append(tool_name)
    
    if not tools_called:
        self.logger.error(f"TOOL USAGE POLICY VIOLATION: No tools were called for intent_type='{intent_type}'")
        return False
    
    self.logger.info(f"Tool usage policy satisfied: {len(tools_called)} tool(s) called")
    return True
```

**Integration Points**:
1. `process_query()` - Validates after agent execution completes
2. `process_query_structured_stream()` - Validates after streaming completes

## Expected Behavior Changes

### Before Fix:
```
Round 1: "Tell me about cold email" 
  → intent_type: instructional_inquiry
  → Tools called: retrieve_core_beliefs, retrieve_transcripts ✓

Round 2: "Rewrite my email sequence"
  → intent_type: creative_task  
  → Tools called: NONE ✗
  → Generated generic response without persona context
```

### After Fix:
```
Round 1: "Tell me about cold email"
  → intent_type: instructional_inquiry
  → System prompt: "⚠️ MANDATORY TOOL USAGE"
  → Tools called: retrieve_mental_models, retrieve_transcripts ✓

Round 2: "Rewrite my email sequence"  
  → intent_type: creative_task
  → System prompt: "⚠️ MANDATORY TOOL USAGE - ALL THREE TOOLS"
  → Tools called: retrieve_mental_models, retrieve_core_beliefs, retrieve_transcripts ✓
  → Generated persona-authentic response with frameworks, beliefs, and examples
  
Validation Log: "Tool usage policy satisfied: 3 tool(s) called - ['retrieve_mental_models', 'retrieve_core_beliefs', 'retrieve_transcripts']"
```

### Conversational Exchange (Still Allowed to Skip):
```
User: "Thanks!"
  → intent_type: conversational_exchange
  → System prompt: "TOOL USAGE: OPTIONAL"
  → Tools called: NONE (OK)
  → Quick acknowledgment without retrieval
```

## Log Messages to Monitor

### Success Indicators:
```
"Tool usage policy satisfied: 3 tool(s) called for intent_type='creative_task'"
"Tools used: ['retrieve_mental_models', 'retrieve_core_beliefs', 'retrieve_transcripts']"
```

### Policy Violations:
```
"TOOL USAGE POLICY VIOLATION: No tools were called for intent_type='creative_task'"
"Expected at least one tool call for non-conversational queries"
```

## Files Modified

1. **dk_rag/agent/persona_agent.py**
   - `_build_system_prompt()` - Added conditional tool enforcement (lines 175-201)
   - `_build_system_prompt()` - Updated intent instructions (lines 377-410)
   - `_check_tool_usage_policy()` - New validation method (lines 153-178)
   - `process_query()` - Added validation after execution (lines 675-680)
   - `process_query_structured_stream()` - Added validation tracking (lines 851-913)

## Testing Checklist

- [ ] Test Round 1: Initial question with `instructional_inquiry` → Verify tools called
- [ ] Test Round 2: Creative task like "rewrite email" → Verify ALL 3 tools called
- [ ] Test Round 3: Follow-up question → Verify tools still called (not skipped)
- [ ] Test: Simple "thanks" → Verify tools can be skipped (conversational_exchange)
- [ ] Monitor logs for "Tool usage policy satisfied" messages
- [ ] Monitor logs for "TOOL USAGE POLICY VIOLATION" messages (should not appear)

## Key Design Decisions

1. **Intent-Based Enforcement**: Uses existing query analyzer's `intent_type` classification
   - Single source of truth
   - No duplicate logic
   - Easy to maintain

2. **Logging Only (No Hard Failure)**: Policy violations are logged but don't block responses
   - Allows for monitoring in production
   - Can be changed to hard failure if needed
   - Provides visibility into agent behavior

3. **Clear Hierarchy**:
   - `conversational_exchange` = OPTIONAL tools
   - All other intents = MANDATORY tools (at least one)
   - `creative_task` = MANDATORY all three tools

## Production Deployment Notes

1. **Monitor Initial Deployment**: Watch for policy violation logs
2. **Track Intent Classification**: Ensure query analyzer is classifying intents correctly
3. **Adjust Thresholds**: If too aggressive, can adjust intent-specific requirements
4. **Consider Hard Failures**: If violations are common, can change validation to block responses

## Success Metrics

- **Tool Usage Rate**: Should be ~95%+ for non-conversational queries
- **Policy Violations**: Should be near 0% after deployment
- **Response Quality**: Should improve due to consistent RAG usage
- **Persona Authenticity**: Should improve with mandatory tool usage for creative tasks

## Rollback Plan

If issues arise, revert changes to `_build_system_prompt()`:
```python
# Remove conditional tool_enforcement
# Restore original intent instructions without [MANDATORY] tags
# Remove _check_tool_usage_policy() calls
```

## Related Documentation

- `TOOL_CALLING_FIX_SUMMARY.md` - Original tool name concatenation fix
- `COMPILATION_TEST_REPORT.md` - Compilation and validation tests