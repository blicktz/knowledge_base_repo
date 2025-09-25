"""
Universal LLM Logging Callback Handler

This module provides comprehensive logging for all LLM interactions in the persona agent,
including individual tool calls and the final synthesis step. Each LLM call gets its own
timestamped subfolder with complete prompt, response, and extracted answer data.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from ..utils.logging import get_logger, get_component_logger


class UniversalLLMLoggingCallbackHandler(BaseCallbackHandler):
    """
    Comprehensive callback handler that logs every LLM interaction in the agent pipeline.
    
    Creates timestamped subfolders for each LLM call with complete logging data:
    - Complete filled prompt
    - Raw LLM response  
    - Extracted final answer
    - Metadata and context
    """
    
    def __init__(self, persona_id: str, base_cache_dir: str):
        """
        Initialize the universal LLM logger.
        
        Args:
            persona_id: The persona identifier
            base_cache_dir: Base directory for caching logs
        """
        self.persona_id = persona_id
        self.base_cache_dir = Path(base_cache_dir)
        self.logger = get_component_logger("LLM-Universal", persona_id)
        
        # Track current LLM call context
        self.current_call_context = None
        self.call_start_time = None
        self.current_user_query = None
        
        # Track tool execution timeline for better context detection
        self.tool_execution_stack = []  # Stack of currently executing tools
        self.recent_tool_activity = []  # Recent tool activity for context
        
        # Create base directory
        self.base_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"UniversalLLMLoggingCallbackHandler initialized for persona: {persona_id}")
    
    def set_user_query(self, query: str):
        """Set the current user query for context in logging."""
        self.current_user_query = query
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str],
        **kwargs: Any
    ) -> Any:
        """Called when LLM starts running."""
        self.call_start_time = datetime.now()
        
        # Try to determine the call context from various sources
        call_context = self._determine_call_context(kwargs, serialized)
        
        # Create timestamped folder name
        timestamp = self.call_start_time.isoformat().replace(':', '-')
        folder_name = f"{call_context}_{timestamp}"
        
        # Create call directory
        self.current_call_dir = self.base_cache_dir / folder_name
        self.current_call_dir.mkdir(parents=True, exist_ok=True)
        
        # Store context for later use
        self.current_call_context = {
            "call_type": "synthesis" if call_context == "synthesis" else "tool_call",
            "tool_name": call_context,
            "timestamp": self.call_start_time.isoformat(),
            "folder_name": folder_name,
            "prompts": prompts,
            "model_info": serialized
        }
        
        # Save prompt data
        self._save_prompt_data(prompts)
        
        self.logger.info(f"Started LLM call: {call_context} -> {folder_name}")
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Called when LLM ends running."""
        if not self.current_call_context:
            return
            
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - self.call_start_time).total_seconds()
        
        # Extract response data
        raw_response = ""
        final_answer = ""
        
        if response.generations and len(response.generations) > 0:
            if len(response.generations[0]) > 0:
                raw_response = response.generations[0][0].text
                final_answer = self._extract_final_answer(raw_response)
        
        # Save response data
        self._save_response_data(raw_response, final_answer)
        
        # Create comprehensive log entry with safe serialization
        log_entry = {
            "timestamp": self.current_call_context["timestamp"],
            "call_type": self.current_call_context["call_type"],
            "tool_name": self.current_call_context["tool_name"],
            "persona_id": self.persona_id,
            "user_query": self.current_user_query or "",
            "processing_time_seconds": processing_time,
            "model_info": self._safe_serialize(self.current_call_context["model_info"]),
            "llm_usage": self._safe_serialize(response.llm_output) if response.llm_output else {},
            "response_metadata": {
                "raw_response_length": len(raw_response),
                "final_answer_length": len(final_answer),
                "generation_count": len(response.generations) if response.generations else 0
            }
        }
        
        # Save comprehensive log
        self._save_log_entry(log_entry)
        
        self.logger.info(
            f"Completed LLM call: {self.current_call_context['tool_name']} "
            f"(time: {processing_time:.2f}s, response: {len(raw_response)} chars)"
        )
        
        # Reset context
        self.current_call_context = None
        self.call_start_time = None
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Called when LLM encounters an error."""
        if not self.current_call_context:
            return
            
        # Log error information
        error_entry = {
            "timestamp": self.current_call_context["timestamp"],
            "call_type": self.current_call_context["call_type"],
            "tool_name": self.current_call_context["tool_name"],
            "persona_id": self.persona_id,
            "user_query": self.current_user_query or "",
            "error": str(error),
            "error_type": type(error).__name__
        }
        
        # Save error log
        error_file = self.current_call_dir / "error.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_entry, f, indent=2, ensure_ascii=False)
        
        self.logger.error(f"LLM call failed: {self.current_call_context['tool_name']} - {error}")
        
        # Reset context
        self.current_call_context = None
        self.call_start_time = None
    
    def _determine_call_context(self, kwargs: Dict[str, Any], serialized: Dict[str, Any]) -> str:
        """
        Determine the context of the LLM call (which tool or synthesis).
        
        Args:
            kwargs: Additional arguments from the callback
            serialized: Serialized model information
            
        Returns:
            String identifier for the call context
        """
        # Method 1: Check for run_name or parent_run_id patterns
        if 'run_name' in kwargs:
            run_name = str(kwargs['run_name']).lower()
            for tool_name in ['query_analyzer', 'get_persona_data', 'retrieve_mental_models', 
                             'retrieve_core_beliefs', 'retrieve_transcripts']:
                if tool_name in run_name:
                    return tool_name
        
        # Method 2: Check tags for tool identification
        if 'tags' in kwargs and kwargs['tags']:
            tags = kwargs['tags']
            for tag in tags:
                tag_str = str(tag).lower()
                for tool_name in ['query_analyzer', 'get_persona_data', 'retrieve_mental_models', 
                                 'retrieve_core_beliefs', 'retrieve_transcripts']:
                    if tool_name in tag_str:
                        return tool_name
        
        # Method 3: Enhanced call stack analysis for indirect LLM calls
        import inspect
        stack = inspect.stack()
        
        # Check deeper into the call stack (25 frames) to catch indirect calls
        for frame_info in stack[:25]:
            filename = frame_info.filename.lower()
            function_name = frame_info.function.lower()
            
            # Get the code context for more detailed analysis
            code_context = frame_info.code_context[0] if frame_info.code_context else ""
            code_context = code_context.lower().strip()
            
            # Method 3a: Direct tool name detection in filenames/functions
            for tool_name in ['query_analyzer', 'get_persona_data', 'retrieve_mental_models', 
                             'retrieve_core_beliefs', 'retrieve_transcripts']:
                if tool_name in filename or tool_name in function_name:
                    return tool_name
            
            # Method 3b: Detect indirect calls within specific tool contexts
            # HyDE generation patterns (usually within retrieve_transcripts)
            if any(pattern in filename or pattern in function_name or pattern in code_context 
                   for pattern in ['hyde', 'hypothesis', 'generate_hypothesis', 'hypothetical']):
                # Look backwards in stack for the originating tool
                for prev_frame in stack[:25]:
                    prev_filename = prev_frame.filename.lower()
                    prev_function = prev_frame.function.lower()
                    if 'retrieve_transcripts' in prev_filename or 'retrieve_transcripts' in prev_function:
                        return 'retrieve_transcripts'
                    # Could also be in mental models or core beliefs
                    if 'retrieve_mental_models' in prev_filename or 'retrieve_mental_models' in prev_function:
                        return 'retrieve_mental_models'
                    if 'retrieve_core_beliefs' in prev_filename or 'retrieve_core_beliefs' in prev_function:
                        return 'retrieve_core_beliefs'
                # Default to retrieve_transcripts for HyDE calls if we can't determine
                return 'retrieve_transcripts'
            
            # Method 3c: Advanced retrieval pipeline patterns
            if any(pattern in filename or pattern in function_name or pattern in code_context 
                   for pattern in ['advanced_retrieval', 'pipeline.retrieve', 'retrieval_pipeline']):
                # Look for the originating retrieval tool
                for prev_frame in stack[:25]:
                    prev_filename = prev_frame.filename.lower()
                    prev_function = prev_frame.function.lower()
                    for tool_name in ['retrieve_transcripts', 'retrieve_mental_models', 'retrieve_core_beliefs']:
                        if tool_name in prev_filename or tool_name in prev_function:
                            return tool_name
                # Default to retrieve_transcripts for advanced pipeline calls
                return 'retrieve_transcripts'
            
            # Method 3d: Reranking calls (could be within any retrieval tool)
            if any(pattern in filename or pattern in function_name or pattern in code_context 
                   for pattern in ['rerank', 'cross_encoder', 'mixedbread']):
                # Look for the originating retrieval tool
                for prev_frame in stack[:25]:
                    prev_filename = prev_frame.filename.lower()
                    prev_function = prev_frame.function.lower()
                    for tool_name in ['retrieve_transcripts', 'retrieve_mental_models', 'retrieve_core_beliefs']:
                        if tool_name in prev_filename or tool_name in prev_function:
                            return tool_name
                # These don't usually make LLM calls, but if they do, default to retrieve_transcripts
                return 'retrieve_transcripts'
        
        # Method 4: Check for metadata in invocation_params
        if 'invocation_params' in kwargs:
            params = kwargs['invocation_params']
            if isinstance(params, dict) and 'metadata' in params:
                metadata = params['metadata']
                if isinstance(metadata, dict) and 'tool_name' in metadata:
                    return metadata['tool_name']
        
        # Method 5: Use tool execution context if available
        if self.tool_execution_stack:
            # Use the most recent tool in the execution stack
            current_tool = self.tool_execution_stack[-1]
            self.logger.debug(f"Using tool execution context: {current_tool}")
            return current_tool
        
        if self.recent_tool_activity:
            # Use the most recent tool activity as context
            recent_tool = self.recent_tool_activity[-1]
            self.logger.debug(f"Using recent tool activity context: {recent_tool}")
            return recent_tool
        
        # Method 6: Improved fallback based on call patterns and timing
        if not hasattr(self, '_call_counter'):
            self._call_counter = 0
            self._recent_tool_calls = []
        
        self._call_counter += 1
        
        # Smarter heuristics based on call sequence
        if self._call_counter == 1:
            # First call is almost always query_analyzer
            tool_name = "query_analyzer"
            self._recent_tool_calls.append(tool_name)
            self.recent_tool_activity.append(tool_name)
            return tool_name
        elif self._call_counter == 2:
            # Second call might be query_analyzer retry or first real tool
            # Check timing - if close to first call, likely retry
            if hasattr(self, 'call_start_time') and self.call_start_time:
                # This is likely a different call pattern, could be various tools
                pass
            # Could be any tool, but often retrieve_transcripts (HyDE)
            tool_name = "retrieve_transcripts"  # Most common for LLM calls
            self._recent_tool_calls.append(tool_name)
            self.recent_tool_activity.append(tool_name)
            return tool_name
        elif self._call_counter <= 4:
            # Middle calls are often retrieval tools making indirect LLM calls
            retrieval_tools = ['retrieve_transcripts', 'retrieve_mental_models', 'retrieve_core_beliefs']
            if self._call_counter - 2 < len(retrieval_tools):
                tool_name = retrieval_tools[self._call_counter - 2]
                self._recent_tool_calls.append(tool_name)
                self.recent_tool_activity.append(tool_name)
                return tool_name
        
        # Later calls are likely final synthesis
        return "synthesis"
    
    def _extract_final_answer(self, raw_response: str) -> str:
        """
        Extract the final answer from the raw LLM response.
        
        Args:
            raw_response: Raw response from the LLM
            
        Returns:
            Cleaned final answer
        """
        if not raw_response:
            return ""
        
        # For tool calls, the response might be JSON - try to parse it
        try:
            parsed = json.loads(raw_response)
            if isinstance(parsed, dict):
                # Common fields that might contain the answer
                for key in ['answer', 'result', 'response', 'content', 'core_task', 'rag_query']:
                    if key in parsed:
                        return str(parsed[key])
                # If no specific field, return the JSON as formatted string
                return json.dumps(parsed, indent=2)
        except (json.JSONDecodeError, TypeError):
            pass
        
        # For synthesis/final responses, return the raw text (it's the final answer)
        return raw_response.strip()
    
    def _save_prompt_data(self, prompts: List[str]):
        """Save the prompt data to markdown file."""
        if not self.current_call_dir:
            return
            
        # Create markdown content with all prompts
        markdown_content = f"# LLM Prompt Data\n\n"
        markdown_content += f"**Prompt Count:** {len(prompts)}\n\n"
        markdown_content += f"**Total Length:** {sum(len(p) for p in prompts)} characters\n\n"
        
        for i, prompt in enumerate(prompts):
            if len(prompts) > 1:
                markdown_content += f"## Prompt {i + 1}\n\n"
            else:
                markdown_content += f"## Complete Prompt\n\n"
            
            # Add the prompt content, preserving any existing markdown
            markdown_content += prompt + "\n\n"
        
        prompt_file = self.current_call_dir / "prompt.md"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def _save_response_data(self, raw_response: str, final_answer: str):
        """Save the response data to markdown files."""
        if not self.current_call_dir:
            return
        
        # Save raw response as markdown
        raw_response_content = f"# Raw LLM Response\n\n{raw_response}\n"
        raw_response_file = self.current_call_dir / "raw_response.md"
        with open(raw_response_file, 'w', encoding='utf-8') as f:
            f.write(raw_response_content)
        
        # Save final answer as markdown
        final_answer_content = f"# Final Answer\n\n{final_answer}\n"
        final_answer_file = self.current_call_dir / "final_answer.md" 
        with open(final_answer_file, 'w', encoding='utf-8') as f:
            f.write(final_answer_content)
    
    def _save_log_entry(self, log_entry: Dict[str, Any]):
        """Save the comprehensive log entry."""
        if not self.current_call_dir:
            return
            
        log_file = self.current_call_dir / "llm_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
    
    def _safe_serialize(self, obj) -> Any:
        """
        Safely serialize objects that may contain non-JSON-serializable types.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation
        """
        if obj is None:
            return None
        
        try:
            # Try to serialize directly first
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # Handle non-serializable objects
            if hasattr(obj, '__dict__'):
                # Convert objects with __dict__ to dictionary
                return {
                    k: self._safe_serialize(v) 
                    for k, v in obj.__dict__.items() 
                    if not k.startswith('_')
                }
            elif isinstance(obj, dict):
                # Recursively handle dictionaries
                return {k: self._safe_serialize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                # Handle sequences
                return [self._safe_serialize(item) for item in obj]
            else:
                # Convert to string for other types
                return str(obj)


def create_universal_llm_logger(persona_id: str, base_cache_dir: str) -> UniversalLLMLoggingCallbackHandler:
    """
    Factory function to create a universal LLM logging callback handler.
    
    Args:
        persona_id: The persona identifier
        base_cache_dir: Base directory for caching logs
        
    Returns:
        Configured UniversalLLMLoggingCallbackHandler instance
    """
    return UniversalLLMLoggingCallbackHandler(persona_id, base_cache_dir)