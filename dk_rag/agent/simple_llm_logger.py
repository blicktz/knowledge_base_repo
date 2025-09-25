"""
Simple LLM Logging Callback Handler with Explicit Context

This replaces the complex UniversalLLMLoggingCallbackHandler with a much simpler
implementation that uses explicit context instead of trying to guess what each
LLM call is for.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from ..config.settings import Settings
from ..utils.logging import get_logger, get_component_logger


class SimpleLLMLoggingCallbackHandler(BaseCallbackHandler):
    """
    Simple callback handler that logs LLM interactions with explicit context.
    
    No more guessing - we know exactly what each call is for because it's
    explicitly provided when the LLM is created.
    """
    
    def __init__(
        self, 
        context_name: str,
        persona_id: str,
        logging_base_dir: str,
        settings: Settings
    ):
        """
        Initialize the simple LLM logger.
        
        Args:
            context_name: Explicit context (e.g., "query_analysis", "hyde_generation")
            persona_id: The persona identifier
            logging_base_dir: Base directory for logging
            settings: Application settings
        """
        self.context_name = context_name
        self.persona_id = persona_id
        self.logging_base_dir = Path(logging_base_dir)
        self.settings = settings
        self.logger = get_component_logger("LLM-Log", f"{persona_id}:{context_name}")
        
        # Current call tracking
        self.current_call_dir = None
        self.call_start_time = None
        self.current_user_query = None
        
        # Create base directory
        self.logging_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"SimpleLLMLogger initialized for {context_name} (persona: {persona_id})")
    
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
        
        # Create timestamped folder name with explicit context
        timestamp = self.call_start_time.isoformat().replace(':', '-')
        folder_name = f"{self.context_name}_{timestamp}"
        
        # Create call directory
        self.current_call_dir = self.logging_base_dir / folder_name
        self.current_call_dir.mkdir(parents=True, exist_ok=True)
        
        # Save prompt data immediately
        self._save_prompt_data(prompts)
        
        # Save metadata
        self._save_metadata(serialized, kwargs)
        
        self.logger.info(f"Started LLM call: {self.context_name} -> {folder_name}")
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Called when LLM ends running."""
        if not self.current_call_dir or not self.call_start_time:
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
        
        # Create comprehensive log entry
        log_entry = {
            "timestamp": self.call_start_time.isoformat(),
            "context_name": self.context_name,
            "persona_id": self.persona_id,
            "user_query": self.current_user_query or "",
            "processing_time_seconds": processing_time,
            "model_info": self._safe_serialize(response.llm_output) if response.llm_output else {},
            "response_metadata": {
                "raw_response_length": len(raw_response),
                "final_answer_length": len(final_answer),
                "generation_count": len(response.generations) if response.generations else 0
            }
        }
        
        # Save comprehensive log
        self._save_log_entry(log_entry)
        
        self.logger.info(
            f"Completed LLM call: {self.context_name} "
            f"(time: {processing_time:.2f}s, response: {len(raw_response)} chars)"
        )
        
        # Reset for next call
        self.current_call_dir = None
        self.call_start_time = None
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Called when LLM encounters an error."""
        if not self.current_call_dir or not self.call_start_time:
            return
            
        # Log error information
        error_entry = {
            "timestamp": self.call_start_time.isoformat(),
            "context_name": self.context_name,
            "persona_id": self.persona_id,
            "user_query": self.current_user_query or "",
            "error": str(error),
            "error_type": type(error).__name__
        }
        
        # Save error log
        error_file = self.current_call_dir / "error.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_entry, f, indent=2, ensure_ascii=False)
        
        self.logger.error(f"LLM call failed: {self.context_name} - {error}")
        
        # Reset for next call
        self.current_call_dir = None
        self.call_start_time = None
    
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
        
        # For JSON responses, try to parse and extract relevant fields
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
        
        # For text responses, return the raw text
        return raw_response.strip()
    
    def _save_prompt_data(self, prompts: List[str]):
        """Save the prompt data to markdown file."""
        if not self.current_call_dir:
            return
            
        # Create markdown content with all prompts
        markdown_content = f"# LLM Prompt Data\\n\\n"
        markdown_content += f"**Context:** {self.context_name}\\n\\n"
        markdown_content += f"**Persona:** {self.persona_id}\\n\\n"
        markdown_content += f"**Prompt Count:** {len(prompts)}\\n\\n"
        markdown_content += f"**Total Length:** {sum(len(p) for p in prompts)} characters\\n\\n"
        
        for i, prompt in enumerate(prompts):
            if len(prompts) > 1:
                markdown_content += f"## Prompt {i + 1}\\n\\n"
            else:
                markdown_content += f"## Complete Prompt\\n\\n"
            
            # Add the prompt content, preserving any existing markdown
            markdown_content += prompt + "\\n\\n"
        
        prompt_file = self.current_call_dir / "prompt.md"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def _save_response_data(self, raw_response: str, final_answer: str):
        """Save the response data to markdown files."""
        if not self.current_call_dir:
            return
        
        # Save raw response as markdown
        raw_response_content = f"# Raw LLM Response\\n\\n{raw_response}\\n"
        raw_response_file = self.current_call_dir / "raw_response.md"
        with open(raw_response_file, 'w', encoding='utf-8') as f:
            f.write(raw_response_content)
        
        # Save final answer as markdown
        final_answer_content = f"# Final Answer\\n\\n{final_answer}\\n"
        final_answer_file = self.current_call_dir / "final_answer.md" 
        with open(final_answer_file, 'w', encoding='utf-8') as f:
            f.write(final_answer_content)
    
    def _save_metadata(self, serialized: Dict[str, Any], kwargs: Dict[str, Any]):
        """Save metadata about the LLM call."""
        if not self.current_call_dir:
            return
            
        metadata = {
            "context_name": self.context_name,
            "persona_id": self.persona_id,
            "timestamp": self.call_start_time.isoformat(),
            "serialized_model": self._safe_serialize(serialized),
            "call_kwargs": self._safe_serialize(kwargs)
        }
        
        metadata_file = self.current_call_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
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