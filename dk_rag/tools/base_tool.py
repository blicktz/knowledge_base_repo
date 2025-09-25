"""Base tool class for all persona agent tools"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..config.settings import Settings
from ..utils.logging import get_logger


class ToolInput(BaseModel):
    """Base input model for all tools"""
    query: str = Field(..., description="The query or input to process")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")


class BasePersonaTool(BaseTool, ABC):
    """Base class for all persona agent tools"""
    
    # Pydantic model configuration
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, persona_id: str, settings: Settings, **kwargs):
        # Initialize tool first with any kwargs
        super().__init__(**kwargs)
        
        # Then set our custom attributes using object.__setattr__ to bypass Pydantic
        object.__setattr__(self, 'persona_id', persona_id)
        object.__setattr__(self, 'settings', settings)
        object.__setattr__(self, 'logger', get_logger(f"{__name__}.{self.__class__.__name__}"))
    
    def _run(self, query: str, metadata: Optional[Dict] = None) -> Any:
        """Execute the tool with comprehensive logging"""
        self.logger.info(f"Starting {self.name} for query: {query[:100]}...")
        
        try:
            result = self.execute(query, metadata)
            self.logger.info(f"{self.name} completed successfully")
            return result
        except Exception as e:
            self.logger.error(
                f"{self.name} failed: {str(e)}", 
                exc_info=True,
                extra={
                    'persona_id': self.persona_id,
                    'component': self.__class__.__name__,
                    'operation': 'execute'
                }
            )
            raise
    
    async def _arun(self, query: str, metadata: Optional[Dict] = None) -> Any:
        """Async version - not implemented yet"""
        raise NotImplementedError("Async execution not implemented")
    
    @abstractmethod
    def execute(self, query: str, metadata: Optional[Dict]) -> Any:
        """Tool-specific execution logic"""
        pass
    
    def log_llm_interaction(
        self, 
        prompt: str, 
        response: str, 
        extracted: Any,
        component_name: str = None
    ):
        """Save LLM interaction for debugging"""
        if not self.settings.agent.logging.enabled:
            return
            
        try:
            # Determine component directory
            component = component_name or self.name
            
            # Build log directory path
            base_dir = Path(self.settings.base_storage_dir)
            log_dir = base_dir / "personas" / self.persona_id / "logging" / "llm_interactions" / component
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_file = log_dir / f"{timestamp}_interaction.json"
            
            # Build interaction data
            interaction_data = {
                "timestamp": datetime.now().isoformat(),
                "component": component,
                "persona_id": self.persona_id,
                "input": {
                    "query": prompt[:1000] if len(prompt) > 1000 else prompt,
                    "metadata": {}
                },
                "prompt": {
                    "template": "See filled prompt",
                    "filled": prompt
                },
                "response": {
                    "raw": response,
                    "model": getattr(self, 'model_name', 'unknown'),
                    "tokens": {"input": -1, "output": -1},  # Placeholder
                    "latency_ms": -1  # Placeholder
                },
                "extracted": extracted
            }
            
            # Save to file
            with open(log_file, 'w') as f:
                json.dump(interaction_data, f, indent=2, default=str)
                
            self.logger.debug(f"LLM interaction logged to: {log_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log LLM interaction: {str(e)}")
    
    def _get_cache_key(self, query: str, metadata: Optional[Dict] = None) -> str:
        """Generate cache key for query results"""
        import hashlib
        
        cache_input = f"{self.persona_id}:{self.name}:{query}"
        if metadata:
            cache_input += f":{json.dumps(metadata, sort_keys=True)}"
            
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_timestamp: datetime, ttl_hours: int = 24) -> bool:
        """Check if cached data is still valid"""
        if not cache_timestamp:
            return False
            
        age_hours = (datetime.now() - cache_timestamp).total_seconds() / 3600
        return age_hours < ttl_hours