"""Agent configuration management"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from ..config.settings import Settings


@dataclass
class AgentConfig:
    """Configuration for persona agent"""
    
    persona_id: str
    settings: Settings
    
    # Tool configuration
    enable_parallel_execution: bool = True
    tool_timeout_seconds: int = 30
    fail_fast: bool = True
    
    # LLM configuration
    synthesis_model: str = "gemini/gemini-2.5-pro"
    synthesis_temperature: float = 0.7
    synthesis_max_tokens: int = 4000
    
    # Logging configuration
    enable_logging: bool = True
    log_llm_interactions: bool = True
    
    @classmethod
    def from_settings(cls, persona_id: str, settings: Settings) -> 'AgentConfig':
        """Create agent config from settings"""
        
        agent_settings = settings.agent
        
        return cls(
            persona_id=persona_id,
            settings=settings,
            enable_parallel_execution=agent_settings.tools.parallel_execution,
            tool_timeout_seconds=agent_settings.tools.timeout_seconds,
            fail_fast=agent_settings.error_handling.throw_on_error,
            synthesis_model=agent_settings.synthesis.llm_model,
            synthesis_temperature=agent_settings.synthesis.temperature,
            synthesis_max_tokens=agent_settings.synthesis.max_tokens,
            enable_logging=agent_settings.logging.enabled,
            log_llm_interactions=agent_settings.logging.save_responses
        )