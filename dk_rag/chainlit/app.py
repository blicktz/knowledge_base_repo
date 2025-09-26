#!/usr/bin/env python3
"""
Chainlit UI for Virtual Influencer Persona Agent
Simple chat interface with persona selection for internal testing
"""

# Fix engineio packet limit for large responses (safety net)
from engineio.payload import Payload
Payload.max_decode_packets = 200

import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Optional

import chainlit as cl
from chainlit.input_widget import Select

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dk_rag.config.settings import Settings
from dk_rag.agent.persona_agent import LangChainPersonaAgent, StreamEvent
from dk_rag.utils.logging import get_component_logger

logger = get_component_logger("Chainlit")


class ChainlitPersonaApp:
    """Manages persona agents and session state for Chainlit"""
    
    def __init__(self):
        self.settings = Settings.from_default_config()
        self.agent_cache: Dict[str, LangChainPersonaAgent] = {}
        self.available_personas = self._discover_personas()
        logger.info(f"Initialized with personas: {self.available_personas}")
    
    def _discover_personas(self) -> list[str]:
        """Discover available personas from storage directory"""
        personas_dir = Path(self.settings.get_personas_base_dir())
        if not personas_dir.exists():
            logger.warning(f"Personas directory not found: {personas_dir}")
            return ["greg_startup"]
        
        personas = []
        for persona_dir in personas_dir.iterdir():
            if persona_dir.is_dir() and persona_dir.name != "__pycache__":
                personas.append(persona_dir.name)
        
        return personas if personas else ["greg_startup"]
    
    def get_or_create_agent(self, persona_id: str) -> LangChainPersonaAgent:
        """Get cached agent or create new one"""
        if persona_id not in self.agent_cache:
            logger.info(f"Creating new agent for persona: {persona_id}")
            self.agent_cache[persona_id] = LangChainPersonaAgent(persona_id, self.settings)
        return self.agent_cache[persona_id]


app_state = ChainlitPersonaApp()


@cl.on_chat_start
async def start():
    """Initialize chat session with persona selector"""
    
    persona_choices = app_state.available_personas
    default_persona = persona_choices[0] if persona_choices else "greg_startup"
    
    settings = await cl.ChatSettings(
        [
            Select(
                id="persona",
                label="🎭 Select Influencer Persona",
                values=persona_choices,
                initial_index=0,
            )
        ]
    ).send()
    
    selected_persona = settings.get("persona", default_persona)
    session_id = str(uuid.uuid4())
    
    cl.user_session.set("persona_id", selected_persona)
    cl.user_session.set("session_id", session_id)
    
    agent = app_state.get_or_create_agent(selected_persona)
    cl.user_session.set("agent", agent)
    
    logger.info(f"Chat started - Persona: {selected_persona}, Session: {session_id}")
    
    await cl.Message(
        content=f"Welcome! You're now chatting with **{selected_persona.replace('_', ' ').title()}**.\n\nAsk me anything!"
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    """Handle persona selection changes"""
    
    new_persona = settings.get("persona")
    current_persona = cl.user_session.get("persona_id")
    
    if new_persona != current_persona:
        new_session_id = str(uuid.uuid4())
        
        cl.user_session.set("persona_id", new_persona)
        cl.user_session.set("session_id", new_session_id)
        
        agent = app_state.get_or_create_agent(new_persona)
        cl.user_session.set("agent", agent)
        
        logger.info(f"Persona switched to: {new_persona}, New session: {new_session_id}")
        
        await cl.Message(
            content=f"Switched to **{new_persona.replace('_', ' ').title()}**. Starting fresh conversation!"
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming chat messages with structured streaming and Steps"""
    
    persona_id = cl.user_session.get("persona_id")
    session_id = cl.user_session.get("session_id")
    agent: LangChainPersonaAgent = cl.user_session.get("agent")
    
    if not agent:
        await cl.Message(content="Error: Agent not initialized. Please refresh the page.").send()
        return
    
    try:
        logger.info(f"Processing message for persona: {persona_id}")
        
        # Track different components
        thinking_step = None
        tool_steps = {}  # Track tool steps by run_id
        final_message = None
        
        logger.info("DEBUG: Starting structured streaming from agent")
        
        async for stream_event in agent.process_query_structured_stream(message.content, session_id):
            logger.info(f"DEBUG: Received {stream_event.event_type} event")
            
            if stream_event.event_type == "thinking":
                # Create a collapsible thinking step
                if not thinking_step:
                    thinking_step = cl.Step(name="💭 Thinking", type="reasoning")
                    thinking_step.start = True
                    await thinking_step.send()
                
                thinking_step.output = stream_event.content
                await thinking_step.update()
                
            elif stream_event.event_type == "tool_call":
                # Create a new tool step
                tool_name = stream_event.metadata.get("tool_name", "unknown")
                run_id = stream_event.metadata.get("run_id", "unknown")
                
                tool_step = cl.Step(name=f"🔍 {tool_name}", type="tool")
                tool_step.start = True
                tool_step.output = stream_event.content
                
                tool_steps[run_id] = tool_step
                await tool_step.send()
                
            elif stream_event.event_type == "tool_result":
                # Update the corresponding tool step
                run_id = stream_event.metadata.get("run_id", "unknown")
                tool_output = stream_event.metadata.get("output", "")
                
                if run_id in tool_steps:
                    tool_step = tool_steps[run_id]
                    tool_step.output = f"{stream_event.content}\n\n**Result:**\n{tool_output}"
                    tool_step.end = True
                    await tool_step.update()
                
            elif stream_event.event_type == "final_answer":
                # Stream the final answer directly to a main message
                if not final_message:
                    final_message = cl.Message(content="")
                    await final_message.send()
                
                await final_message.stream_token(stream_event.content)
        
        # Finalize all steps
        if thinking_step:
            thinking_step.end = True
            await thinking_step.update()
        
        # Ensure tool steps are marked as ended
        for tool_step in tool_steps.values():
            if not tool_step.end:
                tool_step.end = True
                await tool_step.update()
        
        # Final update to main message if it exists
        if final_message:
            await final_message.update()
        
        logger.info("Structured response sent successfully")
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        await cl.Message(
            content=f"Sorry, I encountered an error: {str(e)}\n\nPlease try again."
        ).send()


if __name__ == "__main__":
    pass