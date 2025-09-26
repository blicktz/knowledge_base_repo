#!/usr/bin/env python3
"""
Gradio Chat UI for Virtual Influencer Persona Agent
Simple, standalone interface for internal testing
"""

import os
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

import gradio as gr

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dk_rag.config.settings import Settings
from dk_rag.agent.persona_agent import LangChainPersonaAgent
from dk_rag.utils.logging import get_logger

logger = get_logger(__name__)


class GradioPersonaChat:
    
    def __init__(self):
        self.settings = Settings.from_default_config()
        self.agent_cache: Dict[str, LangChainPersonaAgent] = {}
        self.session_map: Dict[str, str] = {}
        self.available_personas = self._discover_personas()
        
        logger.info(f"Initialized Gradio UI with personas: {self.available_personas}")
    
    def _discover_personas(self) -> List[str]:
        personas_dir = Path(self.settings.get_personas_base_dir())
        if not personas_dir.exists():
            logger.warning(f"Personas directory not found: {personas_dir}")
            return ["greg_startup"]
            
        personas = []
        for persona_dir in personas_dir.iterdir():
            if persona_dir.is_dir() and persona_dir.name != "__pycache__":
                personas.append(persona_dir.name)
        
        return personas if personas else ["greg_startup"]
    
    def _get_or_create_agent(self, persona_id: str) -> LangChainPersonaAgent:
        if persona_id not in self.agent_cache:
            logger.info(f"Creating new agent for persona: {persona_id}")
            self.agent_cache[persona_id] = LangChainPersonaAgent(persona_id, self.settings)
        return self.agent_cache[persona_id]
    
    def _get_session_id(self, request: gr.Request) -> str:
        session_key = str(request.session_hash) if hasattr(request, 'session_hash') else "default"
        
        if session_key not in self.session_map:
            self.session_map[session_key] = str(uuid.uuid4())
            logger.info(f"Created new session: {self.session_map[session_key]}")
        
        return self.session_map[session_key]
    
    def chat_fn(self, message: str, history: List[List[str]], persona_id: str, request: gr.Request) -> str:
        if not message or not message.strip():
            return ""
        
        try:
            logger.info(f"Processing message for persona: {persona_id}")
            
            agent = self._get_or_create_agent(persona_id)
            session_id = self._get_session_id(request)
            
            response = agent.process_query(message, session_id)
            
            logger.info(f"Response generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            return f"Sorry, I encountered an error: {str(e)}"
    
    def clear_conversation(self, persona_id: str, request: gr.Request):
        try:
            session_id = self._get_session_id(request)
            
            if persona_id in self.agent_cache:
                agent = self.agent_cache[persona_id]
                agent.clear_conversation(session_id)
                logger.info(f"Cleared conversation for session: {session_id}")
            
            return None
        except Exception as e:
            logger.error(f"Error clearing conversation: {str(e)}")
            return None
    
    def build_interface(self) -> gr.Blocks:
        with gr.Blocks(
            title="Virtual Influencer Chat",
            theme=gr.themes.Soft()
        ) as demo:
            
            gr.Markdown(
                """
                # 🎭 Virtual Influencer Persona Agent
                Chat with AI personas trained on influencer content
                """
            )
            
            with gr.Row():
                persona_dropdown = gr.Dropdown(
                    choices=self.available_personas,
                    value=self.available_personas[0] if self.available_personas else None,
                    label="Select Influencer Persona",
                    info="Choose which influencer you want to chat with"
                )
            
            chatbot = gr.ChatInterface(
                fn=self.chat_fn,
                additional_inputs=[persona_dropdown],
                type="messages",
                title=None,
                description=None,
                chatbot=gr.Chatbot(
                    type="messages",
                    height=500,
                    show_copy_button=True,
                    show_share_button=False,
                    avatar_images=None
                ),
                textbox=gr.Textbox(
                    placeholder="Type your message here...",
                    scale=7
                ),
                submit_btn="Send",
                stop_btn=True
            )
            
            gr.Markdown(
                """
                ---
                ### Tips:
                - Switch personas using the dropdown above
                - Chat history is maintained per session
                - Use Clear to start a fresh conversation
                """
            )
        
        return demo
    
    def launch(self, **kwargs):
        demo = self.build_interface()
        demo.launch(**kwargs)


def main():
    try:
        chat_app = GradioPersonaChat()
        chat_app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    except KeyboardInterrupt:
        logger.info("Shutting down Gradio UI")
    except Exception as e:
        logger.error(f"Failed to start Gradio UI: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()