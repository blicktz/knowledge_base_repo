"""
LangChain-native FastAPI application for persona agent system
Complete rewrite using the new LangChain agent architecture
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..agent.persona_agent import LangChainPersonaAgent, create_persona_agent
from ..core.persona_manager import PersonaManager
from ..config.settings import Settings
from ..utils.logging import get_logger

# Global variables
settings: Settings = None
logger = None
_agent_cache: Dict[str, LangChainPersonaAgent] = {}
_persona_manager: PersonaManager = None
_available_personas: List[str] = []


# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The user query to process")
    persona_id: str = Field(..., description="The persona identifier")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation continuity")
    stream: bool = Field(False, description="Whether to stream the response")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")


class QueryResponse(BaseModel):
    response: str = Field(..., description="The persona's response")
    persona_id: str = Field(..., description="The persona that responded")
    session_id: str = Field(..., description="The session ID used")
    processing_time: float = Field(..., description="Time taken to process the query")
    framework: str = Field(..., description="Framework used (LangChain)")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")


class ConversationHistoryResponse(BaseModel):
    session_id: str = Field(..., description="The session ID")
    messages: List[Dict[str, Any]] = Field(..., description="List of conversation messages")
    message_count: int = Field(..., description="Total number of messages")


class HealthResponse(BaseModel):
    status: str = Field(..., description="System health status")
    version: str = Field(..., description="API version")
    framework: str = Field(..., description="Agent framework")
    timestamp: str = Field(..., description="Health check timestamp")
    personas_available: int = Field(..., description="Number of available personas")
    agents_cached: int = Field(..., description="Number of cached agents")


class PersonaListResponse(BaseModel):
    personas: List[str] = Field(..., description="List of available personas")
    count: int = Field(..., description="Total number of personas")


class AgentInfoResponse(BaseModel):
    persona_id: str = Field(..., description="The persona identifier")
    agent_info: Dict[str, Any] = Field(..., description="Agent configuration details")
    cached: bool = Field(..., description="Whether the agent is cached")


def initialize_app() -> FastAPI:
    """Initialize FastAPI app with LangChain agent configuration"""
    global settings, logger, _persona_manager, _available_personas
    
    # Initialize settings and logger
    settings = Settings()
    logger = get_logger(__name__)
    
    # Create FastAPI app
    app = FastAPI(
        title="LangChain Persona Agent API",
        version="2.0.0",
        description="LangChain-native Persona Agent System with ReAct pattern and conversation memory",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    if settings.api.cors.enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors.allow_origins,
            allow_methods=settings.api.cors.allow_methods,
            allow_headers=settings.api.cors.allow_headers,
        )
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize system on startup"""
        global _persona_manager, _available_personas
        
        try:
            logger.info("🚀 Starting LangChain Persona Agent API...")
            
            # Initialize persona manager
            _persona_manager = PersonaManager(settings)
            personas = _persona_manager.list_personas()
            _available_personas = [p['name'] for p in personas]
            
            logger.info(f"✅ Initialized with {len(_available_personas)} personas: {_available_personas}")
            logger.info(f"🔧 Framework: LangChain ReAct Agent with MemorySaver")
            logger.info(f"🌐 API available at: http://{settings.api.host}:{settings.api.port}")
            logger.info(f"📚 Docs available at: http://{settings.api.host}:{settings.api.port}/docs")
            
        except Exception as e:
            logger.error(f"❌ Startup failed: {str(e)}")
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        global _agent_cache
        logger.info("🛑 Shutting down Persona Agent API...")
        _agent_cache.clear()
        logger.info("✅ Shutdown complete")
    
    # API Endpoints
    
    @app.post("/api/v1/query", response_model=QueryResponse)
    async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
        """Process a user query through the LangChain persona agent"""
        
        start_time = time.time()
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"🔄 Processing query for persona: {request.persona_id}, session: {session_id[:8]}...")
        
        try:
            # Validate persona
            if request.persona_id not in _available_personas:
                raise HTTPException(
                    status_code=400,
                    detail=f"Persona '{request.persona_id}' not found. Available: {_available_personas}"
                )
            
            # Get or create agent
            agent = await get_or_create_agent(request.persona_id)
            
            # Process the query using LangChain agent
            response_text = agent.process_query(request.query, session_id)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log success
            logger.info(f"✅ Query processed successfully in {processing_time:.2f}s")
            
            return QueryResponse(
                response=response_text,
                persona_id=request.persona_id,
                session_id=session_id,
                processing_time=processing_time,
                framework="LangChain ReAct Agent",
                metadata={
                    "query_length": len(request.query),
                    "response_length": len(response_text),
                    "tools_available": len(agent.tools),
                    "has_memory": True
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Query processing failed after {processing_time:.2f}s: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
    @app.get("/api/v1/conversation/{session_id}", response_model=ConversationHistoryResponse)
    async def get_conversation_history(session_id: str, persona_id: str):
        """Get conversation history for a session"""
        
        try:
            # Validate persona
            if persona_id not in _available_personas:
                raise HTTPException(status_code=400, detail=f"Persona '{persona_id}' not found")
            
            # Get agent
            agent = await get_or_create_agent(persona_id)
            
            # Get conversation history
            messages = agent.get_conversation_history(session_id)
            
            return ConversationHistoryResponse(
                session_id=session_id,
                messages=messages,
                message_count=len(messages)
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get conversation history: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/api/v1/conversation/{session_id}")
    async def clear_conversation(session_id: str, persona_id: str):
        """Clear conversation history for a session"""
        
        try:
            # Validate persona
            if persona_id not in _available_personas:
                raise HTTPException(status_code=400, detail=f"Persona '{persona_id}' not found")
            
            # Get agent
            agent = await get_or_create_agent(persona_id)
            
            # Clear conversation
            success = agent.clear_conversation(session_id)
            
            if success:
                return {"status": "success", "message": f"Conversation {session_id} cleared"}
            else:
                raise HTTPException(status_code=500, detail="Failed to clear conversation")
                
        except Exception as e:
            logger.error(f"❌ Failed to clear conversation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/personas", response_model=PersonaListResponse)
    async def list_personas():
        """List all available personas"""
        return PersonaListResponse(
            personas=_available_personas,
            count=len(_available_personas)
        )
    
    @app.get("/api/v1/agent/{persona_id}", response_model=AgentInfoResponse)
    async def get_agent_info(persona_id: str):
        """Get information about a specific agent"""
        
        try:
            # Validate persona
            if persona_id not in _available_personas:
                raise HTTPException(status_code=400, detail=f"Persona '{persona_id}' not found")
            
            # Check if agent is cached
            is_cached = persona_id in _agent_cache
            
            # Get agent info
            if is_cached:
                agent_info = _agent_cache[persona_id].get_agent_info()
            else:
                # Create temporary agent to get info
                temp_agent = create_persona_agent(persona_id, settings)
                agent_info = temp_agent.get_agent_info()
            
            return AgentInfoResponse(
                persona_id=persona_id,
                agent_info=agent_info,
                cached=is_cached
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get agent info: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health_check():
        """Enhanced health check with system status"""
        
        try:
            # Test agent creation
            test_persona = _available_personas[0] if _available_personas else None
            if test_persona:
                test_agent = await get_or_create_agent(test_persona)
                agent_healthy = test_agent is not None
            else:
                agent_healthy = False
            
            status = "healthy" if agent_healthy else "degraded"
            
            return HealthResponse(
                status=status,
                version="2.0.0",
                framework="LangChain ReAct Agent",
                timestamp=datetime.now().isoformat(),
                personas_available=len(_available_personas),
                agents_cached=len(_agent_cache)
            )
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {str(e)}")
            return HealthResponse(
                status="unhealthy",
                version="2.0.0",
                framework="LangChain ReAct Agent",
                timestamp=datetime.now().isoformat(),
                personas_available=len(_available_personas),
                agents_cached=len(_agent_cache)
            )
    
    return app


async def get_or_create_agent(persona_id: str) -> LangChainPersonaAgent:
    """Get cached agent or create new one"""
    global _agent_cache
    
    if persona_id not in _agent_cache:
        logger.info(f"Creating new LangChain agent for persona: {persona_id}")
        _agent_cache[persona_id] = create_persona_agent(persona_id, settings)
        logger.info(f"✅ Agent cached for: {persona_id}")
    
    return _agent_cache[persona_id]


# Create the app instance
app = initialize_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)