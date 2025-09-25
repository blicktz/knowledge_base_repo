"""FastAPI application for persona agent"""

import time
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    QueryRequest, 
    QueryResponse, 
    HealthResponse, 
    PersonaListResponse,
    ToolStatusResponse
)
from ..agent.persona_agent import PersonaAgent
from ..core.persona_manager import PersonaManager
from ..config.settings import Settings
from ..utils.logging import get_logger

# Initialize settings and logger
settings = Settings()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.api.settings.title,
    version=settings.api.settings.version,
    docs_url=settings.api.settings.docs_url,
    redoc_url=settings.api.settings.redoc_url,
    description="Phase 3 Agentic Persona System with Multi-Knowledge Retrieval"
)

# Add CORS middleware
if settings.api.cors.enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors.allow_origins,
        allow_methods=settings.api.cors.allow_methods,
        allow_headers=settings.api.cors.allow_headers,
    )

# Cache for persona agents (in production, use Redis)
_agent_cache: Dict[str, PersonaAgent] = {}


def get_persona_agent(persona_id: str) -> PersonaAgent:
    """Get or create persona agent instance"""
    if persona_id not in _agent_cache:
        logger.info(f"Creating new persona agent for: {persona_id}")
        _agent_cache[persona_id] = PersonaAgent(persona_id, settings)
    
    return _agent_cache[persona_id]


@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest) -> QueryResponse:
    """Process a user query through the persona agent"""
    
    logger.info(f"API request for persona: {request.persona_id}")
    start_time = time.time()
    
    try:
        # Validate persona exists
        persona_manager = PersonaManager(settings)
        personas = persona_manager.list_personas()
        
        persona_names = [p['name'] for p in personas]
        if request.persona_id not in persona_names:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona '{request.persona_id}' not found. Available personas: {persona_names}"
            )
        
        # Get persona agent
        agent = get_persona_agent(request.persona_id)
        
        # Process the query
        logger.info(f"Processing query: {request.query[:100]}...")
        response = agent.process_query(request.query)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(f"API request completed in {processing_time:.2f}s")
        
        return QueryResponse(
            response=response,
            persona_id=request.persona_id,
            processing_time=processing_time,
            metadata={
                "query_length": len(request.query),
                "response_length": len(response),
                "tools_executed": 5,  # All 5 tools
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"API request failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=settings.api.settings.version,
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/v1/personas", response_model=PersonaListResponse)
async def list_personas() -> PersonaListResponse:
    """List available personas"""
    try:
        manager = PersonaManager(settings)
        personas = manager.list_personas()
        persona_names = [p['name'] for p in personas]
        
        return PersonaListResponse(
            personas=persona_names,
            count=len(persona_names)
        )
        
    except Exception as e:
        logger.error(f"Failed to list personas: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list personas: {str(e)}"
        )


@app.get("/api/v1/personas/{persona_id}/status", response_model=ToolStatusResponse)
async def get_persona_status(persona_id: str) -> ToolStatusResponse:
    """Get status of persona agent tools"""
    try:
        # Validate persona exists
        persona_manager = PersonaManager(settings)
        personas = persona_manager.list_personas()
        
        persona_names = [p['name'] for p in personas]
        if persona_id not in persona_names:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona '{persona_id}' not found"
            )
        
        # Get or create agent
        agent = get_persona_agent(persona_id)
        tool_status = agent.get_tool_status()
        
        return ToolStatusResponse(
            persona_id=persona_id,
            tools=tool_status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get persona status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get persona status: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Persona Agent API",
        "version": settings.api.settings.version,
        "docs": settings.api.settings.docs_url,
        "health": "/api/v1/health"
    }