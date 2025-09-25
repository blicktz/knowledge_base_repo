"""FastAPI application for persona agent"""

import time
from datetime import datetime
from typing import Dict, Any, List

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

# Global variables for settings and caches
settings: Settings = None
logger = None
_agent_cache: Dict[str, PersonaAgent] = {}
_persona_manager: PersonaManager = None
_available_personas: List[str] = []

def initialize_app() -> FastAPI:
    """Initialize FastAPI app with proper configuration"""
    global settings, logger, _persona_manager, _available_personas
    
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

    @app.on_event("startup")
    async def startup_event():
        """Startup event to validate system readiness"""
        global _persona_manager, _available_personas
        
        logger.info("🚀 Starting Persona Agent API...")
        
        try:
            # Initialize persona manager
            _persona_manager = PersonaManager(settings)
            personas = _persona_manager.list_personas()
            _available_personas = [p['name'] for p in personas]
            
            logger.info(f"✅ Found {len(_available_personas)} available personas: {_available_personas}")
            
            # Validate that at least one persona has required knowledge bases
            if not _available_personas:
                logger.warning("⚠️ No personas found - API will have limited functionality")
            else:
                # Test one persona to ensure system is working
                test_persona = _available_personas[0]
                logger.info(f"🔍 Testing system with persona: {test_persona}")
                
                try:
                    # Try to create an agent for the test persona
                    test_agent = PersonaAgent(test_persona, settings)
                    tool_status = test_agent.get_tool_status()
                    
                    healthy_tools = sum(1 for status in tool_status.values() if status)
                    total_tools = len(tool_status)
                    
                    logger.info(f"🛠️ Tool status: {healthy_tools}/{total_tools} tools healthy")
                    
                    if healthy_tools < total_tools:
                        logger.warning(f"⚠️ Some tools are not healthy: {tool_status}")
                    else:
                        logger.info("✅ All tools initialized successfully")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to initialize test agent: {e}")
                    logger.warning("⚠️ API may have limited functionality")
            
            logger.info("🎉 Persona Agent API startup completed")
            
        except Exception as e:
            logger.error(f"❌ Startup validation failed: {e}")
            logger.warning("⚠️ API started but some features may not work correctly")

    @app.on_event("shutdown") 
    async def shutdown_event():
        """Cleanup on shutdown"""
        logger.info("🛑 Shutting down Persona Agent API...")
        
        # Clear agent cache
        global _agent_cache
        _agent_cache.clear()
        
        logger.info("✅ Shutdown completed")

    return app

# Initialize the FastAPI app
app = initialize_app()


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
        # Validate persona exists using cached list
        if request.persona_id not in _available_personas:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona '{request.persona_id}' not found. Available personas: {_available_personas}"
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
    """Enhanced health check endpoint with system validation"""
    try:
        # Basic health check
        health_data = {
            "status": "healthy",
            "version": settings.api.settings.version,
            "timestamp": datetime.now().isoformat(),
            "personas_available": len(_available_personas),
            "personas": _available_personas[:3]  # Show first 3 for brevity
        }
        
        # Test that we can create at least one agent if personas are available
        if _available_personas:
            try:
                test_persona = _available_personas[0]
                agent = get_persona_agent(test_persona)
                tool_status = agent.get_tool_status()
                healthy_tools = sum(1 for status in tool_status.values() if status)
                
                health_data["system_status"] = "operational"
                health_data["tools_healthy"] = f"{healthy_tools}/{len(tool_status)}"
                
            except Exception as e:
                health_data["system_status"] = "degraded"
                health_data["warning"] = f"Agent initialization issues: {str(e)[:100]}"
        else:
            health_data["system_status"] = "degraded"
            health_data["warning"] = "No personas available"
        
        return HealthResponse(**health_data)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version=getattr(settings, 'api', {}).get('settings', {}).get('version', 'unknown'),
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )


@app.get("/api/v1/personas", response_model=PersonaListResponse)
async def list_personas() -> PersonaListResponse:
    """List available personas using cached data"""
    try:
        return PersonaListResponse(
            personas=_available_personas,
            count=len(_available_personas)
        )
        
    except Exception as e:
        logger.error(f"Failed to list personas: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list personas: {str(e)}"
        )


@app.get("/api/v1/personas/{persona_id}/status", response_model=ToolStatusResponse)
async def get_persona_status(persona_id: str) -> ToolStatusResponse:
    """Get status of persona agent tools using cached validation"""
    try:
        # Validate persona exists using cached list
        if persona_id not in _available_personas:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona '{persona_id}' not found. Available: {_available_personas}"
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