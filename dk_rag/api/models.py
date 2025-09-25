"""API request and response models"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for persona queries"""
    query: str = Field(..., description="User query to process", min_length=1, max_length=2000)
    persona_id: str = Field(..., description="ID of the persona to use", min_length=1, max_length=100)
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "How can I improve my sales conversion rates?",
                "persona_id": "dan_kennedy",
                "metadata": {}
            }
        }
    }


class QueryResponse(BaseModel):
    """Response model for persona queries"""
    response: str = Field(..., description="Generated persona response")
    persona_id: str = Field(..., description="ID of the persona used")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "response": "Listen, if you want to improve your sales conversion rates...",
                "persona_id": "dan_kennedy",
                "processing_time": 2.34,
                "metadata": {
                    "query_length": 42,
                    "response_length": 156,
                    "tools_executed": 5
                }
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2025-09-25T10:30:00Z"
            }
        }
    }


class PersonaListResponse(BaseModel):
    """Response model for listing personas"""
    personas: list[str] = Field(..., description="List of available persona IDs")
    count: int = Field(..., description="Number of personas")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "personas": ["dan_kennedy", "gary_vaynerchuk"],
                "count": 2
            }
        }
    }


class ToolStatusResponse(BaseModel):
    """Response model for tool status"""
    persona_id: str = Field(..., description="Persona ID")
    tools: Dict[str, bool] = Field(..., description="Tool health status")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "persona_id": "dan_kennedy",
                "tools": {
                    "query_analyzer": True,
                    "persona_data": True,
                    "mental_models_retriever": True,
                    "core_beliefs_retriever": True,
                    "transcript_retriever": True
                }
            }
        }
    }