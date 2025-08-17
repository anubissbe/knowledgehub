
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel

class APIResponse(BaseModel):
    """Standard API response model"""
    status: str
    data: Optional[Any] = None
    message: Optional[str] = None
    
class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    timestamp: str

class ErrorResponse(BaseModel):
    """Error response model"""
    status: str = "error"
    message: str
    details: Optional[Dict[str, Any]] = None
