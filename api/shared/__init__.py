"""
Shared imports module for KnowledgeHub
Consolidates commonly used imports to reduce duplication and improve load times.
"""

# Standard library imports
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import uuid4, UUID

# Third-party imports  
import aiohttp
import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB

# FastAPI imports
from fastapi import (
    FastAPI,
    APIRouter, 
    Depends, 
    HTTPException, 
    Query, 
    Body, 
    Path as FastAPIPath,
    Request,
    Response,
    BackgroundTasks,
    File,
    UploadFile,
    status,
    Header
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Database and ORM
Base = declarative_base()

# Logging setup
logger = logging.getLogger(__name__)

# Common exceptions
class ServiceException(Exception):
    """Base service exception"""
    pass

class ValidationException(ServiceException):
    """Validation error exception"""
    pass

class DatabaseException(ServiceException):
    """Database operation exception"""  
    pass

# Common utility functions
def get_current_timestamp() -> datetime:
    """Get current UTC timestamp"""
    return datetime.utcnow()

def generate_uuid() -> str:
    """Generate UUID string"""
    return str(uuid4())

def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely parse JSON with fallback"""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default

def validate_uuid(uuid_string: str) -> bool:
    """Validate UUID format"""
    try:
        UUID(uuid_string)
        return True
    except (ValueError, TypeError):
        return False

# Common response models
class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=get_current_timestamp)

class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class PaginatedResponse(BaseResponse):
    """Paginated response model"""
    data: List[Any]
    total: int
    page: int = 1
    limit: int = 50
    has_next: bool = False
    has_previous: bool = False
