#!/usr/bin/env python3
"""
Code Intelligence Router - Serena-inspired semantic code analysis endpoints
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import logging

from ..services.code_intelligence_service import code_intelligence_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/code-intelligence", tags=["code-intelligence"])


# Request/Response Models
class ProjectActivationRequest(BaseModel):
    project_path: str
    force_refresh: bool = False


class SymbolSearchRequest(BaseModel):
    project_path: str
    name_path: str
    include_body: bool = False
    include_references: bool = False


class SymbolReplaceRequest(BaseModel):
    project_path: str
    symbol_path: str
    new_body: str


class PatternSearchRequest(BaseModel):
    project_path: str
    pattern: str
    file_pattern: Optional[str] = None
    context_lines: int = 2


class MemoryRequest(BaseModel):
    project_path: str
    name: str
    content: Optional[str] = None


@router.post("/activate-project")
async def activate_project(request: ProjectActivationRequest):
    """Activate a project for code intelligence analysis"""
    try:
        context = await code_intelligence_service.activate_project(request.project_path)
        
        return {
            "success": True,
            "project": {
                "path": context.project_root,
                "language": context.language,
                "framework": context.framework,
                "dependencies": context.dependencies,
                "memory_count": len(context.memory_items)
            }
        }
    except Exception as e:
        logger.error(f"Failed to activate project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols/overview")
async def get_symbols_overview(
    project_path: str = Query(..., description="Project path"),
    file_path: Optional[str] = Query(None, description="Specific file to analyze")
):
    """Get overview of symbols in a project or file"""
    try:
        symbols = await code_intelligence_service.get_symbols_overview(project_path, file_path)
        
        return {
            "success": True,
            "symbols": symbols,
            "count": len(symbols)
        }
    except Exception as e:
        logger.error(f"Failed to get symbols overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/symbols/find")
async def find_symbol(request: SymbolSearchRequest):
    """Find a specific symbol by name or path"""
    try:
        symbol = await code_intelligence_service.find_symbol(
            request.project_path,
            request.name_path,
            request.include_body,
            request.include_references
        )
        
        if symbol:
            return {
                "success": True,
                "symbol": symbol
            }
        else:
            return {
                "success": False,
                "message": f"Symbol '{request.name_path}' not found"
            }
    except Exception as e:
        logger.error(f"Failed to find symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/symbols/replace")
async def replace_symbol(request: SymbolReplaceRequest):
    """Replace a symbol's body with new code"""
    try:
        result = await code_intelligence_service.replace_symbol(
            request.project_path,
            request.symbol_path,
            request.new_body
        )
        
        return result
    except Exception as e:
        logger.error(f"Failed to replace symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/pattern")
async def search_pattern(request: PatternSearchRequest):
    """Search for a pattern in project files"""
    try:
        results = await code_intelligence_service.search_pattern(
            request.project_path,
            request.pattern,
            request.file_pattern,
            request.context_lines
        )
        
        return {
            "success": True,
            "results": results,
            "file_count": len(results)
        }
    except Exception as e:
        logger.error(f"Failed to search pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/save")
async def save_memory(request: MemoryRequest):
    """Save a project-specific memory"""
    try:
        result = await code_intelligence_service.save_memory(
            request.project_path,
            request.name,
            request.content
        )
        
        return result
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/load")
async def load_memory(
    project_path: str = Query(...),
    name: str = Query(...)
):
    """Load a project-specific memory"""
    try:
        content = await code_intelligence_service.load_memory(project_path, name)
        
        if content:
            return {
                "success": True,
                "content": content
            }
        else:
            return {
                "success": False,
                "message": f"Memory '{name}' not found"
            }
    except Exception as e:
        logger.error(f"Failed to load memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/list")
async def list_memories(project_path: str = Query(...)):
    """List all memories for a project"""
    try:
        memories = await code_intelligence_service.list_memories(project_path)
        
        return {
            "success": True,
            "memories": memories,
            "count": len(memories)
        }
    except Exception as e:
        logger.error(f"Failed to list memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Check code intelligence service health"""
    return {
        "status": "operational",
        "service": "code-intelligence",
        "capabilities": [
            "symbol_analysis",
            "semantic_search",
            "code_replacement",
            "project_memory",
            "pattern_search"
        ]
    }