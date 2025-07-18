"""
Claude Code Workflow Integration API Router

Provides endpoints for automatic memory capture, context extraction,
tool usage tracking, and discovery saving.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..models import get_db
from ..services.claude_workflow_integration import ClaudeWorkflowIntegration

router = APIRouter(prefix="/api/claude-workflow", tags=["claude-workflow"])
workflow_integration = ClaudeWorkflowIntegration()


@router.get("/health")
async def health_check():
    """Check if workflow integration service is healthy"""
    return {
        "status": "healthy",
        "service": "claude-workflow-integration",
        "description": "Automatic memory capture and context extraction",
        "features": [
            "conversation_memory_capture",
            "terminal_context_extraction", 
            "tool_usage_tracking",
            "discovery_saving",
            "auto_insight_extraction"
        ]
    }


@router.post("/capture/conversation")
async def capture_conversation_memory(
    body: Dict[str, Any] = Body(..., description="Request body with conversation text"),
    session_id: Optional[str] = Query(None, description="Current session ID"),
    project_id: Optional[str] = Query(None, description="Current project ID"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Automatically capture memories from conversation text
    
    Extracts:
    - Errors and their solutions
    - Commands used
    - Important discoveries
    - TODOs and notes
    - Decisions made
    """
    try:
        conversation_text = body.get("conversation_text", "")
        result = workflow_integration.capture_conversation_memory(
            db, conversation_text, session_id, project_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/capture/terminal")
async def extract_terminal_context(
    body: Dict[str, Any] = Body(..., description="Request body with terminal output"),
    command: str = Query(..., description="Command that was executed"),
    exit_code: int = Query(0, description="Command exit code"),
    execution_time: Optional[float] = Query(None, description="Execution time in seconds"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Extract context and insights from terminal output
    
    Captures:
    - Error messages and warnings
    - File paths mentioned
    - Performance metrics
    - Success/failure patterns
    """
    try:
        terminal_output = body.get("terminal_output", "")
        result = workflow_integration.extract_terminal_context(
            db, terminal_output, command, exit_code, execution_time
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/capture/tool-usage")
async def capture_tool_usage(
    tool_name: str = Query(..., description="Name of the tool used"),
    execution_time: float = Query(..., description="Tool execution time"),
    body: Dict[str, Any] = Body(..., description="Request body with tool params and result"),
    session_id: Optional[str] = Query(None, description="Current session ID"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Capture memories from Claude Code tool usage
    
    Tracks:
    - File operations (Read, Write, Edit)
    - Command executions (Bash)
    - Search queries
    - Performance metrics
    """
    try:
        tool_params = body.get("tool_params", {})
        tool_result = body.get("tool_result")
        result = workflow_integration.capture_tool_usage(
            db, tool_name, tool_params, tool_result, execution_time, session_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save/discovery")
async def save_discovery(
    discovery_type: str = Query(..., description="Type of discovery (pattern, solution, bug_fix, etc.)"),
    body: Dict[str, Any] = Body(..., description="Request body with content and context"),
    importance: str = Query("high", description="Importance level (high, medium, low)"),
    tags: Optional[List[str]] = Query(None, description="Tags for categorization"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Save an important discovery or insight
    
    Discovery types:
    - pattern: Recurring patterns found
    - solution: Problem solutions
    - bug_fix: Bug fixes discovered
    - optimization: Performance optimizations
    - architecture: Architectural decisions
    - algorithm: Algorithm implementations
    - configuration: Configuration insights
    """
    try:
        content = body.get("content", "")
        context = body.get("context", {})
        result = workflow_integration.save_discovery(
            db, discovery_type, content, context, importance, tags
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/insights")
async def auto_extract_insights(
    body: Dict[str, Any] = Body(..., description="Request body with message"),
    message_type: str = Query("assistant", description="Type of message (assistant, user, system)"),
    session_id: Optional[str] = Query(None, description="Current session ID"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Automatically extract insights from Claude's messages
    
    Extracts:
    - Implementation decisions
    - Findings and observations
    - Code snippets
    - Important statements
    """
    try:
        message = body.get("message", "")
        result = workflow_integration.auto_extract_insights(
            db, message, message_type, session_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_workflow_stats(
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    time_range: int = Query(7, description="Time range in days"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get statistics about workflow integration"""
    try:
        return workflow_integration.get_workflow_stats(db, session_id, time_range)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Batch operations

@router.post("/capture/batch")
async def batch_capture(
    body: Dict[str, Any] = Body(..., description="Request body with captures list"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Batch capture multiple items at once
    
    Each capture should have:
    - type: conversation, terminal, tool, or discovery
    - data: The specific data for that capture type
    """
    captures = body.get("captures", [])
    results = []
    
    for capture in captures:
        capture_type = capture.get('type')
        data = capture.get('data', {})
        
        try:
            if capture_type == 'conversation':
                result = workflow_integration.capture_conversation_memory(
                    db, data.get('text', ''),
                    data.get('session_id'),
                    data.get('project_id')
                )
            elif capture_type == 'terminal':
                result = workflow_integration.extract_terminal_context(
                    db, data.get('output', ''),
                    data.get('command', ''),
                    data.get('exit_code', 0),
                    data.get('execution_time')
                )
            elif capture_type == 'tool':
                result = workflow_integration.capture_tool_usage(
                    db, data.get('tool_name', ''),
                    data.get('tool_params', {}),
                    data.get('tool_result'),
                    data.get('execution_time', 0),
                    data.get('session_id')
                )
            elif capture_type == 'discovery':
                result = workflow_integration.save_discovery(
                    db, data.get('discovery_type', 'pattern'),
                    data.get('content', ''),
                    data.get('context', {}),
                    data.get('importance', 'medium'),
                    data.get('tags')
                )
            else:
                result = {'error': f'Unknown capture type: {capture_type}'}
            
            results.append({
                'type': capture_type,
                'success': 'error' not in result,
                'result': result
            })
        except Exception as e:
            results.append({
                'type': capture_type,
                'success': False,
                'error': str(e)
            })
    
    return {
        'total_captures': len(captures),
        'successful': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'results': results
    }