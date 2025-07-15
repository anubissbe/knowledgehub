"""Session linking API endpoints"""

import logging
from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ....models import get_db
from ...services.session_linking import session_linking_service

logger = logging.getLogger(__name__)
router = APIRouter()


class SessionLinkSuggestionRequest(BaseModel):
    """Request for session link suggestions"""
    user_id: str
    project_id: Optional[UUID] = None
    context_hint: Optional[str] = None


class SessionChainMergeRequest(BaseModel):
    """Request to merge session chains"""
    primary_chain_session_id: UUID
    secondary_chain_session_id: UUID
    merge_reason: str = "Manual merge requested"


@router.post("/suggestions")
async def get_session_link_suggestions(
    request: SessionLinkSuggestionRequest,
    db: Session = Depends(get_db)
):
    """
    Get intelligent suggestions for linking a new session.
    
    This endpoint analyzes recent sessions, contextual similarity, and project
    relationships to suggest the best sessions to link to.
    """
    try:
        # Get all linkable session candidates
        candidates = await session_linking_service.find_linkable_sessions(
            db=db,
            user_id=request.user_id,
            project_id=request.project_id,
            context_hint=request.context_hint
        )
        
        # Format response with session details
        suggestions = []
        for candidate in candidates:
            session = candidate['session']
            suggestion = {
                'session_id': session.id,
                'link_type': candidate['link_type'],
                'priority': candidate['priority'],
                'confidence': candidate['confidence'],
                'reason': candidate['reason'],
                'session_details': {
                    'user_id': session.user_id,
                    'project_id': session.project_id,
                    'started_at': session.started_at,
                    'ended_at': session.ended_at,
                    'tags': session.tags,
                    'memory_count': session.memory_count,
                    'is_active': session.is_active
                }
            }
            suggestions.append(suggestion)
        
        logger.info(f"Found {len(suggestions)} session link suggestions for user {request.user_id}")
        
        return {
            'suggestions': suggestions,
            'total_candidates': len(suggestions),
            'best_suggestion': suggestions[0] if suggestions else None
        }
        
    except Exception as e:
        logger.error(f"Failed to get session link suggestions: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate session link suggestions"
        )


@router.get("/suggestion/best")
async def get_best_session_link_suggestion(
    user_id: str = Query(..., description="User ID"),
    project_id: Optional[UUID] = Query(None, description="Project ID"),
    context_hint: Optional[str] = Query(None, description="Context hint for similarity matching"),
    db: Session = Depends(get_db)
):
    """
    Get the single best session link suggestion.
    
    This is a simplified endpoint that returns only the top recommendation
    for quickly linking sessions.
    """
    try:
        suggestion = await session_linking_service.suggest_session_link(
            db=db,
            user_id=user_id,
            project_id=project_id,
            context_hint=context_hint
        )
        
        if suggestion:
            session = suggestion['session']
            
            return {
                'has_suggestion': True,
                'session_id': session.id,
                'link_type': suggestion['link_type'],
                'confidence': suggestion['confidence'],
                'reason': suggestion['reason'],
                'session_details': {
                    'started_at': session.started_at,
                    'tags': session.tags,
                    'memory_count': session.memory_count,
                    'is_active': session.is_active
                }
            }
        else:
            return {
                'has_suggestion': False,
                'reason': 'No suitable sessions found for linking'
            }
        
    except Exception as e:
        logger.error(f"Failed to get best session link suggestion: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get session link suggestion"
        )


@router.get("/chain/{session_id}/analysis")
async def analyze_session_chain(
    session_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Analyze a session chain for insights and patterns.
    
    Returns detailed information about the session chain including
    duration, memory counts, topics, and continuation patterns.
    """
    try:
        analysis = await session_linking_service.get_session_chain_analysis(
            db=db,
            session_id=session_id
        )
        
        if 'error' in analysis:
            raise HTTPException(status_code=404, detail=analysis['error'])
        
        logger.info(f"Analyzed session chain for {session_id}: {analysis['chain_length']} sessions")
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze session chain: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze session chain"
        )


@router.post("/chain/merge")
async def merge_session_chains(
    request: SessionChainMergeRequest,
    db: Session = Depends(get_db)
):
    """
    Merge two session chains into one.
    
    This endpoint combines two separate session chains by linking
    the root of the secondary chain to the end of the primary chain.
    """
    try:
        result = await session_linking_service.merge_session_chains(
            db=db,
            primary_chain_session_id=request.primary_chain_session_id,
            secondary_chain_session_id=request.secondary_chain_session_id,
            merge_reason=request.merge_reason
        )
        
        if result['success']:
            logger.info(
                f"Successfully merged session chains: "
                f"{request.primary_chain_session_id} + {request.secondary_chain_session_id}"
            )
            return {
                'message': 'Session chains merged successfully',
                'result': result
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to merge session chains: {result['error']}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to merge session chains: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to merge session chains"
        )


@router.get("/chain/{session_id}/full")
async def get_full_session_chain(
    session_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get all sessions in a complete chain.
    
    Returns the full session chain including ancestors and descendants
    of the specified session.
    """
    try:
        chain_sessions = await session_linking_service._get_full_session_chain(
            db=db,
            session_id=session_id
        )
        
        if not chain_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Format sessions for response
        chain_data = []
        for i, session in enumerate(chain_sessions):
            session_data = {
                'session_id': session.id,
                'position': i + 1,
                'user_id': session.user_id,
                'project_id': session.project_id,
                'parent_session_id': session.parent_session_id,
                'started_at': session.started_at,
                'ended_at': session.ended_at,
                'duration': session.duration.total_seconds() if session.duration else None,
                'memory_count': session.memory_count,
                'tags': session.tags,
                'is_active': session.is_active,
                'is_root': session.parent_session_id is None,
                'is_current': session.id == session_id
            }
            chain_data.append(session_data)
        
        logger.info(f"Retrieved full session chain for {session_id}: {len(chain_sessions)} sessions")
        
        return {
            'session_id': session_id,
            'chain_length': len(chain_sessions),
            'sessions': chain_data,
            'root_session_id': chain_sessions[0].id if chain_sessions else None,
            'total_memories': sum(s.memory_count for s in chain_sessions),
            'total_duration_seconds': sum(
                s.duration.total_seconds() if s.duration else 0 
                for s in chain_sessions
            )
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get full session chain: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve session chain"
        )


@router.get("/health")
async def session_linking_health():
    """Health check for session linking service"""
    return {
        'status': 'healthy',
        'service': 'session_linking',
        'features': [
            'intelligent_suggestions',
            'chain_analysis',
            'chain_merging',
            'contextual_similarity',
            'project_relationships'
        ],
        'configuration': {
            'max_chain_length': session_linking_service.max_chain_length,
            'similarity_threshold': session_linking_service.similarity_threshold,
            'recent_window_minutes': session_linking_service.recent_window_minutes,
            'related_window_hours': session_linking_service.related_window_hours
        }
    }