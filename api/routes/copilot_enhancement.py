"""
GitHub Copilot Enhancement API Routes

This module provides real webhook integration and context injection
for GitHub Copilot suggestions using genuine AI intelligence.
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..services.real_copilot_enhancement import (
    RealCopilotEnhancement, 
    CopilotRequest, 
    CopilotSuggestion, 
    CopilotFeedback
)
from ..dependencies import get_user_id, get_current_session
from ..database import get_db_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/copilot", tags=["copilot_enhancement"])
security = HTTPBearer()

# Initialize the enhancement service
copilot_service = RealCopilotEnhancement()

# Request/Response Models

class CopilotWebhookRequest(BaseModel):
    """GitHub Copilot webhook request model"""
    user_id: str
    file_path: str
    language: str
    context_before: str = Field(..., description="Code before cursor")
    context_after: str = Field(default="", description="Code after cursor") 
    cursor_position: Dict[str, int] = Field(..., description="Line and character position")
    project_id: str
    original_suggestions: List[str] = Field(..., description="Original Copilot suggestions")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class CopilotSuggestionResponse(BaseModel):
    """Enhanced Copilot suggestion response"""
    id: str
    request_id: str
    original_text: str
    enhanced_text: str
    confidence: float
    context_injected: bool
    knowledge_used: List[str]
    enhancement_explanation: Optional[str] = None
    timestamp: str

class CopilotFeedbackRequest(BaseModel):
    """User feedback on Copilot suggestions"""
    suggestion_id: str
    action: str = Field(..., pattern="^(accepted|rejected|modified)$")
    modified_text: Optional[str] = None
    time_to_decision: float = Field(..., ge=0.0)
    context_relevance: Optional[float] = Field(None, ge=0.0, le=1.0)

class CopilotMetricsResponse(BaseModel):
    """Copilot enhancement metrics"""
    request_count: int
    success_rate: float
    average_response_time: float
    enhancement_enabled: bool
    context_injection_enabled: bool
    learning_enabled: bool
    total_suggestions_enhanced: int
    user_satisfaction_score: float

class CopilotConfigRequest(BaseModel):
    """Copilot enhancement configuration"""
    enhancement_enabled: Optional[bool] = None
    context_injection_enabled: Optional[bool] = None
    learning_enabled: Optional[bool] = None
    enhancement_aggressiveness: Optional[float] = Field(None, ge=0.0, le=1.0)
    context_window_size: Optional[int] = Field(None, ge=100, le=10000)

# API Endpoints

@router.post("/webhook", response_model=List[CopilotSuggestionResponse])
async def copilot_webhook(
    request: CopilotWebhookRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session)
) -> List[CopilotSuggestionResponse]:
    """
    GitHub Copilot webhook endpoint for real-time suggestion enhancement
    
    This endpoint receives Copilot requests and enhances suggestions with:
    - Relevant memory context
    - Code pattern recognition
    - Past decision knowledge
    - Similar code patterns
    """
    
    try:
        logger.info(f"Received Copilot webhook request from user {request.user_id}")
        
        # Validate request
        if not request.original_suggestions:
            raise HTTPException(status_code=400, detail="No original suggestions provided")
        
        if len(request.original_suggestions) > 10:
            raise HTTPException(status_code=400, detail="Too many suggestions (max 10)")
        
        # Create internal request object
        copilot_request = CopilotRequest(
            id=str(uuid4()),
            user_id=request.user_id,
            file_path=request.file_path,
            language=request.language,
            context_before=request.context_before,
            context_after=request.context_after,
            cursor_position=request.cursor_position,
            project_id=request.project_id,
            timestamp=datetime.now(timezone.utc),
            metadata=request.metadata or {}
        )
        
        # Process with real AI enhancement
        enhanced_suggestions = await copilot_service.process_copilot_request(
            copilot_request, request.original_suggestions
        )
        
        # Convert to response format
        response_suggestions = []
        for suggestion in enhanced_suggestions:
            # Generate enhancement explanation
            explanation = await _generate_enhancement_explanation(
                suggestion, copilot_request
            )
            
            response_suggestions.append(CopilotSuggestionResponse(
                id=suggestion.id,
                request_id=suggestion.request_id,
                original_text=suggestion.original_text,
                enhanced_text=suggestion.enhanced_text,
                confidence=suggestion.confidence,
                context_injected=suggestion.context_injected,
                knowledge_used=suggestion.knowledge_used,
                enhancement_explanation=explanation,
                timestamp=suggestion.timestamp.isoformat()
            ))
        
        # Log metrics in background
        background_tasks.add_task(
            _log_webhook_metrics, 
            copilot_request.id, 
            len(enhanced_suggestions),
            request.user_id
        )
        
        logger.info(f"Enhanced {len(enhanced_suggestions)} Copilot suggestions for request {copilot_request.id}")
        return response_suggestions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Copilot webhook failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {e}")

@router.post("/feedback")
async def submit_copilot_feedback(
    feedback_request: CopilotFeedbackRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_user_id)
) -> Dict[str, str]:
    """
    Submit user feedback on Copilot suggestions for learning
    
    This endpoint processes user feedback to improve future suggestions:
    - Tracks acceptance/rejection patterns
    - Learns from user modifications
    - Adjusts confidence scoring
    - Updates enhancement models
    """
    
    try:
        logger.info(f"Received Copilot feedback from user {user_id} for suggestion {feedback_request.suggestion_id}")
        
        # Create feedback object
        feedback = CopilotFeedback(
            suggestion_id=feedback_request.suggestion_id,
            action=feedback_request.action,
            modified_text=feedback_request.modified_text,
            time_to_decision=feedback_request.time_to_decision,
            context_relevance=feedback_request.context_relevance,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Process feedback for learning
        await copilot_service.process_copilot_feedback(feedback)
        
        # Update user-specific learning models in background
        background_tasks.add_task(
            _update_user_learning_models,
            user_id,
            feedback
        )
        
        return {
            "status": "success",
            "message": f"Feedback processed for suggestion {feedback_request.suggestion_id}",
            "learning_applied": True
        }
        
    except Exception as e:
        logger.error(f"Feedback processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {e}")

@router.get("/metrics", response_model=CopilotMetricsResponse)
async def get_copilot_metrics(
    user_id: str = Depends(get_user_id)
) -> CopilotMetricsResponse:
    """
    Get GitHub Copilot enhancement metrics and performance data
    """
    
    try:
        # Get service metrics
        service_metrics = await copilot_service.get_enhancement_metrics()
        
        # Get additional user-specific metrics
        user_metrics = await _get_user_copilot_metrics(user_id)
        
        return CopilotMetricsResponse(
            request_count=service_metrics['request_count'],
            success_rate=service_metrics['success_rate'],
            average_response_time=service_metrics['average_response_time'],
            enhancement_enabled=service_metrics['enhancement_enabled'],
            context_injection_enabled=service_metrics['context_injection_enabled'],
            learning_enabled=service_metrics['learning_enabled'],
            total_suggestions_enhanced=user_metrics.get('total_enhanced', 0),
            user_satisfaction_score=user_metrics.get('satisfaction_score', 0.0)
        )
        
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {e}")

@router.post("/configure")
async def configure_copilot_enhancement(
    config_request: CopilotConfigRequest,
    user_id: str = Depends(get_user_id)
) -> Dict[str, str]:
    """
    Configure GitHub Copilot enhancement settings
    """
    
    try:
        # Convert to configuration dict
        config_dict = {
            k: v for k, v in config_request.dict().items() 
            if v is not None
        }
        
        if not config_dict:
            raise HTTPException(status_code=400, detail="No configuration provided")
        
        # Apply configuration
        await copilot_service.configure_enhancement(config_dict)
        
        # Store user-specific settings
        await _store_user_copilot_config(user_id, config_dict)
        
        logger.info(f"Updated Copilot configuration for user {user_id}: {config_dict}")
        
        return {
            "status": "success", 
            "message": "Copilot enhancement configuration updated",
            "applied_settings": list(config_dict.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {e}")

@router.get("/health")
async def copilot_health_check() -> Dict[str, Any]:
    """
    Health check for Copilot enhancement service
    """
    
    try:
        metrics = await copilot_service.get_enhancement_metrics()
        
        # Determine health status
        health_status = "healthy"
        if metrics['success_rate'] < 0.9:
            health_status = "degraded"
        if metrics['average_response_time'] > 2.0:
            health_status = "slow"
        
        return {
            "status": health_status,
            "service": "copilot_enhancement",
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "copilot_enhancement", 
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.post("/test")
async def test_copilot_enhancement(
    test_request: CopilotWebhookRequest,
    user_id: str = Depends(get_user_id)
) -> Dict[str, Any]:
    """
    Test endpoint for Copilot enhancement functionality
    """
    
    try:
        # Create test request
        test_request.user_id = user_id
        test_request.project_id = test_request.project_id or "test_project"
        
        # Process the test request
        response = await copilot_webhook(test_request, BackgroundTasks())
        
        return {
            "status": "success",
            "message": "Copilot enhancement test completed",
            "enhanced_suggestions": len(response),
            "test_results": {
                "context_injection_working": any(s.context_injected for s in response),
                "knowledge_used": list(set([k for s in response for k in s.knowledge_used])),
                "average_confidence": sum(s.confidence for s in response) / len(response) if response else 0.0
            }
        }
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {e}")

# Helper Functions

async def _generate_enhancement_explanation(
    suggestion: CopilotSuggestion, 
    request: CopilotRequest
) -> Optional[str]:
    """Generate human-readable explanation of enhancements made"""
    
    try:
        if not suggestion.context_injected:
            return None
        
        explanations = []
        
        if 'memory_patterns' in suggestion.knowledge_used:
            explanations.append("Applied patterns from your previous code")
        
        if 'code_patterns' in suggestion.knowledge_used:
            explanations.append("Enhanced based on recognized coding patterns")
        
        if 'past_decisions' in suggestion.knowledge_used:
            explanations.append("Incorporated past architectural decisions")
        
        if 'similar_code' in suggestion.knowledge_used:
            explanations.append("Improved based on similar code in your project")
        
        if explanations:
            return "Enhanced with: " + ", ".join(explanations)
        
        return "Enhanced with project context"
        
    except Exception as e:
        logger.warning(f"Enhancement explanation generation failed: {e}")
        return None

async def _log_webhook_metrics(
    request_id: str, 
    suggestion_count: int, 
    user_id: str
) -> None:
    """Log webhook metrics for monitoring"""
    
    try:
        # This would log metrics to monitoring system
        logger.info(f"Webhook metrics: request_id={request_id}, "
                   f"suggestions={suggestion_count}, user_id={user_id}")
        
    except Exception as e:
        logger.warning(f"Metrics logging failed: {e}")

async def _update_user_learning_models(
    user_id: str, 
    feedback: CopilotFeedback
) -> None:
    """Update user-specific learning models"""
    
    try:
        # This would update user-specific ML models
        logger.debug(f"Updating learning models for user {user_id} "
                    f"based on {feedback.action} feedback")
        
    except Exception as e:
        logger.warning(f"User learning model update failed: {e}")

async def _get_user_copilot_metrics(user_id: str) -> Dict[str, Any]:
    """Get user-specific Copilot metrics"""
    
    try:
        # This would retrieve user-specific metrics from database
        return {
            'total_enhanced': 0,
            'satisfaction_score': 0.0
        }
        
    except Exception as e:
        logger.warning(f"User metrics retrieval failed: {e}")
        return {'total_enhanced': 0, 'satisfaction_score': 0.0}

async def _store_user_copilot_config(
    user_id: str, 
    config: Dict[str, Any]
) -> None:
    """Store user-specific Copilot configuration"""
    
    try:
        # This would store config in database
        logger.debug(f"Stored Copilot config for user {user_id}: {config}")
        
    except Exception as e:
        logger.warning(f"Config storage failed: {e}")