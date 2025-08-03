"""
GitHub Copilot Webhook Receiver Router.

Provides REST API endpoints for receiving and processing GitHub Copilot
webhook events and enhancement requests.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field

from ..auth.dependencies import get_current_user_optional
from ..services.copilot_service import CopilotEnhancementService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/copilot", tags=["copilot"])

# Initialize service
copilot_service = CopilotEnhancementService()


# Request/Response Models
class WebhookRequest(BaseModel):
    """GitHub Copilot webhook request model."""
    
    webhook_type: str = Field(..., description="Type of webhook event")
    payload: Dict[str, Any] = Field(..., description="Webhook payload data")
    timestamp: Optional[str] = Field(None, description="Event timestamp")
    user_id: Optional[str] = Field(None, description="User ID for context")
    project_id: Optional[str] = Field(None, description="Project ID for context")


class SuggestionEnhanceRequest(BaseModel):
    """Request to enhance a Copilot suggestion."""
    
    original_suggestion: str = Field(..., description="Original Copilot suggestion")
    context: Dict[str, Any] = Field(..., description="Code and project context")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    project_id: Optional[str] = Field(None, description="Project ID for context")


class ContextInjectionRequest(BaseModel):
    """Request to inject context into Copilot request."""
    
    request: Dict[str, Any] = Field(..., description="Original Copilot request")
    user_id: Optional[str] = Field(None, description="User ID for context")
    project_id: Optional[str] = Field(None, description="Project ID for context")


class FeedbackRequest(BaseModel):
    """Feedback request for suggestion improvement."""
    
    suggestion_id: str = Field(..., description="ID of the suggestion")
    feedback_type: str = Field(..., description="Type of feedback")
    feedback_data: Dict[str, Any] = Field(default_factory=dict, description="Additional feedback data")
    user_id: Optional[str] = Field(None, description="User providing feedback")


class SuggestionResponse(BaseModel):
    """Enhanced suggestion response."""
    
    suggestion_id: str = Field(..., description="Unique suggestion ID")
    original_suggestion: str = Field(..., description="Original suggestion")
    enhanced_suggestion: str = Field(..., description="Enhanced suggestion")
    confidence: float = Field(..., description="Enhancement confidence score")
    context_sources: List[str] = Field(..., description="Sources used for enhancement")
    timestamp: str = Field(..., description="Enhancement timestamp")


# Webhook Endpoints
@router.post("/webhook")
async def receive_webhook(
    webhook_request: WebhookRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user_optional)
):
    """
    Receive GitHub Copilot webhook events.
    
    Processes various webhook events from GitHub Copilot including suggestion
    requests, acceptances, rejections, and feedback.
    """
    try:
        # Extract user context
        user_id = webhook_request.user_id or (current_user.get("user_id") if current_user else None)
        
        # Process webhook in background to avoid blocking
        background_tasks.add_task(
            _process_webhook_background,
            webhook_request.webhook_type,
            webhook_request.payload,
            user_id
        )
        
        # Return immediate response
        return {
            "status": "received",
            "webhook_type": webhook_request.webhook_type,
            "timestamp": datetime.utcnow().isoformat(),
            "processing": "background"
        }
        
    except Exception as e:
        logger.error(f"Error receiving webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {e}")


@router.post("/webhook/immediate")
async def receive_webhook_immediate(
    webhook_request: WebhookRequest,
    current_user: Optional[Dict] = Depends(get_current_user_optional)
):
    """
    Receive GitHub Copilot webhook events with immediate processing.
    
    For webhooks that require immediate response (like suggestion requests).
    """
    try:
        # Extract user context
        user_id = webhook_request.user_id or (current_user.get("user_id") if current_user else None)
        
        # Process webhook immediately
        result = await copilot_service.receive_webhook(
            webhook_request.webhook_type,
            webhook_request.payload,
            user_id
        )
        
        return {
            "status": "processed",
            "webhook_type": webhook_request.webhook_type,
            "timestamp": datetime.utcnow().isoformat(),
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error processing immediate webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {e}")


# Enhancement Endpoints
@router.post("/enhance", response_model=SuggestionResponse)
async def enhance_suggestion(
    enhance_request: SuggestionEnhanceRequest,
    current_user: Optional[Dict] = Depends(get_current_user_optional)
):
    """
    Enhance a GitHub Copilot suggestion with KnowledgeHub intelligence.
    
    Takes an original Copilot suggestion and enhances it using project context,
    relevant memories, detected patterns, and AI intelligence.
    """
    try:
        # Extract user context
        user_id = enhance_request.user_id or (current_user.get("user_id") if current_user else None)
        
        # Enhance the suggestion
        enhanced = await copilot_service.enhance_suggestion(
            enhance_request.original_suggestion,
            enhance_request.context,
            user_id,
            enhance_request.project_id
        )
        
        return SuggestionResponse(
            suggestion_id=enhanced.id,
            original_suggestion=enhanced.original_suggestion,
            enhanced_suggestion=enhanced.enhanced_suggestion,
            confidence=enhanced.confidence,
            context_sources=enhanced.context_sources,
            timestamp=enhanced.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error enhancing suggestion: {e}")
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {e}")


@router.post("/context/inject")
async def inject_context(
    injection_request: ContextInjectionRequest,
    current_user: Optional[Dict] = Depends(get_current_user_optional)
):
    """
    Inject KnowledgeHub context into a Copilot request.
    
    Enhances Copilot requests with relevant project context, session state,
    technical decisions, and other KnowledgeHub intelligence.
    """
    try:
        # Extract user context
        user_id = injection_request.user_id or (current_user.get("user_id") if current_user else None)
        
        # Inject context
        enhanced_request = await copilot_service.inject_context(
            injection_request.request,
            user_id,
            injection_request.project_id
        )
        
        return {
            "status": "context_injected",
            "enhanced_request": enhanced_request,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error injecting context: {e}")
        raise HTTPException(status_code=500, detail=f"Context injection failed: {e}")


# Feedback Endpoints
@router.post("/feedback")
async def provide_feedback(
    feedback_request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user_optional)
):
    """
    Provide feedback on a Copilot suggestion for continuous learning.
    
    Records feedback (acceptance, rejection, modification) to improve future
    suggestions and enhance the AI learning process.
    """
    try:
        # Extract user context
        user_id = feedback_request.user_id or (current_user.get("user_id") if current_user else None)
        
        # Process feedback in background
        background_tasks.add_task(
            _process_feedback_background,
            feedback_request.suggestion_id,
            feedback_request.feedback_type,
            feedback_request.feedback_data,
            user_id
        )
        
        return {
            "status": "feedback_received",
            "suggestion_id": feedback_request.suggestion_id,
            "feedback_type": feedback_request.feedback_type,
            "timestamp": datetime.utcnow().isoformat(),
            "processing": "background"
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {e}")


@router.post("/feedback/immediate")
async def provide_feedback_immediate(
    feedback_request: FeedbackRequest,
    current_user: Optional[Dict] = Depends(get_current_user_optional)
):
    """
    Provide feedback with immediate processing and response.
    
    For feedback that requires immediate learning updates.
    """
    try:
        # Extract user context
        user_id = feedback_request.user_id or (current_user.get("user_id") if current_user else None)
        
        # Add user context to feedback data
        enhanced_feedback_data = feedback_request.feedback_data.copy()
        if user_id:
            enhanced_feedback_data["user_id"] = user_id
        
        # Process feedback immediately
        result = await copilot_service.create_feedback_loop(
            feedback_request.suggestion_id,
            feedback_request.feedback_type,
            enhanced_feedback_data
        )
        
        return {
            "status": "feedback_processed",
            "suggestion_id": feedback_request.suggestion_id,
            "feedback_type": feedback_request.feedback_type,
            "timestamp": datetime.utcnow().isoformat(),
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error processing immediate feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {e}")


# Analytics Endpoints
@router.get("/analytics/suggestions")
async def get_suggestion_analytics(
    time_window: str = "24h",
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    current_user: Optional[Dict] = Depends(get_current_user_optional)
):
    """
    Get analytics data for Copilot suggestions and enhancements.
    
    Provides insights into suggestion performance, enhancement effectiveness,
    and user feedback patterns.
    """
    try:
        # Use current user if no specific user provided
        if not user_id and current_user:
            user_id = current_user.get("user_id")
        
        # TODO: Implement analytics gathering
        # This would integrate with the analytics service
        
        return {
            "time_window": time_window,
            "user_id": user_id,
            "project_id": project_id,
            "total_suggestions": 0,
            "enhancement_rate": 0.0,
            "acceptance_rate": 0.0,
            "avg_confidence": 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting suggestion analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {e}")


@router.get("/analytics/feedback")
async def get_feedback_analytics(
    time_window: str = "24h",
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    current_user: Optional[Dict] = Depends(get_current_user_optional)
):
    """
    Get analytics data for Copilot feedback and learning.
    
    Provides insights into feedback patterns, learning effectiveness,
    and continuous improvement metrics.
    """
    try:
        # Use current user if no specific user provided
        if not user_id and current_user:
            user_id = current_user.get("user_id")
        
        # TODO: Implement feedback analytics
        # This would integrate with the analytics service
        
        return {
            "time_window": time_window,
            "user_id": user_id,
            "project_id": project_id,
            "total_feedback": 0,
            "feedback_types": {},
            "learning_rate": 0.0,
            "improvement_score": 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting feedback analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {e}")


# Health Check
@router.get("/health")
async def health_check():
    """Health check endpoint for Copilot service."""
    try:
        return {
            "status": "healthy",
            "service": "copilot_enhancement",
            "timestamp": datetime.utcnow().isoformat(),
            "features": [
                "webhook_receiver",
                "suggestion_enhancement",
                "context_injection",
                "feedback_loop",
                "analytics"
            ]
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")


# Background Task Functions
async def _process_webhook_background(
    webhook_type: str,
    payload: Dict[str, Any],
    user_id: Optional[str]
):
    """Process webhook in background task."""
    try:
        result = await copilot_service.receive_webhook(webhook_type, payload, user_id)
        logger.info(f"Background webhook processing completed: {result}")
    except Exception as e:
        logger.error(f"Background webhook processing failed: {e}")


async def _process_feedback_background(
    suggestion_id: str,
    feedback_type: str,
    feedback_data: Dict[str, Any],
    user_id: Optional[str]
):
    """Process feedback in background task."""
    try:
        # Add user context to feedback data
        if user_id:
            feedback_data["user_id"] = user_id
        
        result = await copilot_service.create_feedback_loop(
            suggestion_id, feedback_type, feedback_data
        )
        logger.info(f"Background feedback processing completed: {result}")
    except Exception as e:
        logger.error(f"Background feedback processing failed: {e}")