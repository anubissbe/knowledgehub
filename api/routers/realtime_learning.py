"""
Real-time Learning Pipeline API Router
Exposes endpoints for real-time event streaming and learning
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional
import json
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field

from ..services.realtime_learning_pipeline import (
    RealtimeLearningPipeline,
    StreamEvent,
    EventType,
    get_learning_pipeline
)

router = APIRouter(prefix="/api/realtime", tags=["realtime-learning"])


class EventRequest(BaseModel):
    """Request model for publishing events"""
    event_type: EventType
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CodeChangeEvent(BaseModel):
    """Specific model for code change events"""
    file_path: str
    changes: List[Dict[str, Any]]
    language: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class DecisionEvent(BaseModel):
    """Model for decision events"""
    decision: str
    category: str
    context: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    session_id: Optional[str] = None


class ErrorEvent(BaseModel):
    """Model for error events"""
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None


@router.get("/health")
async def health_check():
    """Check real-time pipeline health"""
    try:
        pipeline = await get_learning_pipeline()
        # Test Redis connection
        await pipeline.redis_client.ping()
        
        return {
            "status": "healthy",
            "service": "realtime-learning",
            "redis_connected": True,
            "consumer_name": pipeline.consumer_name
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "realtime-learning",
            "error": str(e)
        }


@router.post("/events")
async def publish_event(
    event: EventRequest,
    pipeline: RealtimeLearningPipeline = Depends(get_learning_pipeline)
) -> Dict[str, Any]:
    """Publish an event to the real-time pipeline"""
    try:
        stream_event = StreamEvent(
            event_type=event.event_type,
            session_id=event.session_id,
            user_id=event.user_id,
            data=event.data,
            metadata=event.metadata
        )
        
        stream_id = await pipeline.publish_event(stream_event)
        
        return {
            "success": True,
            "event_id": stream_event.event_id,
            "stream_id": stream_id,
            "timestamp": stream_event.timestamp.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events/code-change")
async def publish_code_change(
    event: CodeChangeEvent,
    pipeline: RealtimeLearningPipeline = Depends(get_learning_pipeline)
) -> Dict[str, Any]:
    """Publish a code change event"""
    try:
        stream_event = StreamEvent(
            event_type=EventType.CODE_CHANGE,
            session_id=event.session_id,
            user_id=event.user_id,
            data={
                "file_path": event.file_path,
                "changes": event.changes,
                "language": event.language,
                "timestamp": datetime.utcnow().isoformat()
            },
            metadata={"source": "api"}
        )
        
        stream_id = await pipeline.publish_event(stream_event)
        
        return {
            "success": True,
            "event_id": stream_event.event_id,
            "stream_id": stream_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events/decision")
async def publish_decision(
    event: DecisionEvent,
    pipeline: RealtimeLearningPipeline = Depends(get_learning_pipeline)
) -> Dict[str, Any]:
    """Publish a decision event"""
    try:
        stream_event = StreamEvent(
            event_type=EventType.DECISION_MADE,
            session_id=event.session_id,
            data={
                "decision": event.decision,
                "category": event.category,
                "context": event.context,
                "confidence": event.confidence
            },
            metadata={"source": "api"}
        )
        
        stream_id = await pipeline.publish_event(stream_event)
        
        return {
            "success": True,
            "event_id": stream_event.event_id,
            "stream_id": stream_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events/error")
async def publish_error(
    event: ErrorEvent,
    pipeline: RealtimeLearningPipeline = Depends(get_learning_pipeline)
) -> Dict[str, Any]:
    """Publish an error event for learning"""
    try:
        stream_event = StreamEvent(
            event_type=EventType.ERROR_OCCURRED,
            session_id=event.session_id,
            data={
                "error_type": event.error_type,
                "error_message": event.error_message,
                "stack_trace": event.stack_trace,
                "context": event.context
            },
            metadata={"source": "api", "severity": "high"}
        )
        
        stream_id = await pipeline.publish_event(stream_event)
        
        return {
            "success": True,
            "event_id": stream_event.event_id,
            "stream_id": stream_id,
            "learning": "Error pattern will be analyzed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context/{session_id}")
async def get_session_context(
    session_id: str,
    pipeline: RealtimeLearningPipeline = Depends(get_learning_pipeline)
) -> Dict[str, Any]:
    """Get real-time context for a session"""
    try:
        context = await pipeline.get_real_time_context(session_id)
        return context
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context")
async def get_global_context(
    pipeline: RealtimeLearningPipeline = Depends(get_learning_pipeline)
) -> Dict[str, Any]:
    """Get global real-time context"""
    try:
        context = await pipeline.get_real_time_context()
        return context
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def event_stream_generator(
    pipeline: RealtimeLearningPipeline,
    event_types: Optional[List[EventType]] = None
):
    """Generate Server-Sent Events from the pipeline"""
    try:
        async for event in pipeline.consume_events():
            # Filter by event types if specified
            if event_types and event.event_type not in event_types:
                continue
                
            # Format as SSE
            data = {
                "id": event.event_id,
                "event": event.event_type.value,
                "data": {
                    "timestamp": event.timestamp.isoformat(),
                    "session_id": event.session_id,
                    "user_id": event.user_id,
                    "payload": event.data,
                    "metadata": event.metadata
                }
            }
            
            yield f"data: {json.dumps(data)}\n\n"
            
    except asyncio.CancelledError:
        # Clean shutdown
        pass
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@router.get("/stream")
async def stream_events(
    event_types: Optional[str] = None,
    pipeline: RealtimeLearningPipeline = Depends(get_learning_pipeline)
):
    """Stream real-time events using Server-Sent Events"""
    # Parse event types
    types = None
    if event_types:
        types = [EventType(t.strip()) for t in event_types.split(",")]
    
    return StreamingResponse(
        event_stream_generator(pipeline, types),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/process/start")
async def start_processing(
    background_tasks: BackgroundTasks,
    pipeline: RealtimeLearningPipeline = Depends(get_learning_pipeline)
) -> Dict[str, Any]:
    """Start background event processing"""
    try:
        background_tasks.add_task(pipeline.process_event_stream)
        
        return {
            "success": True,
            "message": "Background processing started",
            "consumer": pipeline.consumer_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process/stop")
async def stop_processing(
    pipeline: RealtimeLearningPipeline = Depends(get_learning_pipeline)
) -> Dict[str, Any]:
    """Stop background event processing"""
    try:
        await pipeline.stop_processing()
        
        return {
            "success": True,
            "message": "Processing stopped"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_pipeline_stats(
    pipeline: RealtimeLearningPipeline = Depends(get_learning_pipeline)
) -> Dict[str, Any]:
    """Get pipeline statistics"""
    try:
        # Get stream lengths
        stats = {}
        for name, stream in pipeline.streams.items():
            length = await pipeline.redis_client.xlen(stream)
            stats[name] = {
                "stream": stream,
                "length": length
            }
            
        # Get consumer group info
        groups = await pipeline.redis_client.xinfo_groups(pipeline.streams["events"])
        
        return {
            "streams": stats,
            "consumer_groups": groups,
            "consumer_name": pipeline.consumer_name,
            "processing": pipeline.processing
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))