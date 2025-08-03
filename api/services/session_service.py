"""
Advanced AI Session Management Service Implementation.

This service provides intelligent session management with state preservation,
context windows, handoff mechanisms, and cross-session continuity.
"""

import logging
import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import text, and_, or_, desc, asc, func
from sqlalchemy.exc import IntegrityError

from ..models.session import (
    Session, SessionHandoff, SessionCheckpoint, SessionMetrics,
    SessionState, SessionType, HandoffReason,
    SessionCreate, SessionUpdate, SessionHandoffCreate, SessionCheckpointCreate,
    SessionResponse, SessionContextResponse, SessionAnalytics, SessionRecoveryInfo
)
from ..models.base import get_db_context
from ..services.cache import redis_client
from ..services.time_series_analytics import TimeSeriesAnalyticsService, MetricType
from ..services.memory_service import memory_service
from shared.config import Config
from shared.logging import setup_logging

logger = setup_logging("session_service")


@dataclass
class SessionStateSnapshot:
    """Complete session state for serialization/recovery."""
    session_id: str
    user_id: str
    project_id: Optional[str]
    context_window: List[str]
    context_summary: Optional[str]
    active_tasks: List[Dict[str, Any]]
    task_queue: List[Dict[str, Any]]
    session_variables: Dict[str, Any]
    preferences: Dict[str, Any]
    interaction_count: int
    success_rate: float
    timestamp: datetime


@dataclass
class HandoffContext:
    """Context data for session handoffs."""
    source_session: SessionStateSnapshot
    handoff_reason: HandoffReason
    continuation_data: Dict[str, Any]
    priority_tasks: List[Dict[str, Any]]
    learned_patterns: List[Dict[str, Any]]
    error_context: Optional[Dict[str, Any]]


class SessionService:
    """
    Advanced AI Session Management Service with intelligent state management.
    
    Features:
    - Automatic session state preservation
    - Context window management with intelligent sizing
    - Session handoff with context transfer
    - Cross-session linking and continuity
    - Performance monitoring and optimization
    - Automatic recovery from failures
    - Session analytics and insights
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.analytics_service = TimeSeriesAnalyticsService()
        self.memory_service = memory_service
        self._initialized = False
        
        # Configuration
        self.default_context_size = 100
        self.max_context_size = 500
        self.auto_checkpoint_interval = 50  # Every 50 interactions
        self.session_timeout_hours = 24
        self.handoff_timeout_hours = 2
        
        logger.info("Initialized SessionService")
    
    async def initialize(self):
        """Initialize the session service."""
        if self._initialized:
            return
        
        try:
            # Initialize dependencies
            await self.analytics_service.initialize()
            await self.memory_service.initialize()
            await redis_client.initialize()
            
            self._initialized = True
            logger.info("SessionService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SessionService: {e}")
            raise
    
    async def create_session(
        self,
        session_data: SessionCreate,
        auto_restore: bool = True
    ) -> SessionResponse:
        """
        Create a new session with intelligent initialization.
        
        Args:
            session_data: Session creation data
            auto_restore: Whether to restore context from related sessions
            
        Returns:
            Created session response
        """
        start_time = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        try:
            with get_db_context() as db:
                # Check for existing active sessions
                existing_session = db.query(Session).filter(
                    Session.user_id == session_data.user_id,
                    Session.project_id == session_data.project_id,
                    Session.state == SessionState.ACTIVE.value
                ).first()
                
                if existing_session:
                    logger.info(f"Found existing active session: {existing_session.id}")
                    # Update activity and return existing session
                    existing_session.update_activity()
                    db.commit()
                    return self._session_to_response(existing_session)
                
                # Create new session
                session = Session(
                    user_id=session_data.user_id,
                    project_id=session_data.project_id,
                    session_type=session_data.session_type.value,
                    title=session_data.title,
                    description=session_data.description,
                    preferences=session_data.preferences,
                    max_context_size=session_data.max_context_size,
                    parent_session_id=uuid.UUID(session_data.parent_session_id) if session_data.parent_session_id else None
                )
                
                db.add(session)
                db.flush()  # Get the ID
                
                # Auto-restore context if requested
                if auto_restore:
                    await self._restore_session_context(db, session)
                
                # Create initial checkpoint
                await self._create_initial_checkpoint(db, session)
                
                # Link to parent session if specified
                if session.parent_session_id:
                    await self._link_to_parent_session(db, session)
                
                db.commit()
                
                # Record analytics
                processing_time = (time.time() - start_time) * 1000
                await self._record_session_analytics(
                    session_type=session_data.session_type,
                    processing_time=processing_time,
                    user_id=session_data.user_id,
                    project_id=session_data.project_id
                )
                
                logger.info(f"Created session: {session.id} for user {session.user_id}")
                return self._session_to_response(session)
                
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def update_session(
        self,
        session_id: str,
        update_data: SessionUpdate
    ) -> SessionResponse:
        """Update session with new data."""
        
        try:
            with get_db_context() as db:
                session = db.query(Session).filter_by(id=session_id).first()
                if not session:
                    raise ValueError(f"Session not found: {session_id}")
                
                # Update fields
                if update_data.title is not None:
                    session.title = update_data.title
                if update_data.description is not None:
                    session.description = update_data.description
                if update_data.state is not None:
                    old_state = session.state
                    session.state = update_data.state.value
                    
                    # Handle state transitions
                    if old_state != session.state:
                        await self._handle_state_transition(db, session, old_state, session.state)
                
                if update_data.preferences is not None:
                    session.preferences = {**(session.preferences or {}), **update_data.preferences}
                if update_data.max_context_size is not None:
                    session.max_context_size = update_data.max_context_size
                if update_data.user_satisfaction is not None:
                    session.user_satisfaction = update_data.user_satisfaction
                if update_data.completion_status is not None:
                    session.completion_status = update_data.completion_status
                
                session.update_activity()
                db.commit()
                
                logger.info(f"Updated session: {session_id}")
                return self._session_to_response(session)
                
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[SessionResponse]:
        """Get session by ID."""
        
        try:
            with get_db_context() as db:
                session = db.query(Session).filter_by(id=session_id).first()
                if not session:
                    return None
                
                # Update activity tracking
                session.update_activity()
                db.commit()
                
                return self._session_to_response(session)
                
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            raise
    
    async def add_to_context(
        self,
        session_id: str,
        memory_id: str,
        auto_optimize: bool = True
    ) -> SessionContextResponse:
        """Add memory to session context window."""
        
        try:
            with get_db_context() as db:
                session = db.query(Session).filter_by(id=session_id).first()
                if not session:
                    raise ValueError(f"Session not found: {session_id}")
                
                # Add to context window
                session.add_to_context(memory_id)
                
                # Auto-optimize context size if requested
                if auto_optimize:
                    await self._optimize_context_window(db, session)
                
                # Update context summary
                await self._update_context_summary(db, session)
                
                db.commit()
                
                logger.debug(f"Added memory {memory_id} to session {session_id} context")
                return self._session_context_to_response(session)
                
        except Exception as e:
            logger.error(f"Failed to add to context: {e}")
            raise
    
    async def create_handoff(
        self,
        handoff_data: SessionHandoffCreate
    ) -> str:
        """Create a session handoff for context transfer."""
        
        try:
            with get_db_context() as db:
                # Get source session
                source_session = db.query(Session).filter_by(
                    id=handoff_data.source_session_id
                ).first()
                
                if not source_session:
                    raise ValueError(f"Source session not found: {handoff_data.source_session_id}")
                
                # Create state snapshot
                state_snapshot = await self._create_state_snapshot(source_session)
                
                # Create handoff record
                handoff = SessionHandoff(
                    source_session_id=uuid.UUID(handoff_data.source_session_id),
                    target_session_id=uuid.UUID(handoff_data.target_session_id) if handoff_data.target_session_id else None,
                    reason=handoff_data.reason.value,
                    handoff_message=handoff_data.handoff_message,
                    continuation_instructions=handoff_data.continuation_instructions,
                    context_data={
                        "state_snapshot": self._serialize_state_snapshot(state_snapshot),
                        "custom_context": handoff_data.context_data
                    }
                )
                
                db.add(handoff)
                db.flush()
                
                # Update source session state
                source_session.state = SessionState.TRANSFERRED.value
                source_session.ended_at = datetime.utcnow()
                
                # Create final checkpoint
                await self._create_handoff_checkpoint(db, source_session, handoff.id)
                
                db.commit()
                
                logger.info(f"Created handoff: {handoff.id} from session {handoff_data.source_session_id}")
                return str(handoff.id)
                
        except Exception as e:
            logger.error(f"Failed to create handoff: {e}")
            raise
    
    async def restore_from_handoff(
        self,
        handoff_id: str,
        new_session_data: SessionCreate
    ) -> SessionResponse:
        """Create new session from handoff context."""
        
        try:
            with get_db_context() as db:
                # Get handoff record
                handoff = db.query(SessionHandoff).filter_by(id=handoff_id).first()
                if not handoff:
                    raise ValueError(f"Handoff not found: {handoff_id}")
                
                # Extract state snapshot
                context_data = handoff.context_data or {}
                state_snapshot_data = context_data.get("state_snapshot", {})
                
                # Create new session
                new_session = await self.create_session(new_session_data, auto_restore=False)
                
                # Restore context from handoff
                with get_db_context() as restore_db:
                    session = restore_db.query(Session).filter_by(id=new_session.id).first()
                    
                    # Restore context window
                    if state_snapshot_data.get("context_window"):
                        session.context_window = state_snapshot_data["context_window"]
                        session.context_size = len(session.context_window)
                    
                    # Restore session variables
                    if state_snapshot_data.get("session_variables"):
                        session.session_variables = state_snapshot_data["session_variables"]
                    
                    # Restore active tasks
                    if state_snapshot_data.get("active_tasks"):
                        session.active_tasks = state_snapshot_data["active_tasks"]
                    
                    # Restore task queue
                    if state_snapshot_data.get("task_queue"):
                        session.task_queue = state_snapshot_data["task_queue"]
                    
                    # Update handoff record
                    handoff.target_session_id = uuid.UUID(new_session.id)
                    handoff.status = "completed"
                    handoff.completed_at = datetime.utcnow()
                    
                    restore_db.commit()
                
                logger.info(f"Restored session {new_session.id} from handoff {handoff_id}")
                return new_session
                
        except Exception as e:
            logger.error(f"Failed to restore from handoff: {e}")
            raise
    
    async def create_checkpoint(
        self,
        checkpoint_data: SessionCheckpointCreate
    ) -> str:
        """Create a manual session checkpoint."""
        
        try:
            with get_db_context() as db:
                session = db.query(Session).filter_by(id=checkpoint_data.session_id).first()
                if not session:
                    raise ValueError(f"Session not found: {checkpoint_data.session_id}")
                
                # Create state snapshot
                state_snapshot = await self._create_state_snapshot(session)
                
                # Create checkpoint
                checkpoint = SessionCheckpoint(
                    session_id=uuid.UUID(checkpoint_data.session_id),
                    checkpoint_name=checkpoint_data.checkpoint_name,
                    description=checkpoint_data.description,
                    checkpoint_type=checkpoint_data.checkpoint_type,
                    session_state=self._serialize_state_snapshot(state_snapshot),
                    context_snapshot={
                        "context_window": session.context_window,
                        "context_summary": session.context_summary
                    },
                    memory_ids=[],  # Skip memory_ids for now since they're strings not UUIDs
                    variables_snapshot=session.session_variables,
                    interaction_count=session.interaction_count,
                    created_by=session.user_id,
                    is_recovery_point=checkpoint_data.is_recovery_point,
                    recovery_priority=checkpoint_data.recovery_priority
                )
                
                db.add(checkpoint)
                db.flush()
                
                # Update session checkpoint reference
                session.last_checkpoint = datetime.utcnow()
                session.checkpoint_data = {
                    "latest_checkpoint_id": str(checkpoint.id),
                    "checkpoint_count": (session.checkpoint_data or {}).get("checkpoint_count", 0) + 1
                }
                
                db.commit()
                
                logger.info(f"Created checkpoint: {checkpoint.id} for session {checkpoint_data.session_id}")
                return str(checkpoint.id)
                
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    async def get_session_analytics(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        time_window_hours: int = 24
    ) -> SessionAnalytics:
        """Get comprehensive session analytics."""
        
        try:
            with get_db_context() as db:
                # Base query
                base_query = db.query(Session).filter(Session.user_id == user_id)
                
                if project_id:
                    base_query = base_query.filter(Session.project_id == project_id)
                
                # Time window filter
                time_threshold = datetime.utcnow() - timedelta(hours=time_window_hours)
                recent_query = base_query.filter(Session.started_at >= time_threshold)
                
                # Calculate metrics
                total_sessions = base_query.count()
                active_sessions = base_query.filter(Session.state == SessionState.ACTIVE.value).count()
                
                # Average metrics
                avg_duration = db.query(func.avg(Session.total_duration)).filter(
                    Session.user_id == user_id,
                    Session.total_duration > 0
                ).scalar() or 0.0
                
                avg_success_rate = db.query(func.avg(Session.success_rate)).filter(
                    Session.user_id == user_id
                ).scalar() or 0.0
                
                # User satisfaction
                user_satisfaction_avg = db.query(func.avg(Session.user_satisfaction)).filter(
                    Session.user_id == user_id,
                    Session.user_satisfaction.isnot(None)
                ).scalar()
                
                # Sessions by type
                type_counts = db.query(
                    Session.session_type,
                    func.count(Session.id)
                ).filter(Session.user_id == user_id).group_by(Session.session_type).all()
                
                sessions_by_type = {session_type: count for session_type, count in type_counts}
                
                # Sessions by state
                state_counts = db.query(
                    Session.state,
                    func.count(Session.id)
                ).filter(Session.user_id == user_id).group_by(Session.state).all()
                
                sessions_by_state = {state: count for state, count in state_counts}
                
                # Top projects
                project_counts = db.query(
                    Session.project_id,
                    func.count(Session.id),
                    func.avg(Session.success_rate)
                ).filter(
                    Session.user_id == user_id,
                    Session.project_id.isnot(None)
                ).group_by(Session.project_id).order_by(
                    func.count(Session.id).desc()
                ).limit(10).all()
                
                top_projects = [
                    {
                        "project_id": project_id,
                        "session_count": count,
                        "avg_success_rate": float(avg_rate or 0.0)
                    }
                    for project_id, count, avg_rate in project_counts
                ]
                
                # Session trends (last 7 days)
                trends = db.query(
                    func.date_trunc('day', Session.started_at).label('date'),
                    func.count(Session.id).label('count'),
                    func.avg(Session.success_rate).label('avg_success_rate')
                ).filter(
                    Session.user_id == user_id,
                    Session.started_at >= datetime.utcnow() - timedelta(days=7)
                ).group_by(
                    func.date_trunc('day', Session.started_at)
                ).order_by('date').all()
                
                session_trends = [
                    {
                        "date": trend.date.isoformat(),
                        "session_count": trend.count,
                        "avg_success_rate": float(trend.avg_success_rate or 0.0)
                    }
                    for trend in trends
                ]
                
                # Handoff statistics
                handoff_stats = db.query(
                    func.count(SessionHandoff.id).label('total_handoffs'),
                    func.avg(SessionHandoff.success_rate).label('avg_success_rate')
                ).join(Session, SessionHandoff.source_session_id == Session.id).filter(
                    Session.user_id == user_id
                ).first()
                
                handoff_statistics = {
                    "total_handoffs": handoff_stats.total_handoffs or 0,
                    "success_rate": float(handoff_stats.avg_success_rate or 0.0)
                }
                
                return SessionAnalytics(
                    total_sessions=total_sessions,
                    active_sessions=active_sessions,
                    avg_session_duration=float(avg_duration),
                    avg_success_rate=float(avg_success_rate),
                    sessions_by_type=sessions_by_type,
                    sessions_by_state=sessions_by_state,
                    user_satisfaction_avg=float(user_satisfaction_avg) if user_satisfaction_avg else None,
                    top_projects=top_projects,
                    session_trends=session_trends,
                    handoff_statistics=handoff_statistics
                )
                
        except Exception as e:
            logger.error(f"Failed to get session analytics: {e}")
            raise
    
    async def get_recovery_info(self, session_id: str) -> SessionRecoveryInfo:
        """Get session recovery information."""
        
        try:
            with get_db_context() as db:
                session = db.query(Session).filter_by(id=session_id).first()
                if not session:
                    raise ValueError(f"Session not found: {session_id}")
                
                # Find recovery checkpoints
                recovery_checkpoints = db.query(SessionCheckpoint).filter(
                    SessionCheckpoint.session_id == session.id,
                    SessionCheckpoint.is_recovery_point == True
                ).order_by(desc(SessionCheckpoint.created_at)).limit(5).all()
                
                # Determine if session is recoverable
                recoverable = len(recovery_checkpoints) > 0 or session.recovery_data is not None
                
                # Latest checkpoint
                last_checkpoint = recovery_checkpoints[0].created_at if recovery_checkpoints else None
                
                # Recovery options
                recovery_options = []
                for checkpoint in recovery_checkpoints:
                    recovery_options.append({
                        "checkpoint_id": str(checkpoint.id),
                        "name": checkpoint.checkpoint_name,
                        "created_at": checkpoint.created_at.isoformat(),
                        "interaction_count": checkpoint.interaction_count,
                        "recovery_priority": checkpoint.recovery_priority
                    })
                
                # Estimate data loss
                if last_checkpoint:
                    # Ensure both datetimes are timezone-aware
                    now = datetime.utcnow().replace(tzinfo=last_checkpoint.tzinfo) if last_checkpoint.tzinfo else datetime.utcnow()
                    checkpoint_time = last_checkpoint.replace(tzinfo=None) if last_checkpoint.tzinfo else last_checkpoint
                    hours_since_checkpoint = (now.replace(tzinfo=None) - checkpoint_time).total_seconds() / 3600
                    if hours_since_checkpoint < 1:
                        estimated_data_loss = "Minimal (< 1 hour)"
                    elif hours_since_checkpoint < 6:
                        estimated_data_loss = f"Low (~{int(hours_since_checkpoint)} hours)"
                    else:
                        estimated_data_loss = f"Moderate ({int(hours_since_checkpoint)} hours)"
                else:
                    estimated_data_loss = "High (no checkpoints)"
                
                # Recommended action
                if not recoverable:
                    recommended_action = "Create new session"
                elif last_checkpoint and (datetime.utcnow() - last_checkpoint).total_seconds() < 3600:
                    recommended_action = "Restore from latest checkpoint"
                else:
                    recommended_action = "Review recovery options"
                
                return SessionRecoveryInfo(
                    session_id=session_id,
                    recoverable=recoverable,
                    last_checkpoint=last_checkpoint,
                    recovery_options=recovery_options,
                    estimated_data_loss=estimated_data_loss,
                    recommended_action=recommended_action
                )
                
        except Exception as e:
            logger.error(f"Failed to get recovery info: {e}")
            raise
    
    # Helper methods
    
    async def _restore_session_context(self, db: DBSession, session: Session):
        """Restore context from related sessions."""
        
        try:
            # Find recent sessions for the same user/project
            related_sessions = db.query(Session).filter(
                Session.user_id == session.user_id,
                Session.project_id == session.project_id,
                Session.id != session.id,
                Session.state.in_([SessionState.COMPLETED.value, SessionState.TRANSFERRED.value])
            ).order_by(desc(Session.ended_at)).limit(3).all()
            
            # Restore context from most recent session
            if related_sessions:
                latest_session = related_sessions[0]
                
                # Copy relevant context
                if latest_session.context_window:
                    # Take the most recent items from context
                    context_to_restore = latest_session.context_window[-20:]  # Last 20 items
                    session.context_window = context_to_restore
                    session.context_size = len(context_to_restore)
                
                # Copy preferences
                if latest_session.preferences:
                    session.preferences = {
                        **(session.preferences or {}),
                        **latest_session.preferences
                    }
                
                # Link sessions
                session.related_sessions = [str(s.id) for s in related_sessions[:5]]
                session.session_chain = [str(latest_session.id)]
                
                logger.info(f"Restored context from session {latest_session.id}")
                
        except Exception as e:
            logger.warning(f"Context restoration failed: {e}")
    
    async def _create_initial_checkpoint(self, db: DBSession, session: Session):
        """Create initial checkpoint for new session."""
        
        try:
            state_snapshot = await self._create_state_snapshot(session)
            
            checkpoint = SessionCheckpoint(
                session_id=session.id,
                checkpoint_name="Session Initialized",
                description="Initial session setup checkpoint",
                checkpoint_type="auto",
                session_state=self._serialize_state_snapshot(state_snapshot),
                context_snapshot={"initialized": True},
                variables_snapshot={},
                interaction_count=0,
                created_by="system",
                is_recovery_point=True,
                recovery_priority=10
            )
            
            db.add(checkpoint)
            session.last_checkpoint = datetime.utcnow()
            
        except Exception as e:
            logger.warning(f"Initial checkpoint creation failed: {e}")
    
    async def _create_state_snapshot(self, session: Session) -> SessionStateSnapshot:
        """Create a complete state snapshot of the session."""
        
        return SessionStateSnapshot(
            session_id=str(session.id),
            user_id=session.user_id,
            project_id=session.project_id,
            context_window=session.context_window or [],
            context_summary=session.context_summary,
            active_tasks=session.active_tasks or [],
            task_queue=session.task_queue or [],
            session_variables=session.session_variables or {},
            preferences=session.preferences or {},
            interaction_count=session.interaction_count,
            success_rate=session.success_rate,
            timestamp=datetime.utcnow()
        )
    
    def _serialize_state_snapshot(self, snapshot: SessionStateSnapshot) -> Dict[str, Any]:
        """Serialize state snapshot for JSON storage."""
        return {
            "session_id": snapshot.session_id,
            "user_id": snapshot.user_id,
            "project_id": snapshot.project_id,
            "context_window": snapshot.context_window,
            "context_summary": snapshot.context_summary,
            "active_tasks": snapshot.active_tasks,
            "task_queue": snapshot.task_queue,
            "session_variables": snapshot.session_variables,
            "preferences": snapshot.preferences,
            "interaction_count": snapshot.interaction_count,
            "success_rate": snapshot.success_rate,
            "timestamp": snapshot.timestamp.isoformat() if snapshot.timestamp else None
        }
    
    async def _optimize_context_window(self, db: DBSession, session: Session):
        """Optimize context window size based on performance."""
        
        try:
            # Calculate optimal size using PostgreSQL function
            result = db.execute(text("""
                SELECT calculate_optimal_context_size(
                    :session_id, :current_size, :max_size, :avg_response_time
                )
            """), {
                "session_id": session.id,
                "current_size": session.context_size,
                "max_size": session.max_context_size,
                "avg_response_time": session.avg_response_time
            })
            
            optimal_size = result.scalar()
            
            if optimal_size and optimal_size != session.context_size:
                # Adjust context window
                if optimal_size < session.context_size:
                    # Trim context window
                    session.context_window = session.context_window[-optimal_size:]
                    session.context_size = optimal_size
                    logger.debug(f"Reduced context window to {optimal_size}")
                
        except Exception as e:
            logger.warning(f"Context optimization failed: {e}")
    
    async def _update_context_summary(self, db: DBSession, session: Session):
        """Update context summary with recent memories."""
        
        try:
            if not session.context_window:
                return
            
            # Get recent memory contents for summarization
            memory_contents = []
            for memory_id in session.context_window[-10:]:  # Last 10 memories
                # This would integrate with memory service to get content
                # For now, create a simple summary
                memory_contents.append(f"Memory {memory_id}")
            
            # Create simple summary
            if memory_contents:
                session.context_summary = f"Context includes {len(memory_contents)} recent memories"
            
        except Exception as e:
            logger.warning(f"Context summary update failed: {e}")
    
    async def _create_handoff_checkpoint(self, db: DBSession, session: Session, handoff_id: uuid.UUID):
        """Create a final checkpoint before session handoff."""
        
        try:
            state_snapshot = await self._create_state_snapshot(session)
            
            checkpoint = SessionCheckpoint(
                session_id=session.id,
                checkpoint_name="Session Handoff",
                description=f"Final checkpoint before handoff {handoff_id}",
                checkpoint_type="auto",
                session_state=self._serialize_state_snapshot(state_snapshot),
                context_snapshot={
                    "handoff_id": str(handoff_id),
                    "final_state": True
                },
                memory_ids=[],
                variables_snapshot=session.session_variables or {},
                interaction_count=session.interaction_count,
                created_by="system",
                is_recovery_point=True,
                recovery_priority=8
            )
            
            db.add(checkpoint)
            session.last_checkpoint = datetime.utcnow()
            
            logger.info(f"Created handoff checkpoint for session {session.id}")
            
        except Exception as e:
            logger.warning(f"Handoff checkpoint creation failed: {e}")
    
    async def _link_to_parent_session(self, db: DBSession, session: Session):
        """Link session to its parent session."""
        
        try:
            if not session.parent_session_id:
                return
            
            # Update parent session's related sessions
            parent = db.query(Session).filter_by(id=session.parent_session_id).first()
            if parent:
                related_sessions = parent.related_sessions or []
                if str(session.id) not in related_sessions:
                    related_sessions.append(str(session.id))
                    parent.related_sessions = related_sessions
                
                # Update session chain
                session_chain = parent.session_chain or []
                session_chain.append(str(session.id))
                session.session_chain = session_chain
                
                logger.info(f"Linked session {session.id} to parent {parent.id}")
            
        except Exception as e:
            logger.warning(f"Parent session linking failed: {e}")
    
    async def _handle_state_transition(self, db: DBSession, session: Session, old_state: str, new_state: str):
        """Handle session state transitions."""
        
        try:
            # Set end time for completed/transferred states
            if new_state in [SessionState.COMPLETED.value, SessionState.TRANSFERRED.value, SessionState.ABANDONED.value]:
                session.ended_at = datetime.utcnow()
                session.total_duration = int((session.ended_at - session.started_at).total_seconds())
            
            # Create transition checkpoint for important state changes
            if old_state == SessionState.ACTIVE.value and new_state != SessionState.PAUSED.value:
                state_snapshot = await self._create_state_snapshot(session)
                
                checkpoint = SessionCheckpoint(
                    session_id=session.id,
                    checkpoint_name=f"State Transition: {old_state} → {new_state}",
                    description=f"Automatic checkpoint for state transition",
                    checkpoint_type="auto",
                    session_state=self._serialize_state_snapshot(state_snapshot),
                    context_snapshot={"transition": {"from": old_state, "to": new_state}},
                    memory_ids=[],
                    variables_snapshot=session.session_variables or {},
                    interaction_count=session.interaction_count,
                    created_by="system",
                    is_recovery_point=True,
                    recovery_priority=6
                )
                
                db.add(checkpoint)
                session.last_checkpoint = datetime.utcnow()
            
            logger.info(f"Handled state transition: {old_state} → {new_state} for session {session.id}")
            
        except Exception as e:
            logger.warning(f"State transition handling failed: {e}")
    
    async def _record_session_analytics(
        self,
        session_type: SessionType,
        processing_time: float,
        user_id: str,
        project_id: Optional[str]
    ):
        """Record session operation analytics."""
        
        try:
            # Record in time-series analytics
            await self.analytics_service.record_metric(
                metric_type=MetricType.SESSION_CREATION,
                value=1.0,
                tags={
                    "session_type": session_type.value,
                    "user_id": user_id,
                    "project_id": project_id or "unknown"
                },
                metadata={"processing_time_ms": processing_time}
            )
            
        except Exception as e:
            logger.warning(f"Analytics recording failed: {e}")
    
    def _session_to_response(self, session: Session) -> SessionResponse:
        """Convert Session model to SessionResponse."""
        
        return SessionResponse(
            id=str(session.id),
            user_id=session.user_id,
            project_id=session.project_id,
            session_type=SessionType(session.session_type),
            state=SessionState(session.state),
            title=session.title,
            description=session.description,
            context_size=session.context_size,
            max_context_size=session.max_context_size,
            interaction_count=session.interaction_count,
            error_count=session.error_count,
            success_rate=session.success_rate,
            started_at=session.started_at,
            last_active=session.last_active,
            ended_at=session.ended_at,
            total_duration=session.total_duration,
            parent_session_id=str(session.parent_session_id) if session.parent_session_id else None,
            completion_status=session.completion_status,
            user_satisfaction=session.user_satisfaction
        )
    
    def _session_context_to_response(self, session: Session) -> SessionContextResponse:
        """Convert session context to response."""
        
        return SessionContextResponse(
            session_id=str(session.id),
            context_window=session.context_window or [],
            context_summary=session.context_summary,
            context_size=session.context_size,
            max_context_size=session.max_context_size,
            last_updated=session.last_active
        )
    
    async def cleanup(self):
        """Clean up service resources."""
        await self.analytics_service.cleanup()
        self._initialized = False
        logger.info("SessionService cleaned up")


# Global session service instance
session_service = SessionService()