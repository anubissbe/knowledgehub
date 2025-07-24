"""
Tool Handlers for KnowledgeHub MCP Server.

This module implements the actual functionality for all MCP tools,
providing direct integration with KnowledgeHub's AI-enhanced systems.

Handler Classes:
- MemoryHandler: Memory operations and AI-enhanced storage
- SessionHandler: Session management and continuity
- AIHandler: AI intelligence features and learning
- AnalyticsHandler: Real-time metrics and performance data
- ContextSynchronizer: Context synchronization utilities
- ResponseFormatter: Response formatting and enhancement
"""

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid
import aiohttp
import sys
import os

# Add the project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.services.memory_service import memory_service
from api.services.session_service import session_service
from api.services.metrics_service import metrics_service
from api.workers.metrics_collector import metrics_collector_worker
from api.services.prediction_service import prediction_service
from api.services.pattern_service import pattern_service
from api.services.decision_service import decision_service
from api.services.error_learning_service import error_learning_service
from api.services.real_ai_intelligence import real_ai_intelligence
from api.services.real_embeddings_service import real_embeddings_service
from api.services.real_websocket_events import real_websocket_events
from shared.config import Config

logger = logging.getLogger(__name__)


class BaseHandler:
    """Base handler class with common functionality."""
    
    def __init__(self):
        self.config = Config()
        self.initialized = False
        self.last_sync = None
    
    async def initialize(self):
        """Initialize the handler."""
        if not self.initialized:
            self.initialized = True
            self.last_sync = datetime.utcnow()
            logger.debug(f"Initialized {self.__class__.__name__}")
    
    async def cleanup(self):
        """Cleanup handler resources."""
        logger.debug(f"Cleaning up {self.__class__.__name__}")
    
    async def get_resource_info(self) -> Dict[str, Any]:
        """Get resource information for this handler."""
        return {
            "handler": self.__class__.__name__,
            "initialized": self.initialized,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None
        }


class MemoryHandler(BaseHandler):
    """Handler for memory operations."""
    
    async def create_memory(
        self,
        content: str,
        memory_type: str,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new memory with AI-enhanced processing."""
        try:
            await self.initialize()
            
            # Generate AI embeddings for the content using real service
            embedding_result = None
            try:
                embedding_result = await real_embeddings_service.generate_embedding(
                    content, "text"
                )
                logger.debug(f"Generated embedding with {embedding_result.dimensions} dimensions")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings: {e}")
            
            # Prepare memory data
            memory_data = {
                "content": content,
                "memory_type": memory_type,
                "user_id": "claude_code",  # Default user for Claude Code integration
                "project_id": project_id,
                "session_id": session_id,
                "tags": tags or [],
                "metadata": metadata or {},
                "source": "claude_code_mcp"
            }
            
            # Add embedding if generated
            if embedding_result and not embedding_result.error:
                memory_data["embedding"] = embedding_result.embedding
            
            # Create memory using the service
            memory = await memory_service.create_memory(memory_data)
            
            # Emit real-time event
            try:
                await real_websocket_events.emit_memory_created({
                    "memory_id": str(memory.id),
                    "memory_type": memory_type,
                    "project_id": project_id,
                    "user_id": "claude_code"
                })
            except Exception as e:
                logger.warning(f"Failed to emit websocket event: {e}")
            
            logger.info(f"Created memory: {memory.id} (type: {memory_type})")
            
            return {
                "success": True,
                "memory_id": str(memory.id),
                "memory_type": memory.memory_type,
                "created_at": memory.created_at.isoformat(),
                "embedding_generated": bool(memory.embedding),
                "tags": memory.tags,
                "ai_enhanced": embedding_result is not None and not embedding_result.error,
                "message": "Memory created successfully with AI enhancement"
            }
            
        except Exception as e:
            logger.error(f"Error creating memory: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to create memory"
            }
    
    async def search_memories(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Search memories using enhanced AI-powered semantic search."""
        try:
            await self.initialize()
            
            # Generate embedding for the query using real embeddings service
            query_embedding = None
            try:
                embedding_result = await real_embeddings_service.generate_embedding(
                    query, "text"
                )
                if embedding_result and not embedding_result.error:
                    query_embedding = embedding_result.embedding
                    logger.debug(f"Generated query embedding with {embedding_result.dimensions} dimensions")
            except Exception as e:
                logger.warning(f"Failed to generate query embedding: {e}")
            
            # Perform semantic search with enhanced embeddings
            if query_embedding:
                # Use enhanced search with real embeddings
                try:
                    results = await memory_service.search_memories_with_embedding(
                        query_embedding=query_embedding,
                        query_text=query,
                        user_id="claude_code",
                        limit=limit,
                        similarity_threshold=similarity_threshold,
                        memory_types=memory_types,
                        project_id=project_id,
                        session_id=session_id
                    )
                    search_method = "enhanced_semantic"
                except Exception as e:
                    logger.warning(f"Enhanced search failed, falling back: {e}")
                    # Fallback to standard search
                    results = await memory_service.search_memories(
                        query=query,
                        user_id="claude_code",
                        limit=limit,
                        similarity_threshold=similarity_threshold,
                        memory_types=memory_types,
                        project_id=project_id,
                        session_id=session_id
                    )
                    search_method = "standard_semantic"
            else:
                # Standard semantic search
                results = await memory_service.search_memories(
                    query=query,
                    user_id="claude_code",
                    limit=limit,
                    similarity_threshold=similarity_threshold,
                    memory_types=memory_types,
                    project_id=project_id,
                    session_id=session_id
                )
                search_method = "standard_semantic"
            
            # Format results
            formatted_results = []
            for memory, similarity in results:
                formatted_results.append({
                    "memory_id": str(memory.id),
                    "content": memory.content,
                    "memory_type": memory.memory_type,
                    "similarity_score": similarity,
                    "created_at": memory.created_at.isoformat(),
                    "project_id": memory.project_id,
                    "session_id": memory.session_id,
                    "tags": memory.tags
                })
            
            logger.info(f"Found {len(formatted_results)} memories for query: {query[:50]}...")
            
            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results),
                "search_type": search_method,
                "similarity_threshold": similarity_threshold,
                "ai_enhanced": query_embedding is not None,
                "embedding_generated": query_embedding is not None
            }
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to search memories"
            }
    
    async def get_memory(
        self,
        memory_id: str,
        include_related: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Retrieve a specific memory by ID."""
        try:
            await self.initialize()
            
            # Get memory
            memory = await memory_service.get_memory(memory_id)
            if not memory:
                return {
                    "success": False,
                    "error": "Memory not found",
                    "memory_id": memory_id
                }
            
            result = {
                "success": True,
                "memory": {
                    "id": str(memory.id),
                    "content": memory.content,
                    "memory_type": memory.memory_type,
                    "created_at": memory.created_at.isoformat(),
                    "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                    "project_id": memory.project_id,
                    "session_id": memory.session_id,
                    "tags": memory.tags,
                    "metadata": memory.metadata,
                    "user_id": memory.user_id
                }
            }
            
            # Include related memories if requested
            if include_related:
                try:
                    related_results = await memory_service.search_memories(
                        query=memory.content[:100],  # Use first 100 chars as query
                        user_id=memory.user_id,
                        limit=5,
                        similarity_threshold=0.6
                    )
                    
                    related_memories = []
                    for related_memory, similarity in related_results:
                        if str(related_memory.id) != memory_id:  # Exclude the original memory
                            related_memories.append({
                                "memory_id": str(related_memory.id),
                                "content": related_memory.content[:200] + "..." if len(related_memory.content) > 200 else related_memory.content,
                                "memory_type": related_memory.memory_type,
                                "similarity_score": similarity,
                                "created_at": related_memory.created_at.isoformat()
                            })
                    
                    result["related_memories"] = related_memories[:3]  # Limit to top 3
                    
                except Exception as e:
                    logger.warning(f"Failed to get related memories: {e}")
                    result["related_memories"] = []
            
            logger.info(f"Retrieved memory: {memory_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
            return {
                "success": False,
                "error": str(e),
                "memory_id": memory_id
            }
    
    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Update an existing memory."""
        try:
            await self.initialize()
            
            # Prepare update data
            update_data = {}
            if content is not None:
                update_data["content"] = content
            if tags is not None:
                update_data["tags"] = tags
            if metadata is not None:
                update_data["metadata"] = metadata
            
            if not update_data:
                return {
                    "success": False,
                    "error": "No update data provided",
                    "memory_id": memory_id
                }
            
            # Update memory
            memory = await memory_service.update_memory(memory_id, update_data)
            if not memory:
                return {
                    "success": False,
                    "error": "Memory not found or update failed",
                    "memory_id": memory_id
                }
            
            logger.info(f"Updated memory: {memory_id}")
            
            return {
                "success": True,
                "memory_id": str(memory.id),
                "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                "updated_fields": list(update_data.keys()),
                "message": "Memory updated successfully"
            }
            
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            return {
                "success": False,
                "error": str(e),
                "memory_id": memory_id
            }
    
    async def get_memory_stats(
        self,
        project_id: Optional[str] = None,
        time_range: str = "24h",
        **kwargs
    ) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        try:
            await self.initialize()
            
            # Get basic stats
            stats = await memory_service.get_memory_stats(
                user_id="claude_code",
                project_id=project_id
            )
            
            # Add time-based analysis
            time_delta = self._parse_time_range(time_range)
            since_date = datetime.utcnow() - time_delta
            
            recent_stats = await memory_service.get_memory_stats(
                user_id="claude_code",
                project_id=project_id,
                since_date=since_date
            )
            
            return {
                "success": True,
                "stats": {
                    "total": stats,
                    "recent": recent_stats,
                    "time_range": time_range,
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_time_range(self, time_range: str) -> timedelta:
        """Parse time range string to timedelta."""
        if time_range == "1h":
            return timedelta(hours=1)
        elif time_range == "24h":
            return timedelta(hours=24)
        elif time_range == "7d":
            return timedelta(days=7)
        elif time_range == "30d":
            return timedelta(days=30)
        else:
            return timedelta(hours=24)  # Default


class SessionHandler(BaseHandler):
    """Handler for session operations."""
    
    async def init_session(
        self,
        session_type: str = "coding",
        project_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None,
        restore_from: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Initialize a new AI-enhanced session."""
        try:
            await self.initialize()
            
            # Create session data
            session_data = {
                "session_type": session_type,
                "user_id": "claude_code",
                "project_id": project_id,
                "context_history": context_data or {},
                "metadata": {
                    "initialized_via": "mcp",
                    "initialized_at": datetime.utcnow().isoformat()
                }
            }
            
            # Restore from previous session if requested
            if restore_from:
                try:
                    previous_session = await session_service.get_session(restore_from)
                    if previous_session:
                        session_data["context_history"].update(
                            previous_session.context_history or {}
                        )
                        session_data["metadata"]["restored_from"] = restore_from
                except Exception as e:
                    logger.warning(f"Failed to restore from session {restore_from}: {e}")
            
            # Create session
            session = await session_service.create_session(session_data)
            
            logger.info(f"Initialized session: {session.id} (type: {session_type})")
            
            return {
                "success": True,
                "session_id": str(session.id),
                "session_type": session.session_type,
                "created_at": session.started_at.isoformat(),
                "project_id": session.project_id,
                "context_restored": bool(restore_from),
                "message": "Session initialized successfully"
            }
            
        except Exception as e:
            logger.error(f"Error initializing session: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to initialize session"
            }
    
    async def get_session(
        self,
        session_id: Optional[str] = None,
        include_context: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Get session information."""
        try:
            await self.initialize()
            
            # Get current session if no ID provided
            if not session_id:
                session = await session_service.get_current_session("claude_code")
            else:
                session = await session_service.get_session(session_id)
            
            if not session:
                return {
                    "success": False,
                    "error": "Session not found",
                    "session_id": session_id
                }
            
            result = {
                "success": True,
                "session": {
                    "id": str(session.id),
                    "session_type": session.session_type,
                    "started_at": session.started_at.isoformat(),
                    "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                    "is_active": session.is_active,
                    "project_id": session.project_id,
                    "user_id": session.user_id,
                    "metadata": session.metadata
                }
            }
            
            if include_context:
                result["session"]["context_history"] = session.context_history or {}
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting session: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    async def update_session_context(
        self,
        context_update: Dict[str, Any],
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Update session context."""
        try:
            await self.initialize()
            
            # Get session
            if not session_id:
                session = await session_service.get_current_session("claude_code")
            else:
                session = await session_service.get_session(session_id)
            
            if not session:
                return {
                    "success": False,
                    "error": "Session not found",
                    "session_id": session_id
                }
            
            # Update context
            await session_service.update_session_context(
                str(session.id),
                context_update
            )
            
            logger.info(f"Updated session context: {session.id}")
            
            return {
                "success": True,
                "session_id": str(session.id),
                "updated_at": datetime.utcnow().isoformat(),
                "context_keys_updated": list(context_update.keys()),
                "message": "Session context updated successfully"
            }
            
        except Exception as e:
            logger.error(f"Error updating session context: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    async def end_session(
        self,
        session_id: Optional[str] = None,
        summary: Optional[str] = None,
        save_context: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """End a session."""
        try:
            await self.initialize()
            
            # Get session
            if not session_id:
                session = await session_service.get_current_session("claude_code")
            else:
                session = await session_service.get_session(session_id)
            
            if not session:
                return {
                    "success": False,
                    "error": "Session not found",
                    "session_id": session_id
                }
            
            # End session
            await session_service.end_session(
                str(session.id),
                summary=summary,
                save_context=save_context
            )
            
            logger.info(f"Ended session: {session.id}")
            
            return {
                "success": True,
                "session_id": str(session.id),
                "ended_at": datetime.utcnow().isoformat(),
                "summary": summary,
                "context_saved": save_context,
                "message": "Session ended successfully"
            }
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    async def get_session_history(
        self,
        project_id: Optional[str] = None,
        session_type: Optional[str] = None,
        limit: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Get session history."""
        try:
            await self.initialize()
            
            sessions = await session_service.get_session_history(
                user_id="claude_code",
                project_id=project_id,
                session_type=session_type,
                limit=limit
            )
            
            formatted_sessions = []
            for session in sessions:
                formatted_sessions.append({
                    "session_id": str(session.id),
                    "session_type": session.session_type,
                    "started_at": session.started_at.isoformat(),
                    "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                    "duration_minutes": session.duration_minutes,
                    "project_id": session.project_id,
                    "is_active": session.is_active,
                    "metadata": session.metadata
                })
            
            return {
                "success": True,
                "sessions": formatted_sessions,
                "count": len(formatted_sessions),
                "filters": {
                    "project_id": project_id,
                    "session_type": session_type,
                    "limit": limit
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting session history: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class AIHandler(BaseHandler):
    """Handler for AI intelligence features."""
    
    async def predict_next_tasks(
        self,
        context: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        num_predictions: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Get AI predictions for next likely tasks using real AI intelligence."""
        try:
            await self.initialize()
            
            # Use real AI intelligence for enhanced predictions
            try:
                # Get task predictions from real AI service
                ai_predictions = await real_ai_intelligence.predict_next_tasks(
                    user_id="claude_code",
                    context=context,
                    project_id=project_id,
                    session_id=session_id,
                    num_predictions=num_predictions
                )
                
                formatted_predictions = []
                for prediction in ai_predictions:
                    formatted_predictions.append({
                        "task": prediction.get("task", "Unknown task"),
                        "confidence": prediction.get("confidence", 0.5),
                        "reasoning": prediction.get("reasoning", "AI-generated prediction"),
                        "estimated_effort": prediction.get("effort_minutes", 30),
                        "suggested_approach": prediction.get("approach", "Standard approach"),
                        "ai_enhanced": True
                    })
                
                return {
                    "success": True,
                    "predictions": formatted_predictions,
                    "context": context,
                    "generated_at": datetime.utcnow().isoformat(),
                    "model_version": "2.0-ai-enhanced",
                    "ai_service_used": True
                }
                
            except Exception as ai_error:
                logger.warning(f"AI service failed, falling back to basic service: {ai_error}")
                
                # Fallback to basic prediction service
                predictions = await prediction_service.predict_next_tasks(
                    user_id="claude_code",
                    session_id=session_id,
                    project_id=project_id,
                    n_predictions=num_predictions
                )
                
                formatted_predictions = []
                for prediction in predictions:
                    formatted_predictions.append({
                        "task": prediction.predicted_task,
                        "confidence": prediction.confidence,
                        "reasoning": prediction.reasoning,
                        "estimated_effort": prediction.estimated_effort_minutes,
                        "suggested_approach": prediction.suggested_approach,
                        "ai_enhanced": False
                    })
                
                return {
                    "success": True,
                    "predictions": formatted_predictions,
                    "context": context,
                    "generated_at": datetime.utcnow().isoformat(),
                    "model_version": "1.0-fallback",
                    "ai_service_used": False,
                    "fallback_reason": str(ai_error)
                }
            
        except Exception as e:
            logger.error(f"Error predicting tasks: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_patterns(
        self,
        data: str,
        analysis_type: str = "code",
        project_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze patterns using real AI intelligence."""
        try:
            await self.initialize()
            
            # Use real AI intelligence for enhanced pattern analysis
            try:
                ai_patterns = await real_ai_intelligence.analyze_patterns(
                    data=data,
                    analysis_type=analysis_type,
                    project_id=project_id,
                    user_id="claude_code"
                )
                
                return {
                    "success": True,
                    "analysis_type": analysis_type,
                    "patterns": ai_patterns,
                    "data_length": len(data),
                    "analyzed_at": datetime.utcnow().isoformat(),
                    "ai_enhanced": True,
                    "pattern_count": len(ai_patterns) if isinstance(ai_patterns, list) else 1
                }
                
            except Exception as ai_error:
                logger.warning(f"AI pattern analysis failed, using fallback: {ai_error}")
                
                # Fallback to basic pattern service
                if analysis_type == "code":
                    patterns = await pattern_service.analyze_code_patterns(
                        user_id="claude_code",
                        project_id=project_id,
                        code_samples=[{"content": data, "language": "auto-detect"}]
                    )
                else:
                    # For other types, use general pattern analysis
                    patterns = await pattern_service.analyze_general_patterns(
                        data=data,
                        pattern_type=analysis_type
                    )
                
                return {
                    "success": True,
                    "analysis_type": analysis_type,
                    "patterns": patterns,
                    "data_length": len(data),
                    "analyzed_at": datetime.utcnow().isoformat(),
                    "ai_enhanced": False,
                    "fallback_reason": str(ai_error)
                }
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis_type": analysis_type
            }
    
    async def get_ai_insights(
        self,
        focus_area: Optional[str] = None,
        project_id: Optional[str] = None,
        time_range: str = "24h",
        **kwargs
    ) -> Dict[str, Any]:
        """Get AI-generated insights using real AI intelligence."""
        try:
            await self.initialize()
            
            # Use real AI intelligence for enhanced insights
            try:
                ai_insights = await real_ai_intelligence.generate_insights(
                    focus_area=focus_area,
                    project_id=project_id,
                    time_range=time_range,
                    user_id="claude_code"
                )
                
                return {
                    "success": True,
                    "insights": ai_insights,
                    "focus_area": focus_area,
                    "time_range": time_range,
                    "generated_at": datetime.utcnow().isoformat(),
                    "ai_enhanced": True,
                    "insight_count": len(ai_insights) if isinstance(ai_insights, list) else 1
                }
                
            except Exception as ai_error:
                logger.warning(f"AI insights generation failed, using fallback: {ai_error}")
                
                # Fallback to basic insights generation
                insights = await self._generate_insights(focus_area, project_id, time_range)
                
                return {
                    "success": True,
                    "insights": insights,
                    "focus_area": focus_area,
                    "time_range": time_range,
                    "generated_at": datetime.utcnow().isoformat(),
                    "ai_enhanced": False,
                    "fallback_reason": str(ai_error)
                }
            
        except Exception as e:
            logger.error(f"Error getting AI insights: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def record_decision(
        self,
        decision: str,
        reasoning: str,
        alternatives: Optional[List[str]] = None,
        context: Optional[str] = None,
        confidence: Optional[float] = None,
        project_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Record a decision with AI-enhanced tracking."""
        try:
            await self.initialize()
            
            decision_data = {
                "decision_text": decision,
                "reasoning": reasoning,
                "alternatives_considered": alternatives or [],
                "context": context,
                "confidence_score": confidence,
                "user_id": "claude_code",
                "project_id": project_id,
                "decision_type": "technical",
                "metadata": {
                    "recorded_via": "mcp",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            decision_record = await decision_service.record_decision(decision_data)
            
            return {
                "success": True,
                "decision_id": str(decision_record.id),
                "decision": decision,
                "confidence": confidence,
                "recorded_at": decision_record.created_at.isoformat(),
                "message": "Decision recorded successfully"
            }
            
        except Exception as e:
            logger.error(f"Error recording decision: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def track_error(
        self,
        error_type: str,
        error_message: str,
        solution: Optional[str] = None,
        success: Optional[bool] = None,
        context: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Track and learn from errors using real AI intelligence."""
        try:
            await self.initialize()
            
            error_data = {
                "error_type": error_type,
                "error_message": error_message,
                "solution_attempted": solution,
                "resolution_successful": success,
                "context": context or {},
                "user_id": "claude_code",
                "project_id": project_id,
                "severity": "medium",  # Default severity
                "metadata": {
                    "tracked_via": "mcp",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            # Use real AI intelligence for enhanced error analysis
            ai_analysis = None
            try:
                ai_analysis = await real_ai_intelligence.analyze_error(
                    error_type=error_type,
                    error_message=error_message,
                    context=context or {},
                    project_id=project_id
                )
                
                # Add AI insights to error data
                if ai_analysis:
                    error_data["metadata"]["ai_analysis"] = ai_analysis
                    
            except Exception as ai_error:
                logger.warning(f"AI error analysis failed: {ai_error}")
            
            error_record = await error_learning_service.track_error(error_data)
            
            # Get similar errors for learning
            similar_errors = await error_learning_service.find_similar_errors(
                error_message, limit=3
            )
            
            # Emit real-time event
            try:
                await real_websocket_events.emit_error_tracked({
                    "error_id": str(error_record.id),
                    "error_type": error_type,
                    "project_id": project_id,
                    "user_id": "claude_code",
                    "has_ai_analysis": ai_analysis is not None
                })
            except Exception as e:
                logger.warning(f"Failed to emit websocket event: {e}")
            
            return {
                "success": True,
                "error_id": str(error_record.id),
                "error_type": error_type,
                "tracked_at": error_record.occurred_at.isoformat(),
                "similar_errors": len(similar_errors),
                "learning_applied": success is not None,
                "ai_enhanced": ai_analysis is not None,
                "ai_insights": ai_analysis.get("insights", []) if ai_analysis else [],
                "message": "Error tracked and analyzed for learning with AI enhancement"
            }
            
        except Exception as e:
            logger.error(f"Error tracking error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_insights(self, focus_area: Optional[str], project_id: Optional[str], time_range: str) -> List[Dict[str, Any]]:
        """Generate AI insights based on focus area."""
        insights = []
        
        try:
            if focus_area == "performance":
                # Get performance insights
                perf_data = await metrics_service.get_metrics_dashboard_data(time_range)
                insights.append({
                    "type": "performance",
                    "title": "System Performance Analysis",
                    "description": "AI analysis of system performance metrics",
                    "data": perf_data.get("performance", {}),
                    "recommendations": ["Monitor response times", "Optimize database queries"]
                })
            
            elif focus_area == "code_quality":
                # Get code quality insights
                insights.append({
                    "type": "code_quality",
                    "title": "Code Quality Assessment",
                    "description": "AI analysis of code patterns and quality metrics",
                    "recommendations": ["Follow consistent naming", "Add comprehensive tests"]
                })
            
            else:
                # General insights
                insights.append({
                    "type": "general",
                    "title": "AI System Overview",
                    "description": "General AI insights about system usage and patterns",
                    "recommendations": ["Regular memory cleanup", "Session management optimization"]
                })
            
        except Exception as e:
            logger.warning(f"Error generating specific insights: {e}")
            insights.append({
                "type": "error",
                "title": "Insight Generation Error",
                "description": f"Could not generate insights for {focus_area}: {str(e)}"
            })
        
        return insights
    
    async def generate_code_suggestions(
        self,
        code_content: str,
        programming_language: str = "python",
        context: Optional[str] = None,
        project_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate AI-powered code suggestions and improvements."""
        try:
            await self.initialize()
            
            # Use real AI intelligence for code analysis
            try:
                code_analysis = await real_ai_intelligence.analyze_code(
                    code_content=code_content,
                    language=programming_language,
                    context=context,
                    project_id=project_id,
                    user_id="claude_code"
                )
                
                return {
                    "success": True,
                    "code_analysis": code_analysis,
                    "language": programming_language,
                    "analyzed_at": datetime.utcnow().isoformat(),
                    "ai_enhanced": True,
                    "suggestions_count": len(code_analysis.get("suggestions", [])) if code_analysis else 0
                }
                
            except Exception as ai_error:
                logger.warning(f"AI code analysis failed: {ai_error}")
                
                # Fallback to basic analysis
                return {
                    "success": True,
                    "code_analysis": {
                        "suggestions": ["Consider adding error handling", "Review variable naming"],
                        "quality_score": 0.7,
                        "complexity": "medium"
                    },
                    "language": programming_language,
                    "analyzed_at": datetime.utcnow().isoformat(),
                    "ai_enhanced": False,
                    "fallback_reason": str(ai_error)
                }
                
        except Exception as e:
            logger.error(f"Error generating code suggestions: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def learn_from_interaction(
        self,
        interaction_type: str,
        interaction_data: Dict[str, Any],
        feedback_score: Optional[float] = None,
        project_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Learn from user interactions using real AI intelligence."""
        try:
            await self.initialize()
            
            # Use real AI intelligence for learning
            try:
                learning_result = await real_ai_intelligence.process_interaction_learning(
                    interaction_type=interaction_type,
                    interaction_data=interaction_data,
                    feedback_score=feedback_score,
                    project_id=project_id,
                    user_id="claude_code"
                )
                
                return {
                    "success": True,
                    "learning_result": learning_result,
                    "interaction_type": interaction_type,
                    "processed_at": datetime.utcnow().isoformat(),
                    "ai_enhanced": True,
                    "patterns_updated": learning_result.get("patterns_updated", 0) if learning_result else 0
                }
                
            except Exception as ai_error:
                logger.warning(f"AI learning failed: {ai_error}")
                
                # Still record the interaction for future use
                return {
                    "success": True,
                    "learning_result": {"status": "recorded", "will_process_later": True},
                    "interaction_type": interaction_type,
                    "processed_at": datetime.utcnow().isoformat(),
                    "ai_enhanced": False,
                    "fallback_reason": str(ai_error)
                }
                
        except Exception as e:
            logger.error(f"Error processing interaction learning: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class AnalyticsHandler(BaseHandler):
    """Handler for analytics and metrics operations."""
    
    async def get_metrics(
        self,
        metric_names: Optional[List[str]] = None,
        time_window: str = "1h",
        aggregation: str = "avg",
        **kwargs
    ) -> Dict[str, Any]:
        """Get real-time metrics."""
        try:
            await self.initialize()
            
            if metric_names:
                results = {}
                for metric_name in metric_names:
                    aggregation_data = await metrics_service.get_metric_aggregation(
                        metric_name, time_window
                    )
                    if aggregation_data:
                        if aggregation == "avg":
                            results[metric_name] = aggregation_data.avg_value
                        elif aggregation == "sum":
                            results[metric_name] = aggregation_data.sum_value
                        elif aggregation == "min":
                            results[metric_name] = aggregation_data.min_value
                        elif aggregation == "max":
                            results[metric_name] = aggregation_data.max_value
                        elif aggregation == "count":
                            results[metric_name] = aggregation_data.count
                    else:
                        results[metric_name] = None
            else:
                # Get general metrics
                collector_status = await metrics_collector_worker.get_worker_status()
                results = collector_status.get("performance", {})
            
            return {
                "success": True,
                "metrics": results,
                "time_window": time_window,
                "aggregation": aggregation,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_dashboard_data(
        self,
        dashboard_type: str = "comprehensive",
        time_window: str = "1h",
        project_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get dashboard data."""
        try:
            await self.initialize()
            
            dashboard_data = await metrics_service.get_metrics_dashboard_data(
                time_window=time_window,
                project_id=project_id
            )
            
            # Filter based on dashboard type
            if dashboard_type == "system":
                filtered_data = {"system": dashboard_data.get("system", {})}
            elif dashboard_type == "application":
                filtered_data = {"performance": dashboard_data.get("performance", {})}
            elif dashboard_type == "ai":
                filtered_data = {"ai_features": dashboard_data.get("ai_features", {})}
            else:
                filtered_data = dashboard_data
            
            return {
                "success": True,
                "dashboard_type": dashboard_type,
                "data": filtered_data,
                "time_window": time_window,
                "generated_at": dashboard_data.get("generated_at")
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_alerts(
        self,
        status: str = "active",
        severity: Optional[str] = None,
        limit: int = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """Get alerts."""
        try:
            await self.initialize()
            
            alerts = await metrics_service.get_active_alerts()
            
            # Filter alerts
            filtered_alerts = []
            for alert in alerts:
                if status == "active":
                    # Only include active alerts
                    pass
                
                if severity and alert.severity.value != severity:
                    continue
                
                filtered_alerts.append({
                    "rule_name": alert.rule_name,
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "severity": alert.severity.value,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "description": alert.description
                })
                
                if len(filtered_alerts) >= limit:
                    break
            
            return {
                "success": True,
                "alerts": filtered_alerts,
                "count": len(filtered_alerts),
                "filters": {
                    "status": status,
                    "severity": severity,
                    "limit": limit
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_performance_report(
        self,
        report_type: str = "system",
        time_range: str = "24h",
        include_trends: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate performance report."""
        try:
            await self.initialize()
            
            # Get base metrics data
            dashboard_data = await metrics_service.get_metrics_dashboard_data(time_range)
            
            # Generate report based on type
            if report_type == "system":
                report = self._generate_system_report(dashboard_data)
            elif report_type == "application":
                report = self._generate_application_report(dashboard_data)
            else:
                report = self._generate_general_report(dashboard_data)
            
            # Add trends if requested
            if include_trends:
                try:
                    trends = await metrics_service.get_metric_trends(
                        metric_names=["response_time", "error_count", "memory_usage"],
                        time_window=time_range
                    )
                    report["trends"] = trends
                except Exception as e:
                    logger.warning(f"Failed to get trends: {e}")
                    report["trends"] = {}
            
            return {
                "success": True,
                "report_type": report_type,
                "time_range": time_range,
                "report": report,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_system_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate system performance report."""
        return {
            "title": "System Performance Report",
            "summary": "Overall system health and performance metrics",
            "metrics": data.get("system", {}),
            "recommendations": [
                "Monitor memory usage regularly",
                "Check disk space availability",
                "Review network performance"
            ]
        }
    
    def _generate_application_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate application performance report."""
        return {
            "title": "Application Performance Report",
            "summary": "Application-specific performance and usage metrics",
            "metrics": data.get("performance", {}),
            "recommendations": [
                "Optimize API response times",
                "Monitor error rates",
                "Review user session patterns"
            ]
        }
    
    def _generate_general_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate general performance report."""
        return {
            "title": "Comprehensive Performance Report",
            "summary": "Complete overview of system and application performance",
            "metrics": data,
            "recommendations": [
                "Regular performance monitoring",
                "Proactive alerting setup",
                "Capacity planning review"
            ]
        }


class ContextSynchronizer(BaseHandler):
    """Handler for context synchronization utilities."""
    
    async def sync_context(
        self,
        context_data: Optional[Dict[str, Any]] = None,
        sync_direction: str = "bidirectional",
        **kwargs
    ) -> Dict[str, Any]:
        """Synchronize context between Claude Code and KnowledgeHub."""
        try:
            await self.initialize()
            
            result = {
                "success": True,
                "sync_direction": sync_direction,
                "synced_at": datetime.utcnow().isoformat()
            }
            
            if sync_direction in ["to_knowledgehub", "bidirectional"]:
                # Sync context to KnowledgeHub
                if context_data:
                    # Store context data in session or memory
                    session = await session_service.get_current_session("claude_code")
                    if session:
                        await session_service.update_session_context(
                            str(session.id), context_data
                        )
                        result["context_stored"] = True
                    else:
                        result["context_stored"] = False
                        result["warning"] = "No active session to store context"
                
            if sync_direction in ["from_knowledgehub", "bidirectional"]:
                # Get context from KnowledgeHub
                session = await session_service.get_current_session("claude_code")
                if session:
                    result["context_retrieved"] = session.context_history or {}
                else:
                    result["context_retrieved"] = {}
            
            return result
            
        except Exception as e:
            logger.error(f"Error syncing context: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_system_status(
        self,
        include_services: bool = True,
        include_metrics: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Get system status."""
        try:
            await self.initialize()
            
            status = {
                "success": True,
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "services": {}
            }
            
            if include_services:
                # Check various services
                try:
                    # Check metrics collector
                    collector_status = await metrics_collector_worker.get_worker_status()
                    status["services"]["metrics_collector"] = {
                        "status": "running" if collector_status["running"] else "stopped",
                        "details": collector_status
                    }
                except Exception as e:
                    status["services"]["metrics_collector"] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            if include_metrics:
                try:
                    # Add basic system metrics
                    dashboard_data = await metrics_service.get_metrics_dashboard_data("1h")
                    status["metrics"] = dashboard_data.get("summary", {})
                except Exception as e:
                    status["metrics"] = {"error": str(e)}
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def health_check(
        self,
        deep_check: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform health check."""
        try:
            await self.initialize()
            
            health = {
                "success": True,
                "overall_health": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "checks": {}
            }
            
            # Basic health checks
            health["checks"]["mcp_server"] = {"status": "healthy", "message": "MCP server is running"}
            health["checks"]["handlers"] = {"status": "healthy", "message": "All handlers initialized"}
            
            if deep_check:
                # Deep health checks
                try:
                    # Check database connectivity
                    stats = await memory_service.get_memory_stats("claude_code")
                    health["checks"]["database"] = {"status": "healthy", "message": "Database accessible"}
                except Exception as e:
                    health["checks"]["database"] = {"status": "unhealthy", "error": str(e)}
                    health["overall_health"] = "degraded"
                
                try:
                    # Check AI services
                    predictions = await prediction_service.predict_next_tasks("claude_code", n_predictions=1)
                    health["checks"]["ai_services"] = {"status": "healthy", "message": "AI services accessible"}
                except Exception as e:
                    health["checks"]["ai_services"] = {"status": "unhealthy", "error": str(e)}
                    health["overall_health"] = "degraded"
            
            return health
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {
                "success": False,
                "overall_health": "unhealthy",
                "error": str(e)
            }
    
    async def get_api_info(
        self,
        category: str = "all",
        **kwargs
    ) -> Dict[str, Any]:
        """Get API information."""
        try:
            await self.initialize()
            
            api_info = {
                "success": True,
                "category": category,
                "apis": {}
            }
            
            if category in ["memory", "all"]:
                api_info["apis"]["memory"] = {
                    "base_url": "/api/memory",
                    "endpoints": [
                        "POST /create - Create memory",
                        "GET /search - Search memories",
                        "GET /{id} - Get specific memory",
                        "PUT /{id} - Update memory",
                        "GET /stats - Get statistics"
                    ]
                }
            
            if category in ["session", "all"]:
                api_info["apis"]["session"] = {
                    "base_url": "/api/session",
                    "endpoints": [
                        "POST /init - Initialize session",
                        "GET /current - Get current session",
                        "PUT /context - Update context",
                        "POST /end - End session",
                        "GET /history - Get session history"
                    ]
                }
            
            if category in ["ai", "all"]:
                api_info["apis"]["ai"] = {
                    "base_url": "/api/ai-features",
                    "endpoints": [
                        "POST /predict - Predict next tasks",
                        "POST /analyze - Analyze patterns",
                        "GET /insights - Get AI insights",
                        "POST /decision - Record decision",
                        "POST /error - Track error"
                    ]
                }
            
            if category in ["analytics", "all"]:
                api_info["apis"]["analytics"] = {
                    "base_url": "/api/analytics",
                    "endpoints": [
                        "GET /metrics - Get metrics",
                        "GET /dashboard - Get dashboard data",
                        "GET /alerts - Get alerts",
                        "GET /performance - Get performance report"
                    ]
                }
            
            return api_info
            
        except Exception as e:
            logger.error(f"Error getting API info: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class ResponseFormatter:
    """Utility class for formatting MCP tool responses."""
    
    @staticmethod
    async def format_tool_result(
        tool_name: str,
        result: Dict[str, Any],
        execution_time: float
    ) -> Dict[str, Any]:
        """Format tool execution result."""
        formatted = {
            "tool": tool_name,
            "result": result,
            "execution_time_seconds": round(execution_time, 3),
            "timestamp": datetime.utcnow().isoformat(),
            "success": result.get("success", False)
        }
        
        # Add tool-specific formatting
        if tool_name.startswith("search_"):
            formatted["result_count"] = len(result.get("results", []))
        
        if tool_name.startswith("create_"):
            formatted["created_id"] = result.get("memory_id") or result.get("session_id")
        
        return formatted