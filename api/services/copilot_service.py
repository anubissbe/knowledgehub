"""
GitHub Copilot Enhancement Service.

Provides webhook receiver and suggestion enhancement for GitHub Copilot
integration with KnowledgeHub AI intelligence.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from ..models.memory import Memory
from ..models.session import Session
from ..models.analytics import Metric
from ..database import get_db_session
from .memory_service import MemoryService
from .session_service import SessionService
from .ai_service import AIService
from .pattern_service import PatternService

logger = logging.getLogger(__name__)


class CopilotSuggestion:
    """Represents a Copilot suggestion with KnowledgeHub enhancements."""
    
    def __init__(
        self,
        original_suggestion: str,
        enhanced_suggestion: str,
        confidence: float,
        context_sources: List[str],
        learning_data: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid4())
        self.original_suggestion = original_suggestion
        self.enhanced_suggestion = enhanced_suggestion
        self.confidence = confidence
        self.context_sources = context_sources
        self.learning_data = learning_data or {}
        self.timestamp = datetime.utcnow()


class CopilotEnhancementService:
    """Service for enhancing GitHub Copilot with KnowledgeHub intelligence."""
    
    def __init__(self):
        self.memory_service = MemoryService()
        self.session_service = SessionService()
        self.ai_service = AIService()
        self.pattern_service = PatternService()
        self.active_suggestions: Dict[str, CopilotSuggestion] = {}
        
    async def receive_webhook(
        self,
        webhook_type: str,
        payload: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Receive and process GitHub Copilot webhook events.
        
        Args:
            webhook_type: Type of webhook event
            payload: Webhook payload data
            user_id: User ID for context
            
        Returns:
            Dict containing processing results and any enhanced data
        """
        try:
            logger.info(f"Received Copilot webhook: {webhook_type}")
            
            # Process different webhook types
            if webhook_type == "suggestion_request":
                return await self._handle_suggestion_request(payload, user_id)
            elif webhook_type == "suggestion_accepted":
                return await self._handle_suggestion_accepted(payload, user_id)
            elif webhook_type == "suggestion_rejected":
                return await self._handle_suggestion_rejected(payload, user_id)
            elif webhook_type == "completion_request":
                return await self._handle_completion_request(payload, user_id)
            elif webhook_type == "feedback":
                return await self._handle_feedback(payload, user_id)
            else:
                logger.warning(f"Unknown webhook type: {webhook_type}")
                return {"status": "ignored", "reason": f"Unknown type: {webhook_type}"}
                
        except Exception as e:
            logger.error(f"Error processing Copilot webhook: {e}")
            raise HTTPException(status_code=500, detail=f"Webhook processing failed: {e}")
    
    async def enhance_suggestion(
        self,
        original_suggestion: str,
        context: Dict[str, Any],
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> CopilotSuggestion:
        """
        Enhance a Copilot suggestion with KnowledgeHub context and intelligence.
        
        Args:
            original_suggestion: Original Copilot suggestion
            context: Code and project context
            user_id: User ID for personalized enhancement
            project_id: Project ID for project-specific context
            
        Returns:
            Enhanced suggestion with KnowledgeHub intelligence
        """
        try:
            # Extract context information
            code_context = context.get("code", "")
            file_path = context.get("file_path", "")
            language = context.get("language", "")
            cursor_position = context.get("cursor_position", {})
            
            # Get relevant memories and patterns
            relevant_memories = await self._get_relevant_memories(
                code_context, file_path, user_id, project_id
            )
            
            # Analyze patterns in the current context
            patterns = await self._analyze_context_patterns(
                code_context, file_path, language, user_id, project_id
            )
            
            # Get project-specific conventions
            conventions = await self._get_project_conventions(
                project_id, language, file_path
            )
            
            # Enhance the suggestion using AI
            enhanced_suggestion = await self._ai_enhance_suggestion(
                original_suggestion,
                code_context,
                relevant_memories,
                patterns,
                conventions
            )
            
            # Calculate confidence based on context relevance
            confidence = await self._calculate_enhancement_confidence(
                original_suggestion,
                enhanced_suggestion,
                relevant_memories,
                patterns
            )
            
            # Prepare context sources for transparency
            context_sources = self._prepare_context_sources(
                relevant_memories, patterns, conventions
            )
            
            # Create enhanced suggestion
            suggestion = CopilotSuggestion(
                original_suggestion=original_suggestion,
                enhanced_suggestion=enhanced_suggestion,
                confidence=confidence,
                context_sources=context_sources,
                learning_data={
                    "file_path": file_path,
                    "language": language,
                    "patterns_used": len(patterns),
                    "memories_used": len(relevant_memories)
                }
            )
            
            # Store for feedback tracking
            self.active_suggestions[suggestion.id] = suggestion
            
            logger.info(f"Enhanced suggestion with confidence {confidence:.2f}")
            return suggestion
            
        except Exception as e:
            logger.error(f"Error enhancing suggestion: {e}")
            # Return original suggestion if enhancement fails
            return CopilotSuggestion(
                original_suggestion=original_suggestion,
                enhanced_suggestion=original_suggestion,
                confidence=0.5,
                context_sources=["fallback"],
                learning_data={"error": str(e)}
            )
    
    async def inject_context(
        self,
        request: Dict[str, Any],
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Inject KnowledgeHub context into Copilot requests.
        
        Args:
            request: Original Copilot request
            user_id: User ID for context
            project_id: Project ID for context
            
        Returns:
            Enhanced request with injected context
        """
        try:
            # Extract request data
            prompt = request.get("prompt", "")
            context = request.get("context", {})
            
            # Get session context
            session_context = await self._get_session_context(user_id)
            
            # Get project context
            project_context = await self._get_project_context(project_id)
            
            # Get relevant decisions and learnings
            decisions = await self._get_relevant_decisions(prompt, user_id, project_id)
            
            # Inject context into the request
            enhanced_request = request.copy()
            enhanced_request["knowledgehub_context"] = {
                "session": session_context,
                "project": project_context,
                "decisions": decisions,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Enhance the prompt with context
            if session_context or project_context or decisions:
                context_injection = self._build_context_injection(
                    session_context, project_context, decisions
                )
                enhanced_request["prompt"] = f"{prompt}\n\n{context_injection}"
            
            logger.info("Injected KnowledgeHub context into Copilot request")
            return enhanced_request
            
        except Exception as e:
            logger.error(f"Error injecting context: {e}")
            return request  # Return original if injection fails
    
    async def create_feedback_loop(
        self,
        suggestion_id: str,
        feedback_type: str,
        feedback_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create feedback loop for continuous learning from Copilot interactions.
        
        Args:
            suggestion_id: ID of the suggestion being rated
            feedback_type: Type of feedback (accepted, rejected, modified, etc.)
            feedback_data: Additional feedback data
            
        Returns:
            Dict containing feedback processing results
        """
        try:
            # Get the original suggestion
            suggestion = self.active_suggestions.get(suggestion_id)
            if not suggestion:
                logger.warning(f"Suggestion {suggestion_id} not found for feedback")
                return {"status": "ignored", "reason": "Suggestion not found"}
            
            # Process feedback based on type
            learning_update = await self._process_feedback(
                suggestion, feedback_type, feedback_data
            )
            
            # Update AI models based on feedback
            await self._update_models_from_feedback(
                suggestion, feedback_type, learning_update
            )
            
            # Store feedback for analysis
            await self._store_feedback_data(
                suggestion_id, feedback_type, feedback_data, learning_update
            )
            
            # Clean up processed suggestion
            if suggestion_id in self.active_suggestions:
                del self.active_suggestions[suggestion_id]
            
            logger.info(f"Processed feedback for suggestion {suggestion_id}")
            return {
                "status": "processed",
                "suggestion_id": suggestion_id,
                "feedback_type": feedback_type,
                "learning_applied": True
            }
            
        except Exception as e:
            logger.error(f"Error creating feedback loop: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _handle_suggestion_request(
        self,
        payload: Dict[str, Any],
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Handle suggestion request webhook."""
        context = payload.get("context", {})
        suggestion = payload.get("suggestion", "")
        
        if suggestion:
            enhanced = await self.enhance_suggestion(
                suggestion, context, user_id, context.get("project_id")
            )
            return {
                "status": "enhanced",
                "suggestion_id": enhanced.id,
                "enhanced_suggestion": enhanced.enhanced_suggestion,
                "confidence": enhanced.confidence,
                "sources": enhanced.context_sources
            }
        
        return {"status": "no_enhancement", "reason": "No suggestion provided"}
    
    async def _handle_suggestion_accepted(
        self,
        payload: Dict[str, Any],
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Handle suggestion accepted webhook."""
        suggestion_id = payload.get("suggestion_id")
        if suggestion_id:
            return await self.create_feedback_loop(
                suggestion_id, "accepted", payload
            )
        return {"status": "ignored", "reason": "No suggestion ID"}
    
    async def _handle_suggestion_rejected(
        self,
        payload: Dict[str, Any],
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Handle suggestion rejected webhook."""
        suggestion_id = payload.get("suggestion_id")
        if suggestion_id:
            return await self.create_feedback_loop(
                suggestion_id, "rejected", payload
            )
        return {"status": "ignored", "reason": "No suggestion ID"}
    
    async def _handle_completion_request(
        self,
        payload: Dict[str, Any],
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Handle completion request webhook."""
        request = payload.get("request", {})
        enhanced_request = await self.inject_context(
            request, user_id, request.get("project_id")
        )
        return {
            "status": "context_injected",
            "enhanced_request": enhanced_request
        }
    
    async def _handle_feedback(
        self,
        payload: Dict[str, Any],
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Handle general feedback webhook."""
        suggestion_id = payload.get("suggestion_id")
        feedback_type = payload.get("feedback_type", "general")
        
        if suggestion_id:
            return await self.create_feedback_loop(
                suggestion_id, feedback_type, payload
            )
        return {"status": "ignored", "reason": "No suggestion ID"}
    
    async def _get_relevant_memories(
        self,
        code_context: str,
        file_path: str,
        user_id: Optional[str],
        project_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Get relevant memories for enhancement."""
        try:
            # Search for relevant code memories
            memories = await self.memory_service.search_memories(
                query=code_context[:500],  # Limit context size
                user_id=user_id,
                project_id=project_id,
                memory_type="code",
                limit=5
            )
            
            return [
                {
                    "content": m.content,
                    "type": m.memory_type,
                    "relevance": getattr(m, "similarity_score", 0.7)
                }
                for m in memories
            ]
        except Exception as e:
            logger.error(f"Error getting relevant memories: {e}")
            return []
    
    async def _analyze_context_patterns(
        self,
        code_context: str,
        file_path: str,
        language: str,
        user_id: Optional[str],
        project_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Analyze patterns in the current context."""
        try:
            # Use pattern service to analyze code
            patterns = await self.pattern_service.analyze_code_patterns(
                user_id=user_id,
                project_id=project_id,
                file_patterns=[file_path] if file_path else None
            )
            
            return [
                {
                    "type": p.get("pattern_type"),
                    "description": p.get("description"),
                    "confidence": p.get("confidence", 0.7)
                }
                for p in patterns.get("patterns", [])
            ]
        except Exception as e:
            logger.error(f"Error analyzing context patterns: {e}")
            return []
    
    async def _get_project_conventions(
        self,
        project_id: Optional[str],
        language: str,
        file_path: str
    ) -> Dict[str, Any]:
        """Get project-specific coding conventions."""
        # This would integrate with project analysis
        return {
            "language": language,
            "style_guide": "standard",
            "patterns": [],
            "file_path": file_path
        }
    
    async def _ai_enhance_suggestion(
        self,
        original: str,
        context: str,
        memories: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]],
        conventions: Dict[str, Any]
    ) -> str:
        """Use AI to enhance the suggestion."""
        try:
            # Prepare enhancement prompt
            enhancement_data = {
                "original_suggestion": original,
                "code_context": context,
                "relevant_memories": memories[:3],  # Top 3 memories
                "detected_patterns": patterns[:3],  # Top 3 patterns
                "conventions": conventions
            }
            
            # Use AI service for enhancement
            enhanced = await self.ai_service.generate_ai_insights(
                context="copilot_enhancement",
                data=enhancement_data
            )
            
            return enhanced.get("enhanced_suggestion", original)
            
        except Exception as e:
            logger.error(f"Error in AI enhancement: {e}")
            return original
    
    async def _calculate_enhancement_confidence(
        self,
        original: str,
        enhanced: str,
        memories: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for the enhancement."""
        try:
            base_confidence = 0.5
            
            # Boost confidence based on context quality
            if memories:
                avg_memory_relevance = sum(m.get("relevance", 0) for m in memories) / len(memories)
                base_confidence += avg_memory_relevance * 0.2
            
            if patterns:
                avg_pattern_confidence = sum(p.get("confidence", 0) for p in patterns) / len(patterns)
                base_confidence += avg_pattern_confidence * 0.2
            
            # Boost if enhancement significantly different from original
            if len(enhanced) > len(original) * 1.2:
                base_confidence += 0.1
            
            return min(base_confidence, 1.0)
            
        except Exception:
            return 0.5
    
    def _prepare_context_sources(
        self,
        memories: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]],
        conventions: Dict[str, Any]
    ) -> List[str]:
        """Prepare list of context sources for transparency."""
        sources = []
        
        if memories:
            sources.append(f"{len(memories)} relevant memories")
        if patterns:
            sources.append(f"{len(patterns)} code patterns")
        if conventions:
            sources.append("project conventions")
        
        return sources or ["original suggestion"]
    
    async def _get_session_context(
        self,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get current session context."""
        try:
            if not user_id:
                return {}
            
            session = await self.session_service.get_active_session(user_id)
            if session:
                return {
                    "session_id": session.session_id,
                    "focus": session.context_data.get("focus"),
                    "tasks": session.context_data.get("tasks", [])
                }
        except Exception as e:
            logger.error(f"Error getting session context: {e}")
        
        return {}
    
    async def _get_project_context(
        self,
        project_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get project-specific context."""
        if not project_id:
            return {}
        
        # This would integrate with project analysis service
        return {
            "project_id": project_id,
            "type": "software_project",
            "technologies": []
        }
    
    async def _get_relevant_decisions(
        self,
        prompt: str,
        user_id: Optional[str],
        project_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Get relevant technical decisions."""
        try:
            # Search for relevant decisions
            decisions = await self.memory_service.search_memories(
                query=prompt[:200],
                user_id=user_id,
                project_id=project_id,
                memory_type="decision",
                limit=3
            )
            
            return [
                {
                    "decision": d.content,
                    "reasoning": d.metadata.get("reasoning", ""),
                    "timestamp": d.created_at.isoformat()
                }
                for d in decisions
            ]
        except Exception as e:
            logger.error(f"Error getting relevant decisions: {e}")
            return []
    
    def _build_context_injection(
        self,
        session_context: Dict[str, Any],
        project_context: Dict[str, Any],
        decisions: List[Dict[str, Any]]
    ) -> str:
        """Build context injection string."""
        parts = []
        
        if session_context:
            parts.append(f"Current session focus: {session_context.get('focus', 'Development')}")
        
        if project_context:
            parts.append(f"Project type: {project_context.get('type', 'Unknown')}")
        
        if decisions:
            parts.append("Recent technical decisions:")
            for decision in decisions[:2]:  # Top 2 decisions
                parts.append(f"- {decision['decision'][:100]}...")
        
        if parts:
            return "# KnowledgeHub Context\n" + "\n".join(parts)
        
        return ""
    
    async def _process_feedback(
        self,
        suggestion: CopilotSuggestion,
        feedback_type: str,
        feedback_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process feedback for learning."""
        return {
            "feedback_type": feedback_type,
            "original_confidence": suggestion.confidence,
            "context_sources": suggestion.context_sources,
            "learning_data": suggestion.learning_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _update_models_from_feedback(
        self,
        suggestion: CopilotSuggestion,
        feedback_type: str,
        learning_update: Dict[str, Any]
    ) -> None:
        """Update AI models based on feedback."""
        try:
            # Record feedback as a learning memory
            await self.memory_service.create_memory(
                content=f"Copilot suggestion feedback: {feedback_type}",
                memory_type="learning",
                metadata={
                    "suggestion_confidence": suggestion.confidence,
                    "enhancement_sources": suggestion.context_sources,
                    "feedback_type": feedback_type,
                    "learning_update": learning_update
                }
            )
        except Exception as e:
            logger.error(f"Error updating models from feedback: {e}")
    
    async def _store_feedback_data(
        self,
        suggestion_id: str,
        feedback_type: str,
        feedback_data: Dict[str, Any],
        learning_update: Dict[str, Any]
    ) -> None:
        """Store feedback data for analysis."""
        try:
            async with get_db_session() as session:
                # Store as a metric for analytics
                metric = Metric(
                    name="copilot_feedback",
                    value=1.0,
                    metric_type="counter",
                    tags={
                        "feedback_type": feedback_type,
                        "suggestion_id": suggestion_id
                    },
                    metadata={
                        "feedback_data": feedback_data,
                        "learning_update": learning_update
                    }
                )
                
                session.add(metric)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error storing feedback data: {e}")