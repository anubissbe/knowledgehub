"""
Persistent Context Service

High-level service for integrating persistent context with
existing memory system components and providing unified access.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Session

from ..core.persistent_context import (
    PersistentContextManager,
    get_persistent_context_manager,
    ContextType,
    ContextScope,
    ContextVector
)
from ..core.session_manager import SessionManager
from ..models import Memory, MemorySession, MemoryType
from .embedding_service import MemoryEmbeddingService
from .fact_extraction import FactExtractionService
from .entity_extraction import EntityExtractionService
from .importance_scoring import ImportanceScoringService

logger = logging.getLogger(__name__)


class PersistentContextService:
    """
    High-level service for persistent context management
    
    This service provides a unified interface for working with
    persistent context, integrating with existing memory system
    components and providing enhanced functionality.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.context_manager = get_persistent_context_manager(db)
        self.session_manager = SessionManager(db)
        self.embedding_service = MemoryEmbeddingService()
        self.fact_extraction = FactExtractionService()
        self.entity_extraction = EntityExtractionService()
        self.importance_scoring = ImportanceScoringService()
    
    async def process_session_for_context(self, session_id: UUID) -> Dict[str, Any]:
        """
        Process a completed session to extract persistent context
        
        Analyzes session memories and extracts valuable context
        that should persist beyond the current session.
        """
        try:
            # Get session and its memories
            session = self.db.query(MemorySession).filter(
                MemorySession.id == session_id
            ).first()
            
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            memories = self.db.query(Memory).filter(
                Memory.session_id == session_id
            ).order_by(Memory.importance_score.desc()).all()
            
            if not memories:
                logger.info(f"No memories found for session {session_id}")
                return {"context_vectors_created": 0, "message": "No memories to process"}
            
            # Process memories for persistent context
            context_vectors_created = 0
            
            for memory in memories:
                # Only process high-importance memories
                if memory.importance_score < 0.6:
                    continue
                
                # Determine context type based on memory content
                context_type = await self._analyze_context_type(memory)
                
                # Determine scope based on session context
                scope = await self._determine_context_scope(session, memory)
                
                # Extract additional entities and facts
                entities = await self._extract_enhanced_entities(memory.content)
                facts = await self._extract_enhanced_facts(memory.content)
                
                # Create enhanced metadata
                metadata = {
                    "original_memory_id": str(memory.id),
                    "session_id": str(session_id),
                    "session_title": session.title,
                    "memory_type": memory.memory_type.value,
                    "original_importance": memory.importance_score,
                    "extracted_facts": facts,
                    "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "session_processing"
                }
                
                # Add to persistent context
                context_id = await self.context_manager.add_context(
                    content=memory.content,
                    context_type=context_type,
                    scope=scope,
                    importance=memory.importance_score,
                    related_entities=entities,
                    metadata=metadata
                )
                
                if context_id:
                    context_vectors_created += 1
                    logger.info(f"Created context vector {context_id} from memory {memory.id}")
            
            return {
                "context_vectors_created": context_vectors_created,
                "session_id": str(session_id),
                "memories_processed": len(memories),
                "message": f"Successfully processed {context_vectors_created} memories into persistent context"
            }
            
        except Exception as e:
            logger.error(f"Error processing session {session_id} for context: {e}")
            raise
    
    async def _analyze_context_type(self, memory: Memory) -> ContextType:
        """Analyze memory content to determine appropriate context type"""
        try:
            content = memory.content.lower()
            
            # Check for code patterns
            if any(keyword in content for keyword in ["function", "class", "import", "def", "async", "await", "python", "javascript"]):
                return ContextType.TECHNICAL_KNOWLEDGE
            
            # Check for decision patterns
            if any(keyword in content for keyword in ["decided", "chose", "selected", "option", "approach"]):
                return ContextType.DECISIONS
            
            # Check for preference patterns
            if any(keyword in content for keyword in ["prefer", "like", "want", "need", "should"]):
                return ContextType.PREFERENCES
            
            # Check for workflow patterns
            if any(keyword in content for keyword in ["process", "workflow", "steps", "procedure", "method"]):
                return ContextType.WORKFLOWS
            
            # Check for learning patterns
            if any(keyword in content for keyword in ["learned", "discovered", "found", "realized", "understood"]):
                return ContextType.LEARNINGS
            
            # Check for pattern recognition
            if any(keyword in content for keyword in ["pattern", "recurring", "always", "often", "usually"]):
                return ContextType.PATTERNS
            
            # Default based on memory type
            if memory.memory_type == MemoryType.FACT:
                return ContextType.TECHNICAL_KNOWLEDGE
            elif memory.memory_type == MemoryType.PREFERENCE:
                return ContextType.PREFERENCES
            elif memory.memory_type == MemoryType.DECISION:
                return ContextType.DECISIONS
            elif memory.memory_type == MemoryType.PATTERN:
                return ContextType.PATTERNS
            else:
                return ContextType.CONVERSATION_FLOW
                
        except Exception as e:
            logger.warning(f"Error analyzing context type: {e}")
            return ContextType.CONVERSATION_FLOW
    
    async def _determine_context_scope(self, session: MemorySession, memory: Memory) -> ContextScope:
        """Determine the appropriate scope for context"""
        try:
            # Check if memory is project-specific
            if session.project_id:
                return ContextScope.PROJECT
            
            # Check if memory is user-specific
            if session.user_id:
                return ContextScope.USER
            
            # Check for domain-specific knowledge
            content = memory.content.lower()
            if any(domain in content for domain in ["python", "javascript", "docker", "kubernetes", "sql", "react"]):
                return ContextScope.DOMAIN
            
            # Default to session scope
            return ContextScope.SESSION
            
        except Exception as e:
            logger.warning(f"Error determining context scope: {e}")
            return ContextScope.SESSION
    
    async def _extract_enhanced_entities(self, content: str) -> List[str]:
        """Extract enhanced entities from content"""
        try:
            # Use existing entity extraction service
            entities = await self.entity_extraction.extract_entities(content)
            
            # Add additional entity types specific to persistent context
            enhanced_entities = entities.copy()
            
            # Extract technology names
            tech_patterns = [
                "python", "javascript", "typescript", "react", "vue", "angular",
                "docker", "kubernetes", "sql", "postgresql", "redis", "mongodb",
                "fastapi", "flask", "django", "express", "nodejs", "npm", "pip"
            ]
            
            content_lower = content.lower()
            for tech in tech_patterns:
                if tech in content_lower:
                    enhanced_entities.append(f"technology:{tech}")
            
            # Extract file extensions
            import re
            file_patterns = re.findall(r'\.([a-zA-Z0-9]+)', content)
            for ext in file_patterns:
                if ext in ["py", "js", "ts", "jsx", "tsx", "html", "css", "json", "yaml", "yml", "md"]:
                    enhanced_entities.append(f"file_type:{ext}")
            
            return list(set(enhanced_entities))  # Remove duplicates
            
        except Exception as e:
            logger.warning(f"Error extracting enhanced entities: {e}")
            return []
    
    async def _extract_enhanced_facts(self, content: str) -> List[str]:
        """Extract enhanced facts from content"""
        try:
            # Use existing fact extraction service
            facts = await self.fact_extraction.extract_facts(content)
            
            # Add additional fact types for persistent context
            enhanced_facts = facts.copy()
            
            # Extract configuration facts
            if "config" in content.lower():
                enhanced_facts.append("Contains configuration information")
            
            # Extract error facts
            if any(error_word in content.lower() for error_word in ["error", "exception", "failed", "bug"]):
                enhanced_facts.append("Contains error or debugging information")
            
            # Extract solution facts
            if any(solution_word in content.lower() for solution_word in ["solution", "fix", "resolved", "solved"]):
                enhanced_facts.append("Contains solution or resolution")
            
            return enhanced_facts
            
        except Exception as e:
            logger.warning(f"Error extracting enhanced facts: {e}")
            return []
    
    async def query_context_with_session_awareness(self, query: str, 
                                                 session_id: Optional[UUID] = None,
                                                 project_id: Optional[str] = None,
                                                 user_id: Optional[str] = None,
                                                 limit: int = 10) -> List[ContextVector]:
        """
        Query persistent context with session awareness
        
        Provides context queries that are aware of current session
        and can prioritize relevant context accordingly.
        """
        try:
            # Get base context results
            base_results = await self.context_manager.retrieve_context(
                query=query,
                limit=limit * 2  # Get more results for filtering
            )
            
            # Score and filter results based on session context
            scored_results = []
            
            for vector in base_results:
                score = 0.0
                
                # Base similarity score (already computed)
                score += vector.importance
                
                # Boost for session relevance
                if session_id and vector.metadata.get("session_id") == str(session_id):
                    score += 0.3
                
                # Boost for project relevance
                if project_id and vector.scope == ContextScope.PROJECT:
                    score += 0.2
                
                # Boost for user relevance
                if user_id and vector.scope == ContextScope.USER:
                    score += 0.1
                
                # Boost for recent access
                time_since_access = datetime.now(timezone.utc) - vector.last_accessed
                if time_since_access.days < 7:
                    score += 0.1
                
                # Boost for high access count
                if vector.access_count > 5:
                    score += 0.1
                
                scored_results.append((vector, score))
            
            # Sort by score and return top results
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            return [vector for vector, score in scored_results[:limit]]
            
        except Exception as e:
            logger.error(f"Error querying context with session awareness: {e}")
            return []
    
    async def get_context_recommendations(self, session_id: UUID) -> List[Dict[str, Any]]:
        """
        Get context recommendations for a session
        
        Recommends relevant persistent context that might be
        useful for the current session.
        """
        try:
            # Get session information
            session = self.db.query(MemorySession).filter(
                MemorySession.id == session_id
            ).first()
            
            if not session:
                return []
            
            # Get recent memories from session to understand context
            recent_memories = self.db.query(Memory).filter(
                Memory.session_id == session_id
            ).order_by(Memory.created_at.desc()).limit(5).all()
            
            if not recent_memories:
                return []
            
            # Create query from recent memories
            query_content = " ".join([memory.content for memory in recent_memories])
            
            # Get relevant context
            relevant_context = await self.query_context_with_session_awareness(
                query=query_content,
                session_id=session_id,
                project_id=session.project_id,
                user_id=session.user_id,
                limit=5
            )
            
            # Format recommendations
            recommendations = []
            for vector in relevant_context:
                recommendations.append({
                    "id": str(vector.id),
                    "content": vector.content[:200] + "..." if len(vector.content) > 200 else vector.content,
                    "context_type": vector.context_type.value,
                    "scope": vector.scope.value,
                    "importance": vector.importance,
                    "relevance_reason": self._get_relevance_reason(vector, session),
                    "last_accessed": vector.last_accessed.isoformat(),
                    "access_count": vector.access_count
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting context recommendations: {e}")
            return []
    
    def _get_relevance_reason(self, vector: ContextVector, session: MemorySession) -> str:
        """Get explanation for why context is relevant"""
        reasons = []
        
        if vector.metadata.get("session_id") == str(session.id):
            reasons.append("from current session")
        
        if vector.scope == ContextScope.PROJECT and session.project_id:
            reasons.append("related to current project")
        
        if vector.scope == ContextScope.USER and session.user_id:
            reasons.append("personalized for user")
        
        if vector.context_type == ContextType.TECHNICAL_KNOWLEDGE:
            reasons.append("technical knowledge")
        
        if vector.access_count > 5:
            reasons.append("frequently accessed")
        
        if not reasons:
            reasons.append("semantically similar")
        
        return ", ".join(reasons)
    
    async def export_context_for_session(self, session_id: UUID) -> Dict[str, Any]:
        """
        Export all persistent context related to a session
        
        Useful for backing up or transferring context data.
        """
        try:
            # Get all context vectors related to this session
            session_vectors = []
            
            for vector in self.context_manager.context_graph.nodes.values():
                if vector.metadata.get("session_id") == str(session_id):
                    session_vectors.append({
                        "id": str(vector.id),
                        "content": vector.content,
                        "context_type": vector.context_type.value,
                        "scope": vector.scope.value,
                        "importance": vector.importance,
                        "last_accessed": vector.last_accessed.isoformat(),
                        "access_count": vector.access_count,
                        "related_entities": vector.related_entities,
                        "metadata": vector.metadata
                    })
            
            return {
                "session_id": str(session_id),
                "context_vectors": session_vectors,
                "total_count": len(session_vectors),
                "export_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error exporting context for session {session_id}: {e}")
            raise
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get health status of the persistent context service"""
        try:
            analytics = await self.context_manager.get_context_analytics()
            
            return {
                "service_status": "healthy",
                "context_manager": "operational",
                "total_vectors": analytics.get("total_vectors", 0),
                "total_clusters": analytics.get("total_clusters", 0),
                "avg_importance": analytics.get("avg_importance", 0.0),
                "memory_usage": analytics.get("memory_usage", 0),
                "last_check": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting service health: {e}")
            return {
                "service_status": "error",
                "error": str(e),
                "last_check": datetime.now(timezone.utc).isoformat()
            }