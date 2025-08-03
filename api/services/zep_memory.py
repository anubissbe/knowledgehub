"""
Zep Memory Service
Implements conversational memory with temporal knowledge graphs
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from uuid import uuid4

try:
    from zep_python import AsyncZep, Message, Memory, SearchResult
    ZEP_AVAILABLE = True
except ImportError:
    ZEP_AVAILABLE = False
    AsyncZep = None
    Message = None
    Memory = None
    SearchResult = None

from ..config import settings
from ..services.cache import CacheService
from ..services.real_ai_intelligence import RealAIIntelligence

logger = logging.getLogger(__name__)


class ZepMemoryService:
    """
    Service for managing conversational memory using Zep
    Provides temporal knowledge graphs and conversation persistence
    """
    
    def __init__(self):
        self.logger = logger
        self.cache = CacheService()
        self.ai_intelligence = RealAIIntelligence()
        
        # Initialize Zep client if available
        self.zep_client = None
        self.enabled = False
        
        if ZEP_AVAILABLE and settings.ZEP_API_URL:
            try:
                self.zep_client = AsyncZep(
                    api_url=settings.ZEP_API_URL,
                    api_key=settings.ZEP_API_KEY
                )
                self.enabled = True
                logger.info("Zep memory service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Zep client: {e}")
                self.enabled = False
        else:
            logger.warning("Zep not available or not configured")
            
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a message to conversation memory
        
        Args:
            session_id: Unique session identifier
            role: Message role (user, assistant, system)
            content: Message content
            user_id: Optional user identifier
            metadata: Optional message metadata
            
        Returns:
            Success status
        """
        if not self.enabled:
            # Fallback to cache-based memory
            return await self._add_message_to_cache(
                session_id, role, content, user_id, metadata
            )
            
        try:
            # Create Zep message
            message = Message(
                role=role,
                content=content,
                metadata=metadata or {}
            )
            
            if user_id:
                message.metadata["user_id"] = user_id
                
            # Add timestamp
            message.metadata["timestamp"] = datetime.utcnow().isoformat()
            
            # Add to Zep
            await self.zep_client.memory.add_memory(
                session_id=session_id,
                messages=[message]
            )
            
            # Track in AI intelligence
            await self.ai_intelligence.track_session_event(
                user_id=user_id or "unknown",
                event_type="message_added",
                event_data={
                    "session_id": session_id,
                    "role": role,
                    "content_length": len(content)
                }
            )
            
            logger.info(f"Added message to session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message to Zep: {e}")
            # Fallback to cache
            return await self._add_message_to_cache(
                session_id, role, content, user_id, metadata
            )
            
    async def get_memory(
        self,
        session_id: str,
        limit: int = 10,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Get conversation memory for a session
        
        Args:
            session_id: Session identifier
            limit: Maximum messages to return
            include_summary: Whether to include conversation summary
            
        Returns:
            Memory data including messages and summary
        """
        if not self.enabled:
            return await self._get_memory_from_cache(session_id, limit)
            
        try:
            # Get memory from Zep
            memory = await self.zep_client.memory.get_memory(
                session_id=session_id,
                limit=limit
            )
            
            result = {
                "session_id": session_id,
                "messages": [],
                "summary": None,
                "facts": [],
                "entities": []
            }
            
            if memory:
                # Extract messages
                result["messages"] = [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "metadata": msg.metadata,
                        "timestamp": msg.created_at
                    }
                    for msg in memory.messages
                ]
                
                # Include summary if available
                if include_summary and hasattr(memory, 'summary'):
                    result["summary"] = memory.summary
                    
                # Extract facts if available
                if hasattr(memory, 'facts'):
                    result["facts"] = memory.facts
                    
                # Extract entities if available
                if hasattr(memory, 'entities'):
                    result["entities"] = memory.entities
                    
            return result
            
        except Exception as e:
            logger.error(f"Failed to get memory from Zep: {e}")
            return await self._get_memory_from_cache(session_id, limit)
            
    async def search_memory(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search across conversation memories
        
        Args:
            query: Search query
            user_id: Optional user filter
            session_ids: Optional session filter
            limit: Maximum results
            
        Returns:
            List of search results
        """
        if not self.enabled:
            return await self._search_memory_in_cache(query, user_id, limit)
            
        try:
            # Build search parameters
            search_params = {
                "query": query,
                "limit": limit
            }
            
            if user_id:
                search_params["metadata"] = {"user_id": user_id}
                
            if session_ids:
                search_params["session_ids"] = session_ids
                
            # Search in Zep
            results = await self.zep_client.memory.search_memory(**search_params)
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "session_id": result.session_id,
                    "message": {
                        "role": result.message.role,
                        "content": result.message.content,
                        "metadata": result.message.metadata
                    },
                    "score": result.score,
                    "summary": getattr(result, 'summary', None)
                })
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search memory in Zep: {e}")
            return await self._search_memory_in_cache(query, user_id, limit)
            
    async def create_or_update_user(
        self,
        user_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Create or update user profile in Zep
        
        Args:
            user_id: User identifier
            metadata: User metadata
            
        Returns:
            Success status
        """
        if not self.enabled:
            return False
            
        try:
            await self.zep_client.user.add_user(
                user_id=user_id,
                metadata=metadata
            )
            logger.info(f"Created/updated user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create/update user in Zep: {e}")
            return False
            
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[str]:
        """
        Get all sessions for a user
        
        Args:
            user_id: User identifier
            limit: Maximum sessions
            
        Returns:
            List of session IDs
        """
        if not self.enabled:
            return []
            
        try:
            # Search for all messages from user
            results = await self.search_memory(
                query="*",  # Match all
                user_id=user_id,
                limit=limit * 10  # Get more to extract unique sessions
            )
            
            # Extract unique session IDs
            session_ids = list(set(
                result["session_id"] 
                for result in results
            ))
            
            return session_ids[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []
            
    # Fallback cache-based implementation
    async def _add_message_to_cache(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Fallback: Add message to cache"""
        try:
            # Get existing messages
            cache_key = f"zep_session:{session_id}"
            messages = await self.cache.get(cache_key) or []
            
            # Add new message
            messages.append({
                "id": str(uuid4()),
                "role": role,
                "content": content,
                "user_id": user_id,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Keep only recent messages
            if len(messages) > 100:
                messages = messages[-100:]
                
            # Save to cache
            await self.cache.set(cache_key, messages, ttl=86400)  # 24 hours
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message to cache: {e}")
            return False
            
    async def _get_memory_from_cache(
        self,
        session_id: str,
        limit: int
    ) -> Dict[str, Any]:
        """Fallback: Get memory from cache"""
        try:
            cache_key = f"zep_session:{session_id}"
            messages = await self.cache.get(cache_key) or []
            
            # Get recent messages
            recent_messages = messages[-limit:] if len(messages) > limit else messages
            
            return {
                "session_id": session_id,
                "messages": recent_messages,
                "summary": None,
                "facts": [],
                "entities": []
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory from cache: {e}")
            return {
                "session_id": session_id,
                "messages": [],
                "summary": None,
                "facts": [],
                "entities": []
            }
            
    async def _search_memory_in_cache(
        self,
        query: str,
        user_id: Optional[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Fallback: Search memory in cache"""
        # Simple implementation - in production, use proper search
        return []
        
    async def hybrid_retrieval(
        self,
        query: str,
        user_id: str,
        rag_results: List[Dict[str, Any]],
        weight_memory: float = 0.3,
        weight_rag: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Combine RAG results with conversation memory
        
        Args:
            query: Search query
            user_id: User identifier
            rag_results: Results from RAG system
            weight_memory: Weight for memory results
            weight_rag: Weight for RAG results
            
        Returns:
            Combined and re-ranked results
        """
        # Get memory results
        memory_results = await self.search_memory(
            query=query,
            user_id=user_id,
            limit=5
        )
        
        # Combine results
        combined_results = []
        
        # Add RAG results with weight
        for result in rag_results:
            combined_results.append({
                "source": "rag",
                "content": result.get("content", ""),
                "score": result.get("score", 0.5) * weight_rag,
                "metadata": result.get("metadata", {})
            })
            
        # Add memory results with weight
        for result in memory_results:
            combined_results.append({
                "source": "memory",
                "content": result["message"]["content"],
                "score": result.get("score", 0.5) * weight_memory,
                "metadata": {
                    "session_id": result["session_id"],
                    "role": result["message"]["role"],
                    **result["message"].get("metadata", {})
                }
            })
            
        # Sort by combined score
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Remove duplicates (simple content-based)
        seen_content = set()
        unique_results = []
        
        for result in combined_results:
            content_hash = hash(result["content"][:100])  # Hash first 100 chars
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
                
        return unique_results


# Singleton instance
_zep_service = None


def get_zep_service() -> ZepMemoryService:
    """Get singleton Zep service instance"""
    global _zep_service
    if _zep_service is None:
        _zep_service = ZepMemoryService()
    return _zep_service