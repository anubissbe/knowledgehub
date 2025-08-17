"""
Zep Memory Integration Service
Integrates Zep memory system for enhanced conversation memory and context management
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid

import httpx
from pydantic import BaseModel, Field

from .cache import RedisCache
from .real_ai_intelligence import RealAIIntelligence
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class ZepConfig:
    """Configuration for Zep memory system"""
    api_url: str = "http://localhost:8100"
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_caching: bool = True
    cache_ttl: int = 300


class ZepMessage(BaseModel):
    """Zep message format"""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Message metadata")
    timestamp: Optional[datetime] = Field(None, description="Message timestamp")


class ZepSession(BaseModel):
    """Zep session format"""
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Session metadata")
    created_at: Optional[datetime] = Field(None, description="Session creation time")


class ZepMemorySearch(BaseModel):
    """Zep memory search request"""
    query: str = Field(..., description="Search query")
    session_id: Optional[str] = Field(None, description="Session to search in")
    limit: int = Field(10, description="Maximum results")
    min_relevance_score: float = Field(0.7, description="Minimum relevance score")


class ZepMemoryService:
    """
    Service for integrating with Zep memory system
    
    Features:
    - Conversation memory storage and retrieval
    - Semantic search across conversation history
    - Session management and persistence
    - Context summarization and compression
    - Integration with existing KnowledgeHub memory
    """
    
    def __init__(self, config: Optional[ZepConfig] = None):
        self.config = config or ZepConfig(
            api_url=settings.ZEP_API_URL or "http://localhost:8100",
            api_key=settings.ZEP_API_KEY
        )
        self.logger = logger
        self.cache = RedisCache(settings.REDIS_URL)
        self.ai_intelligence = RealAIIntelligence()
        
        # HTTP client for Zep API
        self.client: Optional[httpx.AsyncClient] = None
        
        # Performance tracking
        self.performance_stats = {
            "requests_made": 0,
            "avg_response_time": 0.0,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0
        }
    
    async def initialize(self):
        """Initialize Zep memory service"""
        try:
            # Initialize cache
            await self.cache.initialize()
            
            # Create HTTP client
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            self.client = httpx.AsyncClient(
                base_url=self.config.api_url,
                headers=headers,
                timeout=self.config.timeout
            )
            
            # Test connection
            await self._health_check()
            
            logger.info("Zep memory service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Zep memory service: {e}")
            # Continue without Zep if initialization fails
            self.client = None
    
    async def create_session(
        self,
        session_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new conversation session in Zep
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            metadata: Optional session metadata
            
        Returns:
            Session creation result
        """
        try:
            if not self.client:
                return {"success": False, "error": "Zep client not initialized"}
            
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "metadata": metadata or {}
            }
            
            response = await self._make_request(
                "POST",
                "/api/v1/sessions",
                json=session_data
            )
            
            if response["success"]:
                await self._track_performance("create_session", 0.0, True)
                return {
                    "success": True,
                    "session_id": session_id,
                    "created_at": datetime.utcnow().isoformat()
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Failed to create Zep session: {e}")
            await self._track_performance("create_session", 0.0, False)
            return {"success": False, "error": str(e)}
    
    async def add_memory(
        self,
        session_id: str,
        messages: List[ZepMessage],
        summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add conversation messages to Zep memory
        
        Args:
            session_id: Session identifier
            messages: List of messages to add
            summary: Optional conversation summary
            
        Returns:
            Memory addition result
        """
        try:
            if not self.client:
                return {"success": False, "error": "Zep client not initialized"}
            
            # Convert messages to Zep format
            zep_messages = []
            for msg in messages:
                zep_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                    "metadata": msg.metadata or {},
                    "timestamp": (msg.timestamp or datetime.utcnow()).isoformat()
                })
            
            memory_data = {
                "messages": zep_messages
            }
            
            if summary:
                memory_data["summary"] = summary
            
            response = await self._make_request(
                "POST",
                f"/api/v1/sessions/{session_id}/memory",
                json=memory_data
            )
            
            if response["success"]:
                await self._track_performance("add_memory", 0.0, True)
                
                # Cache recent memories for quick access
                if self.config.enable_caching:
                    cache_key = f"zep_memory:{session_id}:recent"
                    await self.cache.set(cache_key, messages, ttl=self.config.cache_ttl)
                
                return {
                    "success": True,
                    "messages_added": len(messages),
                    "session_id": session_id
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Failed to add memory to Zep: {e}")
            await self._track_performance("add_memory", 0.0, False)
            return {"success": False, "error": str(e)}
    
    async def get_memory(
        self,
        session_id: str,
        limit: int = 50,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve conversation memory from Zep
        
        Args:
            session_id: Session identifier
            limit: Maximum messages to retrieve
            include_summary: Include conversation summary
            
        Returns:
            Retrieved memory
        """
        try:
            if not self.client:
                return {"success": False, "error": "Zep client not initialized"}
            
            # Check cache first
            if self.config.enable_caching:
                cache_key = f"zep_memory:{session_id}:{limit}"
                cached_memory = await self.cache.get(cache_key)
                if cached_memory:
                    return cached_memory
            
            params = {
                "limit": limit,
                "include_summary": include_summary
            }
            
            response = await self._make_request(
                "GET",
                f"/api/v1/sessions/{session_id}/memory",
                params=params
            )
            
            if response["success"]:
                memory_data = response.get("data", {})
                
                result = {
                    "success": True,
                    "session_id": session_id,
                    "messages": memory_data.get("messages", []),
                    "summary": memory_data.get("summary", ""),
                    "message_count": len(memory_data.get("messages", [])),
                    "retrieved_at": datetime.utcnow().isoformat()
                }
                
                # Cache result
                if self.config.enable_caching:
                    cache_key = f"zep_memory:{session_id}:{limit}"
                    await self.cache.set(cache_key, result, ttl=self.config.cache_ttl)
                
                await self._track_performance("get_memory", 0.0, True)
                return result
            else:
                return response
                
        except Exception as e:
            logger.error(f"Failed to get memory from Zep: {e}")
            await self._track_performance("get_memory", 0.0, False)
            return {"success": False, "error": str(e)}
    
    async def search_memory(
        self,
        search_request: ZepMemorySearch,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Search conversation memory semantically
        
        Args:
            search_request: Search parameters
            user_id: User identifier
            
        Returns:
            Search results
        """
        try:
            if not self.client:
                return {"success": False, "error": "Zep client not initialized"}
            
            search_data = {
                "query": search_request.query,
                "limit": search_request.limit,
                "min_relevance_score": search_request.min_relevance_score
            }
            
            if search_request.session_id:
                endpoint = f"/api/v1/sessions/{search_request.session_id}/search"
            else:
                # Search across all user sessions
                endpoint = f"/api/v1/users/{user_id}/search"
            
            response = await self._make_request(
                "POST",
                endpoint,
                json=search_data
            )
            
            if response["success"]:
                search_results = response.get("data", {})
                
                result = {
                    "success": True,
                    "query": search_request.query,
                    "results": search_results.get("results", []),
                    "total_results": len(search_results.get("results", [])),
                    "search_time": search_results.get("search_time", 0.0),
                    "searched_at": datetime.utcnow().isoformat()
                }
                
                await self._track_performance("search_memory", 0.0, True)
                return result
            else:
                return response
                
        except Exception as e:
            logger.error(f"Failed to search Zep memory: {e}")
            await self._track_performance("search_memory", 0.0, False)
            return {"success": False, "error": str(e)}
    
    async def get_session_summary(
        self,
        session_id: str,
        summary_type: str = "conversational"
    ) -> Dict[str, Any]:
        """
        Get conversation summary from Zep
        
        Args:
            session_id: Session identifier
            summary_type: Type of summary to generate
            
        Returns:
            Session summary
        """
        try:
            if not self.client:
                return {"success": False, "error": "Zep client not initialized"}
            
            params = {"type": summary_type}
            
            response = await self._make_request(
                "GET",
                f"/api/v1/sessions/{session_id}/summary",
                params=params
            )
            
            if response["success"]:
                summary_data = response.get("data", {})
                
                result = {
                    "success": True,
                    "session_id": session_id,
                    "summary": summary_data.get("summary", ""),
                    "summary_type": summary_type,
                    "generated_at": datetime.utcnow().isoformat(),
                    "metadata": summary_data.get("metadata", {})
                }
                
                await self._track_performance("get_summary", 0.0, True)
                return result
            else:
                return response
                
        except Exception as e:
            logger.error(f"Failed to get Zep summary: {e}")
            await self._track_performance("get_summary", 0.0, False)
            return {"success": False, "error": str(e)}
    
    async def delete_session(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Delete a session and all its memory
        
        Args:
            session_id: Session identifier
            
        Returns:
            Deletion result
        """
        try:
            if not self.client:
                return {"success": False, "error": "Zep client not initialized"}
            
            response = await self._make_request(
                "DELETE",
                f"/api/v1/sessions/{session_id}"
            )
            
            if response["success"]:
                # Clear related cache entries
                if self.config.enable_caching:
                    cache_keys = [
                        f"zep_memory:{session_id}:*",
                        f"zep_session:{session_id}"
                    ]
                    # Note: In practice, you'd implement cache key pattern deletion
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "deleted_at": datetime.utcnow().isoformat()
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Failed to delete Zep session: {e}")
            return {"success": False, "error": str(e)}
    
    async def classify_message(
        self,
        message: str,
        classes: List[str]
    ) -> Dict[str, Any]:
        """
        Classify a message using Zep's classification
        
        Args:
            message: Message to classify
            classes: List of possible classes
            
        Returns:
            Classification result
        """
        try:
            if not self.client:
                return {"success": False, "error": "Zep client not initialized"}
            
            classification_data = {
                "message": message,
                "classes": classes
            }
            
            response = await self._make_request(
                "POST",
                "/api/v1/classify",
                json=classification_data
            )
            
            if response["success"]:
                classification = response.get("data", {})
                
                result = {
                    "success": True,
                    "message": message,
                    "predicted_class": classification.get("class", ""),
                    "confidence": classification.get("confidence", 0.0),
                    "all_scores": classification.get("scores", {}),
                    "classified_at": datetime.utcnow().isoformat()
                }
                
                return result
            else:
                return response
                
        except Exception as e:
            logger.error(f"Failed to classify message with Zep: {e}")
            return {"success": False, "error": str(e)}
    
    async def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract entities from text using Zep
        
        Args:
            text: Text to extract entities from
            entity_types: Optional list of entity types to extract
            
        Returns:
            Entity extraction result
        """
        try:
            if not self.client:
                return {"success": False, "error": "Zep client not initialized"}
            
            extraction_data = {
                "text": text
            }
            
            if entity_types:
                extraction_data["entity_types"] = entity_types
            
            response = await self._make_request(
                "POST",
                "/api/v1/extract",
                json=extraction_data
            )
            
            if response["success"]:
                entities = response.get("data", {})
                
                result = {
                    "success": True,
                    "text": text,
                    "entities": entities.get("entities", []),
                    "entity_count": len(entities.get("entities", [])),
                    "extracted_at": datetime.utcnow().isoformat()
                }
                
                return result
            else:
                return response
                
        except Exception as e:
            logger.error(f"Failed to extract entities with Zep: {e}")
            return {"success": False, "error": str(e)}
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to Zep API with retry logic"""
        if not self.client:
            return {"success": False, "error": "Zep client not initialized"}
        
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                
                response = await self.client.request(
                    method=method,
                    url=endpoint,
                    json=json,
                    params=params
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "data": data,
                        "response_time": response_time
                    }
                else:
                    error_msg = f"Zep API error {response.status_code}: {response.text}"
                    if attempt == self.config.max_retries - 1:
                        return {"success": False, "error": error_msg}
                    
                    # Retry on server errors
                    if response.status_code >= 500:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    else:
                        return {"success": False, "error": error_msg}
                        
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    return {"success": False, "error": str(e)}
                
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        return {"success": False, "error": "Max retries exceeded"}
    
    async def _health_check(self) -> bool:
        """Check if Zep service is healthy"""
        try:
            if not self.client:
                return False
            
            response = await self.client.get("/healthz")
            return response.status_code == 200
            
        except Exception as e:
            logger.warning(f"Zep health check failed: {e}")
            return False
    
    async def _track_performance(
        self,
        operation: str,
        response_time: float,
        success: bool
    ):
        """Track operation performance"""
        try:
            self.performance_stats["requests_made"] += 1
            
            # Update average response time
            total_requests = self.performance_stats["requests_made"]
            old_avg = self.performance_stats["avg_response_time"]
            self.performance_stats["avg_response_time"] = (
                (old_avg * (total_requests - 1) + response_time) / total_requests
            )
            
            # Update error rate
            if not success:
                current_errors = self.performance_stats["error_rate"] * (total_requests - 1)
                self.performance_stats["error_rate"] = (current_errors + 1) / total_requests
            
            # Track in AI intelligence system
            await self.ai_intelligence.track_performance_metric(
                f"zep_{operation}",
                execution_time=response_time,
                success=success,
                metadata={"operation": operation}
            )
            
        except Exception as e:
            logger.error(f"Failed to track Zep performance: {e}")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get Zep service performance statistics"""
        return {
            **self.performance_stats,
            "service_status": "connected" if self.client else "disconnected",
            "config": {
                "api_url": self.config.api_url,
                "enable_caching": self.config.enable_caching,
                "timeout": self.config.timeout
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Zep service health"""
        health = {
            "status": "unknown",
            "timestamp": datetime.utcnow().isoformat(),
            "details": {}
        }
        
        if not self.client:
            health["status"] = "disconnected"
            health["details"]["client"] = "not_initialized"
        else:
            is_healthy = await self._health_check()
            health["status"] = "healthy" if is_healthy else "unhealthy"
            health["details"]["api_connection"] = "ok" if is_healthy else "failed"
        
        health["details"]["performance"] = self.performance_stats
        
        return health
    
    async def close(self):
        """Close Zep service connections"""
        if self.client:
            await self.client.aclose()
            self.client = None


# Global instance
_zep_memory_service: Optional[ZepMemoryService] = None


async def get_zep_memory_service() -> ZepMemoryService:
    """Get singleton Zep memory service instance"""
    global _zep_memory_service
    
    if _zep_memory_service is None:
        _zep_memory_service = ZepMemoryService()
        await _zep_memory_service.initialize()
    
    return _zep_memory_service