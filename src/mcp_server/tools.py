"""MCP Tools implementation"""

import aiohttp
import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class KnowledgeTools:
    """Tools exposed via MCP protocol"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _ensure_session(self):
        """Ensure HTTP session is created"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def search(
        self,
        query: str,
        source_filter: Optional[str] = None,
        limit: int = 10,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Execute hybrid search against the knowledge base"""
        await self._ensure_session()
        
        payload = {
            "query": query,
            "limit": limit,
            "include_metadata": include_metadata,
            "search_type": "hybrid"
        }
        
        if source_filter:
            payload["source_filter"] = source_filter
        
        try:
            if self.session is None:
                raise Exception("Session not initialized")
            async with self.session.post(
                f"{self.api_url}/api/v1/search",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Search failed: {response.status} - {error_text}")
                
                data = await response.json()
                
                # Format results for AI consumption
                formatted_results = []
                for result in data.get("results", []):
                    formatted_results.append({
                        "content": result["content"],
                        "source": result.get("source_name", "Unknown"),
                        "url": result.get("url", ""),
                        "score": result.get("score", 0.0),
                        "chunk_type": result.get("chunk_type", "text"),
                        "metadata": result.get("metadata", {}) if include_metadata else None
                    })
                
                return {
                    "query": query,
                    "count": len(formatted_results),
                    "results": formatted_results,
                    "search_time_ms": data.get("search_time_ms", 0)
                }
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during search: {e}")
            raise Exception(f"Search request failed: {str(e)}")
    
    async def store_memory(
        self,
        content: str,
        tags: List[str],
        metadata: Dict[str, Any]
    ) -> str:
        """Store a memory item in the knowledge base"""
        await self._ensure_session()
        
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        payload = {
            "content": content,
            "content_hash": content_hash,
            "tags": tags,
            "metadata": {
                **metadata,
                "stored_at": datetime.utcnow().isoformat(),
                "stored_by": "mcp_client"
            }
        }
        
        try:
            if self.session is None:
                raise Exception("Session not initialized")
            async with self.session.post(
                f"{self.api_url}/api/v1/memories",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 201:
                    error_text = await response.text()
                    raise Exception(f"Store memory failed: {response.status} - {error_text}")
                
                data = await response.json()
                return data["id"]
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error storing memory: {e}")
            raise Exception(f"Store memory request failed: {str(e)}")
    
    async def get_context(
        self,
        query: str,
        max_tokens: int = 4000,
        recency_weight: float = 0.1
    ) -> str:
        """Get relevant context optimized for AI consumption"""
        # First, perform a search
        search_results = await self.search(query, limit=20)
        
        # Build context within token limit
        context_parts = [
            f"# Relevant Context for: {query}",
            f"# Found {search_results['count']} relevant results\n"
        ]
        
        current_tokens = 100  # Rough estimate for header
        
        for i, result in enumerate(search_results["results"]):
            # Rough token estimation (1 token ≈ 4 chars)
            result_text = (
                f"\n## [{i+1}] {result['source']} ({result['chunk_type']})\n"
                f"URL: {result['url']}\n"
                f"Score: {result['score']:.3f}\n\n"
                f"{result['content']}\n"
                f"{'-' * 40}\n"
            )
            result_tokens = len(result_text) // 4
            
            if current_tokens + result_tokens > max_tokens:
                context_parts.append(
                    f"\n[... {len(search_results['results']) - i} more results truncated due to token limit ...]"
                )
                break
            
            context_parts.append(result_text)
            current_tokens += result_tokens
        
        return "\n".join(context_parts)
    
    async def list_sources(self) -> List[Dict[str, Any]]:
        """List all knowledge sources"""
        await self._ensure_session()
        
        try:
            if self.session is None:
                raise Exception("Session not initialized")
            async with self.session.get(
                f"{self.api_url}/api/v1/sources"
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"List sources failed: {response.status} - {error_text}")
                
                data = await response.json()
                
                return [{
                    "name": source["name"],
                    "url": source["url"],
                    "status": source["status"],
                    "documents": source.get("stats", {}).get("documents", 0),
                    "chunks": source.get("stats", {}).get("chunks", 0),
                    "last_updated": source.get("last_scraped_at", "Never"),
                    "created_at": source.get("created_at")
                } for source in data.get("sources", [])]
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error listing sources: {e}")
            raise Exception(f"List sources request failed: {str(e)}")
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None