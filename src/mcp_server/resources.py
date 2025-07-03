"""MCP Resources implementation"""

import aiohttp
import json
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class KnowledgeResources:
    """Resources exposed via MCP protocol"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _ensure_session(self):
        """Ensure HTTP session is created"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def get_sources_details(self) -> str:
        """Get detailed information about all knowledge sources"""
        await self._ensure_session()
        
        try:
            async with self.session.get(
                f"{self.api_url}/api/v1/sources"
            ) as response:
                if response.status != 200:
                    return json.dumps({
                        "error": f"Failed to fetch sources: {response.status}"
                    }, indent=2)
                
                data = await response.json()
                sources = data.get("sources", [])
                
                # Enrich with additional details
                detailed_sources = []
                for source in sources:
                    detailed = {
                        "id": source["id"],
                        "name": source["name"],
                        "url": source["url"],
                        "status": source["status"],
                        "statistics": {
                            "documents": source.get("stats", {}).get("documents", 0),
                            "chunks": source.get("stats", {}).get("chunks", 0),
                            "errors": source.get("stats", {}).get("errors", 0)
                        },
                        "timestamps": {
                            "created": source.get("created_at"),
                            "last_updated": source.get("updated_at"),
                            "last_scraped": source.get("last_scraped_at", "Never")
                        },
                        "configuration": source.get("config", {})
                    }
                    detailed_sources.append(detailed)
                
                return json.dumps({
                    "total_sources": len(detailed_sources),
                    "sources": detailed_sources,
                    "fetched_at": datetime.utcnow().isoformat()
                }, indent=2)
                
        except Exception as e:
            logger.error(f"Error fetching sources details: {e}")
            return json.dumps({
                "error": str(e),
                "fetched_at": datetime.utcnow().isoformat()
            }, indent=2)
    
    async def get_system_stats(self) -> str:
        """Get system statistics and metrics"""
        await self._ensure_session()
        
        try:
            # Fetch various statistics
            stats = {
                "timestamp": datetime.utcnow().isoformat(),
                "sources": {},
                "documents": {},
                "search": {},
                "system": {}
            }
            
            # Get sources stats
            async with self.session.get(
                f"{self.api_url}/api/v1/sources"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    sources = data.get("sources", [])
                    
                    stats["sources"] = {
                        "total": len(sources),
                        "by_status": {}
                    }
                    
                    # Count by status
                    for source in sources:
                        status = source.get("status", "unknown")
                        stats["sources"]["by_status"][status] = \
                            stats["sources"]["by_status"].get(status, 0) + 1
                    
                    # Calculate totals
                    total_docs = sum(s.get("stats", {}).get("documents", 0) for s in sources)
                    total_chunks = sum(s.get("stats", {}).get("chunks", 0) for s in sources)
                    
                    stats["documents"] = {
                        "total_documents": total_docs,
                        "total_chunks": total_chunks,
                        "average_chunks_per_document": (
                            round(total_chunks / total_docs, 2) if total_docs > 0 else 0
                        )
                    }
            
            # Get health status
            async with self.session.get(
                f"{self.api_url}/health"
            ) as response:
                if response.status == 200:
                    health_data = await response.json()
                    stats["system"]["health"] = health_data
            
            return json.dumps(stats, indent=2)
            
        except Exception as e:
            logger.error(f"Error fetching system stats: {e}")
            return json.dumps({
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }, indent=2)
    
    async def get_health_status(self) -> str:
        """Get system health status"""
        await self._ensure_session()
        
        try:
            health_status = {
                "timestamp": datetime.utcnow().isoformat(),
                "services": {}
            }
            
            # Check API health
            try:
                async with self.session.get(
                    f"{self.api_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        health_status["services"]["api"] = {
                            "status": "healthy",
                            "details": data
                        }
                    else:
                        health_status["services"]["api"] = {
                            "status": "unhealthy",
                            "error": f"HTTP {response.status}"
                        }
            except Exception as e:
                health_status["services"]["api"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Calculate overall status
            all_healthy = all(
                s.get("status") == "healthy" 
                for s in health_status["services"].values()
            )
            health_status["overall_status"] = "healthy" if all_healthy else "degraded"
            
            return json.dumps(health_status, indent=2)
            
        except Exception as e:
            logger.error(f"Error checking health status: {e}")
            return json.dumps({
                "error": str(e),
                "overall_status": "error",
                "timestamp": datetime.utcnow().isoformat()
            }, indent=2)
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None