"""Simple health check server for RAG processor"""

import asyncio
import os
from aiohttp import web
import logging

logger = logging.getLogger(__name__)

class HealthServer:
    """Simple HTTP server for health checks"""
    
    def __init__(self, port: int = 3013):
        self.port = port
        self.app = web.Application()
        self.app.router.add_get('/health', self.health_check)
        self.runner = None
        
    async def health_check(self, request):
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "service": "rag-processor",
            "timestamp": asyncio.get_event_loop().time()
        })
    
    async def start(self):
        """Start the health server"""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            site = web.TCPSite(self.runner, '0.0.0.0', self.port)
            await site.start()
            logger.info(f"Health server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start health server: {e}")
    
    async def stop(self):
        """Stop the health server"""
        if self.runner:
            await self.runner.cleanup()
            logger.info("Health server stopped")