#!/usr/bin/env python3
"""MCP Server entry point"""

import asyncio
import logging
import os
from .server import KnowledgeHubMCPServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point"""
    try:
        # Get configuration from environment
        api_url = os.getenv("API_URL", "http://knowledgehub-api:3000")
        port = int(os.getenv("MCP_SERVER_PORT", "3002"))
        
        logger.info(f"Starting MCP Server on port {port}, API URL: {api_url}")
        
        # Create and start server
        server = KnowledgeHubMCPServer(knowledge_api_url=api_url, port=port)
        await server.start()
        
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())