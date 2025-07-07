"""Integration script for memory system with KnowledgeHub API"""

from fastapi import FastAPI
from ..api.main import app as main_app
from .api.routers import memory, session


def integrate_memory_system(app: FastAPI):
    """Integrate memory system endpoints into main API"""
    
    # Include memory system routers
    app.include_router(
        session.router,
        prefix="/api/memory/session",
        tags=["memory-session"]
    )
    
    app.include_router(
        memory.router,
        prefix="/api/memory/memories",
        tags=["memory"]
    )
    
    # Add to API documentation
    @app.get("/api/memory", tags=["memory"])
    async def memory_system_info():
        """Get memory system information"""
        return {
            "name": "Claude Memory System",
            "version": "1.0.0",
            "status": "operational",
            "endpoints": {
                "sessions": {
                    "start": "/api/memory/session/start",
                    "get": "/api/memory/session/{session_id}",
                    "end": "/api/memory/session/{session_id}/end",
                    "user_sessions": "/api/memory/session/user/{user_id}"
                },
                "memories": {
                    "create": "/api/memory/memories/",
                    "get": "/api/memory/memories/{memory_id}",
                    "search": "/api/memory/memories/search",
                    "batch": "/api/memory/memories/batch"
                }
            }
        }
    
    print("✅ Memory system integrated with KnowledgeHub API")


# Call this from main.py or during startup
if __name__ == "__main__":
    integrate_memory_system(main_app)