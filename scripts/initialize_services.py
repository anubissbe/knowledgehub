
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'api.config.settings'

from api.config import settings
from api.services.hybrid_rag_service import get_hybrid_rag_service
from api.services.agent_orchestrator import get_agent_orchestrator
from api.services.zep_memory_integration import ZepMemoryIntegration

async def initialize_services():
    """Initialize all core services"""
    results = {}
    
    # Initialize Hybrid RAG
    try:
        rag_service = await get_hybrid_rag_service()
        await rag_service.initialize()
        results["hybrid_rag"] = "initialized"
    except Exception as e:
        results["hybrid_rag"] = f"error: {str(e)}"
    
    # Initialize Agent Orchestrator
    try:
        orchestrator = await get_agent_orchestrator()
        await orchestrator.initialize()
        results["agent_orchestrator"] = "initialized"
    except Exception as e:
        results["agent_orchestrator"] = f"error: {str(e)}"
    
    # Initialize Zep Memory
    try:
        zep = ZepMemoryIntegration(settings.ZEP_URL, settings.ZEP_API_KEY)
        await zep.initialize()
        results["zep_memory"] = "initialized"
    except Exception as e:
        results["zep_memory"] = f"error: {str(e)}"
    
    return results
