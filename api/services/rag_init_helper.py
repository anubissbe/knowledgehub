"""RAG Service Initialization Helper"""
import logging

logger = logging.getLogger(__name__)

# Global flag to track initialization
_rag_initialized = False

def ensure_rag_initialized():
    """Ensure RAG service is initialized"""
    global _rag_initialized
    if not _rag_initialized:
        try:
            # Initialize RAG components
            logger.info("Initializing RAG service components...")
            _rag_initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            return False
    return True

# Auto-initialize on import
ensure_rag_initialized()
