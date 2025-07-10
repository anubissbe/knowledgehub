#!/usr/bin/env python3
"""
Claude Code Integration for Unified Memory System
Provides simple functions for Claude Code to use the unified memory system
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add the memory system to path
MEMORY_SYSTEM_PATH = Path(__file__).parent
sys.path.insert(0, str(MEMORY_SYSTEM_PATH))

try:
    from src.unified_memory import UnifiedMemorySystem, create_memory, search_memory, get_memory_context
    from config import load_config
except ImportError as e:
    print(f"Warning: Could not import unified memory system: {e}")
    print("Falling back to basic memory operations")
    
    # Create minimal fallback implementations
    class UnifiedMemorySystem:
        def __init__(self, *args, **kwargs):
            pass
        
        async def add_memory(self, content, memory_type="context", priority="medium", **kwargs):
            return {"primary_id": "fallback", "systems": [{"type": "fallback", "status": "disabled"}]}
        
        async def search_memories(self, query, limit=10, **kwargs):
            return []
        
        async def get_context(self, max_tokens=4000, **kwargs):
            return {"content": "", "token_count": 0, "sources": {"api": 0, "local": 0}}
        
        def get_status(self):
            return {"unified_system": {"api_available": False}, "local_system": {"status": "unavailable"}}
    
    async def create_memory(content, memory_type="context", priority="medium", **kwargs):
        return {"primary_id": "fallback", "systems": []}
    
    async def search_memory(query, limit=10, **kwargs):
        return []
    
    async def get_memory_context(max_tokens=4000, **kwargs):
        return {"content": "", "token_count": 0}

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise for Claude
logger = logging.getLogger(__name__)

# Global memory system instance
_memory_system = None


def _get_memory_system() -> UnifiedMemorySystem:
    """Get or create the global memory system instance"""
    global _memory_system
    if _memory_system is None:
        try:
            config = load_config()
            _memory_system = UnifiedMemorySystem(config)
            logger.info("Unified memory system initialized for Claude Code")
        except Exception as e:
            logger.warning(f"Failed to initialize memory system: {e}")
            _memory_system = UnifiedMemorySystem()  # Fallback
    return _memory_system


# Claude-friendly functions
def log_interaction(query: str, response: str, importance: str = "medium") -> str:
    """
    Log a Claude Code interaction (user query and AI response)
    
    Args:
        query: User's query/request
        response: Claude's response
        importance: Priority level (critical, high, medium, low)
    
    Returns:
        Memory ID of the logged interaction
    """
    try:
        content = f"Query: {query}\n\nResponse: {response}"
        result = asyncio.run(create_memory(
            content=content,
            memory_type="conversation", 
            priority=importance,
            tags=["claude_interaction", "query_response"],
            metadata={
                "interaction_type": "query_response",
                "timestamp": datetime.now().isoformat(),
                "query_length": len(query),
                "response_length": len(response)
            }
        ))
        return result.get("primary_id", "unknown")
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")
        return "error"


def log_code_change(file_path: str, operation: str, content: str, importance: str = "medium") -> str:
    """
    Log a code change made by Claude
    
    Args:
        file_path: Path to the file that was modified
        operation: Type of operation (create, edit, delete)
        content: Code content or change description
        importance: Priority level
    
    Returns:
        Memory ID of the logged code change
    """
    try:
        memory_content = f"File: {file_path}\nOperation: {operation}\n\nContent:\n{content}"
        result = asyncio.run(create_memory(
            content=memory_content,
            memory_type="code",
            priority=importance,
            tags=["code_change", operation, "claude_edit"],
            metadata={
                "file_path": file_path,
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "content_length": len(content)
            }
        ))
        return result.get("primary_id", "unknown")
    except Exception as e:
        logger.error(f"Failed to log code change: {e}")
        return "error"


def log_decision(decision: str, reasoning: str, outcome: str = None, importance: str = "high") -> str:
    """
    Log an important decision made during the session
    
    Args:
        decision: The decision that was made
        reasoning: Why this decision was made
        outcome: Result of the decision (optional)
        importance: Priority level
    
    Returns:
        Memory ID of the logged decision
    """
    try:
        content = f"Decision: {decision}\n\nReasoning: {reasoning}"
        if outcome:
            content += f"\n\nOutcome: {outcome}"
        
        result = asyncio.run(create_memory(
            content=content,
            memory_type="decision",
            priority=importance,
            tags=["decision", "reasoning", "claude_decision"],
            metadata={
                "decision_type": "claude_decision",
                "timestamp": datetime.now().isoformat(),
                "has_outcome": outcome is not None
            }
        ))
        return result.get("primary_id", "unknown")
    except Exception as e:
        logger.error(f"Failed to log decision: {e}")
        return "error"


def log_error(error_msg: str, context: str, solution: str = None, importance: str = "high") -> str:
    """
    Log an error encountered during the session
    
    Args:
        error_msg: The error message
        context: Context where the error occurred
        solution: How the error was resolved (optional)
        importance: Priority level
    
    Returns:
        Memory ID of the logged error
    """
    try:
        content = f"Error: {error_msg}\n\nContext: {context}"
        if solution:
            content += f"\n\nSolution: {solution}"
        
        result = asyncio.run(create_memory(
            content=content,
            memory_type="error",
            priority=importance,
            tags=["error", "troubleshooting", "claude_error"],
            metadata={
                "error_type": "runtime_error",
                "timestamp": datetime.now().isoformat(),
                "has_solution": solution is not None
            }
        ))
        return result.get("primary_id", "unknown")
    except Exception as e:
        logger.error(f"Failed to log error: {e}")
        return "error"


def log_context(context_info: str, category: str = "general", importance: str = "medium") -> str:
    """
    Log important context information
    
    Args:
        context_info: The context information to store
        category: Category of context (architecture, workflow, etc.)
        importance: Priority level
    
    Returns:
        Memory ID of the logged context
    """
    try:
        result = asyncio.run(create_memory(
            content=context_info,
            memory_type="context",
            priority=importance,
            tags=["context", category, "claude_context"],
            metadata={
                "context_category": category,
                "timestamp": datetime.now().isoformat()
            }
        ))
        return result.get("primary_id", "unknown")
    except Exception as e:
        logger.error(f"Failed to log context: {e}")
        return "error"


def search_memories(query: str, limit: int = 10, memory_type: str = None) -> List[Dict[str, Any]]:
    """
    Search the memory system for relevant information
    
    Args:
        query: Search query
        limit: Maximum number of results
        memory_type: Filter by memory type (optional)
    
    Returns:
        List of matching memories
    """
    try:
        results = asyncio.run(search_memory(
            query=query,
            limit=limit,
            use_vector_search=True
        ))
        
        # Filter by memory type if specified
        if memory_type:
            results = [r for r in results if r.get("memory_type", "").lower() == memory_type.lower()]
        
        return results
    except Exception as e:
        logger.error(f"Failed to search memories: {e}")
        return []


def get_relevant_context(query: str = None, max_tokens: int = 4000) -> str:
    """
    Get relevant context for the current session
    
    Args:
        query: Optional query to find relevant context
        max_tokens: Maximum tokens to return
    
    Returns:
        Formatted context string
    """
    try:
        if query:
            # Search for specific context
            search_results = search_memories(query, limit=10)
            if search_results:
                context_parts = []
                for result in search_results[:5]:  # Use top 5 results
                    source = result.get('_source_system', 'unknown')
                    content = result.get('content', '')
                    context_parts.append(f"[{source}] {content}")
                return "\n\n---\n\n".join(context_parts)
        
        # Get general context
        context = asyncio.run(get_memory_context(max_tokens=max_tokens))
        return context.get("content", "")
    
    except Exception as e:
        logger.error(f"Failed to get relevant context: {e}")
        return ""


def create_checkpoint(description: str) -> str:
    """
    Create a checkpoint of the current session state
    
    Args:
        description: Description of the checkpoint
    
    Returns:
        Checkpoint ID
    """
    try:
        system = _get_memory_system()
        result = asyncio.run(system.create_checkpoint(description))
        return result.get("primary_id", "unknown")
    except Exception as e:
        logger.error(f"Failed to create checkpoint: {e}")
        return "error"


def get_session_summary() -> Dict[str, Any]:
    """
    Get a summary of the current memory session
    
    Returns:
        Dictionary with session statistics and information
    """
    try:
        system = _get_memory_system()
        status = system.get_status()
        
        # Add some computed statistics
        local_stats = status.get("local_system", {}).get("stats", {})
        
        summary = {
            "session_id": status.get("unified_system", {}).get("session_id", "unknown"),
            "api_available": status.get("unified_system", {}).get("api_available", False),
            "total_memories": local_stats.get("total_entries", 0),
            "memory_size_mb": local_stats.get("total_size_mb", 0),
            "memory_types": local_stats.get("type_distribution", {}),
            "system_health": {
                "local": status.get("local_system", {}).get("status", "unknown"),
                "api": status.get("api_system", {}).get("status", "unknown")
            }
        }
        
        return summary
    except Exception as e:
        logger.error(f"Failed to get session summary: {e}")
        return {"error": str(e)}


def initialize_claude_session(description: str = None) -> Dict[str, Any]:
    """
    Initialize a Claude Code session with memory system
    
    Args:
        description: Optional description of the session
    
    Returns:
        Dictionary with session initialization info
    """
    try:
        system = _get_memory_system()
        
        # Log session start
        session_desc = description or f"Claude Code session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        log_context(session_desc, "session_start", "high")
        
        # Get system status
        status = system.get_status()
        
        init_info = {
            "session_id": status.get("unified_system", {}).get("session_id"),
            "memory_system_status": "initialized",
            "api_available": status.get("unified_system", {}).get("api_available", False),
            "local_system": status.get("local_system", {}).get("status", "unknown"),
            "existing_memories": status.get("local_system", {}).get("stats", {}).get("total_entries", 0)
        }
        
        return init_info
    except Exception as e:
        logger.error(f"Failed to initialize Claude session: {e}")
        return {"error": str(e), "memory_system_status": "failed"}


# Convenience aliases for backward compatibility
add_memory = log_context
search = search_memories
get_context = get_relevant_context


if __name__ == "__main__":
    # Test the Claude integration
    print("🧪 Testing Claude Code Memory Integration")
    
    # Initialize session
    init_info = initialize_claude_session("Test session for Claude integration")
    print(f"✅ Session initialized: {init_info}")
    
    # Test logging functions
    interaction_id = log_interaction(
        "How do I implement a memory system?",
        "I'll help you implement a unified memory system with both local and API components..."
    )
    print(f"✅ Logged interaction: {interaction_id}")
    
    decision_id = log_decision(
        "Use unified memory approach",
        "This allows fallback from API to local storage for reliability"
    )
    print(f"✅ Logged decision: {decision_id}")
    
    # Test search
    results = search_memories("memory system", limit=5)
    print(f"✅ Search results: {len(results)} found")
    
    # Test context
    context = get_relevant_context("memory implementation")
    print(f"✅ Context retrieved: {len(context)} characters")
    
    # Test summary
    summary = get_session_summary()
    print(f"✅ Session summary: {summary}")
    
    print("\n🎉 Claude Code memory integration is ready!")