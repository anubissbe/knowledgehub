"""
Tool Registry and Definitions for KnowledgeHub MCP Server.

This module defines all available tools that Claude Code can use to interact
with KnowledgeHub's AI-enhanced systems.

Tool Categories:
- Memory Tools: Create, search, retrieve, and manage memories
- Session Tools: Session management and continuity
- AI Tools: AI intelligence features and learning systems
- Analytics Tools: Real-time metrics and performance data
- Utility Tools: Context synchronization and system utilities
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Custom exception for tool execution errors."""
    pass


class ToolRegistry:
    """Registry for managing MCP tools and their handlers."""
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.categories: Dict[str, List[str]] = {}
    
    def register_tool(
        self,
        tool_id: str,
        handler: Any,
        method: str,
        description: str,
        input_schema: Dict[str, Any],
        category: str = "general",
        sync_context: bool = False,
        requires_auth: bool = True
    ):
        """Register a tool with the registry."""
        self.tools[tool_id] = {
            "handler": handler,
            "method": method,
            "description": description,
            "input_schema": input_schema,
            "category": category,
            "sync_context": sync_context,
            "requires_auth": requires_auth,
            "registered_at": datetime.utcnow()
        }
        
        # Add to category
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(tool_id)
        
        logger.debug(f"Registered tool: {tool_id} in category: {category}")
    
    def get_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get tool information by ID."""
        return self.tools.get(tool_id)
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get all tools in a category."""
        return self.categories.get(category, [])
    
    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered tools."""
        return self.tools.copy()
    
    def list_categories(self) -> List[str]:
        """List all tool categories."""
        return list(self.categories.keys())


class KnowledgeHubTools:
    """Main tools class for KnowledgeHub integration."""
    
    def __init__(self):
        self.registry = ToolRegistry()
        logger.info("KnowledgeHub tools initialized")
    
    def get_tool_count(self) -> int:
        """Get total number of registered tools."""
        return len(self.registry.get_all_tools())
    
    def get_categories(self) -> List[str]:
        """Get all tool categories."""
        return self.registry.list_categories()


# Memory Tools Definition
MEMORY_TOOLS = {
    "create_memory": {
        "method": "create_memory",
        "description": "Create a new memory with AI-enhanced processing and embedding",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to store in memory"
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["conversation", "code", "decision", "error", "insight", "fact"],
                    "description": "Type of memory to create"
                },
                "project_id": {
                    "type": "string",
                    "description": "Optional project ID to associate with memory"
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session ID to associate with memory"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for the memory"
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata for the memory"
                }
            },
            "required": ["content", "memory_type"]
        },
        "category": "memory",
        "sync_context": True
    },
    
    "search_memories": {
        "method": "search_memories",
        "description": "Search memories using AI-powered semantic search",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for finding relevant memories"
                },
                "memory_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by memory types"
                },
                "project_id": {
                    "type": "string",
                    "description": "Filter by project ID"
                },
                "session_id": {
                    "type": "string",
                    "description": "Filter by session ID"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                    "description": "Maximum number of results to return"
                },
                "similarity_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.7,
                    "description": "Minimum similarity score for results"
                }
            },
            "required": ["query"]
        },
        "category": "memory"
    },
    
    "get_memory": {
        "method": "get_memory",
        "description": "Retrieve a specific memory by ID with full context",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The ID of the memory to retrieve"
                },
                "include_related": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include related memories in the response"
                }
            },
            "required": ["memory_id"]
        },
        "category": "memory"
    },
    
    "update_memory": {
        "method": "update_memory",
        "description": "Update an existing memory with new content or metadata",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The ID of the memory to update"
                },
                "content": {
                    "type": "string",
                    "description": "New content for the memory"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Updated tags for the memory"
                },
                "metadata": {
                    "type": "object",
                    "description": "Updated metadata for the memory"
                }
            },
            "required": ["memory_id"]
        },
        "category": "memory",
        "sync_context": True
    },
    
    "get_memory_stats": {
        "method": "get_memory_stats",
        "description": "Get comprehensive statistics about the memory system",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Filter stats by project ID"
                },
                "time_range": {
                    "type": "string",
                    "enum": ["1h", "24h", "7d", "30d"],
                    "default": "24h",
                    "description": "Time range for statistics"
                }
            }
        },
        "category": "memory"
    }
}

# Session Tools Definition
SESSION_TOOLS = {
    "init_session": {
        "method": "init_session",
        "description": "Initialize a new AI-enhanced session with context restoration",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_type": {
                    "type": "string",
                    "enum": ["coding", "research", "debugging", "analysis"],
                    "default": "coding",
                    "description": "Type of session to initialize"
                },
                "project_id": {
                    "type": "string",
                    "description": "Project ID to associate with session"
                },
                "context_data": {
                    "type": "object",
                    "description": "Initial context data for the session"
                },
                "restore_from": {
                    "type": "string",
                    "description": "Previous session ID to restore context from"
                }
            }
        },
        "category": "session",
        "sync_context": True
    },
    
    "get_session": {
        "method": "get_session",
        "description": "Get current session information and context",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID to retrieve (uses current if not provided)"
                },
                "include_context": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include session context in response"
                }
            }
        },
        "category": "session"
    },
    
    "update_session_context": {
        "method": "update_session_context",
        "description": "Update session context with new information",
        "input_schema": {
            "type": "object",
            "properties": {
                "context_update": {
                    "type": "object",
                    "description": "Context information to add or update"
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID to update (uses current if not provided)"
                }
            },
            "required": ["context_update"]
        },
        "category": "session",
        "sync_context": True
    },
    
    "end_session": {
        "method": "end_session",
        "description": "End the current session and save context for future restoration",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID to end (uses current if not provided)"
                },
                "summary": {
                    "type": "string",
                    "description": "Optional summary of the session"
                },
                "save_context": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to save context for future restoration"
                }
            }
        },
        "category": "session",
        "sync_context": True
    },
    
    "get_session_history": {
        "method": "get_session_history",
        "description": "Get history of previous sessions",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Filter by project ID"
                },
                "session_type": {
                    "type": "string",
                    "description": "Filter by session type"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                    "description": "Maximum number of sessions to return"
                }
            }
        },
        "category": "session"
    }
}

# AI Tools Definition
AI_TOOLS = {
    "predict_next_tasks": {
        "method": "predict_next_tasks",
        "description": "Get AI predictions for next likely tasks based on context",
        "input_schema": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "Current context for task prediction"
                },
                "project_id": {
                    "type": "string",
                    "description": "Project ID for context-aware predictions"
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID for predictions"
                },
                "num_predictions": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                    "description": "Number of predictions to return"
                }
            }
        },
        "category": "ai"
    },
    
    "analyze_patterns": {
        "method": "analyze_patterns",
        "description": "Analyze patterns in code, behavior, or data using AI",
        "input_schema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Data to analyze for patterns"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["code", "usage", "error", "performance"],
                    "default": "code",
                    "description": "Type of pattern analysis to perform"
                },
                "project_id": {
                    "type": "string",
                    "description": "Project context for analysis"
                }
            },
            "required": ["data"]
        },
        "category": "ai"
    },
    
    "get_ai_insights": {
        "method": "get_ai_insights",
        "description": "Get AI-generated insights and recommendations",
        "input_schema": {
            "type": "object",
            "properties": {
                "focus_area": {
                    "type": "string",
                    "enum": ["performance", "code_quality", "debugging", "optimization"],
                    "description": "Area to focus insights on"
                },
                "project_id": {
                    "type": "string",
                    "description": "Project ID for context-specific insights"
                },
                "time_range": {
                    "type": "string",
                    "enum": ["1h", "24h", "7d", "30d"],
                    "default": "24h",
                    "description": "Time range for insight generation"
                }
            }
        },
        "category": "ai"
    },
    
    "record_decision": {
        "method": "record_decision",
        "description": "Record a decision with AI-enhanced reasoning tracking",
        "input_schema": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "description": "The decision that was made"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Reasoning behind the decision"
                },
                "alternatives": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Alternative options that were considered"
                },
                "context": {
                    "type": "string",
                    "description": "Context in which the decision was made"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence level in the decision"
                },
                "project_id": {
                    "type": "string",
                    "description": "Project associated with the decision"
                }
            },
            "required": ["decision", "reasoning"]
        },
        "category": "ai",
        "sync_context": True
    },
    
    "track_error": {
        "method": "track_error",
        "description": "Track and learn from errors using AI analysis",
        "input_schema": {
            "type": "object",
            "properties": {
                "error_type": {
                    "type": "string",
                    "description": "Type of error that occurred"
                },
                "error_message": {
                    "type": "string",
                    "description": "Error message or description"
                },
                "solution": {
                    "type": "string",
                    "description": "Solution that was applied"
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the solution was successful"
                },
                "context": {
                    "type": "object",
                    "description": "Context when the error occurred"
                },
                "project_id": {
                    "type": "string",
                    "description": "Project where the error occurred"
                }
            },
            "required": ["error_type", "error_message"]
        },
        "category": "ai",
        "sync_context": True
    }
}

# Analytics Tools Definition
ANALYTICS_TOOLS = {
    "get_metrics": {
        "method": "get_metrics",
        "description": "Get real-time metrics and performance data",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific metrics to retrieve"
                },
                "time_window": {
                    "type": "string",
                    "enum": ["1m", "5m", "15m", "1h", "24h"],
                    "default": "1h",
                    "description": "Time window for metrics"
                },
                "aggregation": {
                    "type": "string",
                    "enum": ["avg", "sum", "min", "max", "count"],
                    "default": "avg",
                    "description": "Aggregation method for metrics"
                }
            }
        },
        "category": "analytics"
    },
    
    "get_dashboard_data": {
        "method": "get_dashboard_data",
        "description": "Get comprehensive dashboard data with real-time updates",
        "input_schema": {
            "type": "object",
            "properties": {
                "dashboard_type": {
                    "type": "string",
                    "enum": ["system", "application", "ai", "comprehensive"],
                    "default": "comprehensive",
                    "description": "Type of dashboard data to retrieve"
                },
                "time_window": {
                    "type": "string",
                    "enum": ["1h", "24h", "7d"],
                    "default": "1h",
                    "description": "Time window for dashboard data"
                },
                "project_id": {
                    "type": "string",
                    "description": "Filter by project ID"
                }
            }
        },
        "category": "analytics"
    },
    
    "get_alerts": {
        "method": "get_alerts",
        "description": "Get active alerts and alert history",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "resolved", "all"],
                    "default": "active",
                    "description": "Alert status filter"
                },
                "severity": {
                    "type": "string",
                    "enum": ["info", "warning", "critical", "emergency"],
                    "description": "Filter by alert severity"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20,
                    "description": "Maximum number of alerts to return"
                }
            }
        },
        "category": "analytics"
    },
    
    "get_performance_report": {
        "method": "get_performance_report",
        "description": "Generate comprehensive performance report",
        "input_schema": {
            "type": "object",
            "properties": {
                "report_type": {
                    "type": "string",
                    "enum": ["system", "application", "user", "project"],
                    "default": "system",
                    "description": "Type of performance report"
                },
                "time_range": {
                    "type": "string",
                    "enum": ["1h", "24h", "7d", "30d"],
                    "default": "24h",
                    "description": "Time range for the report"
                },
                "include_trends": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include trend analysis in the report"
                }
            }
        },
        "category": "analytics"
    }
}

# Utility Tools Definition
UTILITY_TOOLS = {
    "sync_context": {
        "method": "sync_context",
        "description": "Synchronize context between Claude Code and KnowledgeHub",
        "input_schema": {
            "type": "object",
            "properties": {
                "context_data": {
                    "type": "object",
                    "description": "Context data to synchronize"
                },
                "sync_direction": {
                    "type": "string",
                    "enum": ["to_knowledgehub", "from_knowledgehub", "bidirectional"],
                    "default": "bidirectional",
                    "description": "Direction of context synchronization"
                }
            }
        },
        "category": "utility",
        "sync_context": True,
        "requires_auth": False
    },
    
    "get_system_status": {
        "method": "get_system_status",
        "description": "Get current system status and health information",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_services": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include individual service status"
                },
                "include_metrics": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include basic system metrics"
                }
            }
        },
        "category": "utility",
        "requires_auth": False
    },
    
    "health_check": {
        "method": "health_check",
        "description": "Perform comprehensive health check of KnowledgeHub systems",
        "input_schema": {
            "type": "object",
            "properties": {
                "deep_check": {
                    "type": "boolean",
                    "default": False,
                    "description": "Perform deep health check including database and AI services"
                }
            }
        },
        "category": "utility",
        "requires_auth": False
    },
    
    "get_api_info": {
        "method": "get_api_info",
        "description": "Get information about available APIs and endpoints",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["memory", "session", "ai", "analytics", "all"],
                    "default": "all",
                    "description": "API category to get information for"
                }
            }
        },
        "category": "utility",
        "requires_auth": False
    }
}

# Export all tool definitions
ALL_TOOL_CATEGORIES = {
    "memory": MEMORY_TOOLS,
    "session": SESSION_TOOLS,
    "ai": AI_TOOLS,
    "analytics": ANALYTICS_TOOLS,
    "utility": UTILITY_TOOLS
}

def get_total_tool_count() -> int:
    """Get the total number of tools across all categories."""
    return sum(len(tools) for tools in ALL_TOOL_CATEGORIES.values())

def get_tool_by_name(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get tool definition by name across all categories."""
    for category_tools in ALL_TOOL_CATEGORIES.values():
        if tool_name in category_tools:
            return category_tools[tool_name]
    return None

def list_all_tool_names() -> List[str]:
    """Get a list of all tool names."""
    all_tools = []
    for category_tools in ALL_TOOL_CATEGORIES.values():
        all_tools.extend(category_tools.keys())
    return all_tools

logger.info(f"Loaded {get_total_tool_count()} tools across {len(ALL_TOOL_CATEGORIES)} categories")