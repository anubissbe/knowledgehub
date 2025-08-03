#!/usr/bin/env python3
"""
Hybrid Memory MCP Tools - Nova-style local memory with KnowledgeHub power
"""

from typing import Dict, List, Optional, Any
import json
from datetime import datetime

from mcp.types import Tool, TextContent
from ...api.services.hybrid_memory_service import HybridMemoryService
from ...api.utils.token_optimizer import TokenOptimizer


class HybridMemoryTools:
    """MCP tools for hybrid memory system"""
    
    def __init__(self):
        self.memory_service = HybridMemoryService()
        self.token_optimizer = TokenOptimizer()
        
    def get_tools(self) -> List[Tool]:
        """Get all hybrid memory tools"""
        return [
            # Fast local operations
            Tool(
                name="quick_store",
                description="Instantly store memory locally with automatic sync",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content to remember"
                        },
                        "type": {
                            "type": "string",
                            "description": "Memory type (general, code, error, decision, etc.)",
                            "default": "general"
                        },
                        "project": {
                            "type": "string",
                            "description": "Project context"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization"
                        }
                    },
                    "required": ["content"]
                }
            ),
            
            Tool(
                name="quick_recall",
                description="Sub-second memory retrieval with cascade search",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "type": {
                            "type": "string",
                            "description": "Filter by memory type"
                        },
                        "project": {
                            "type": "string",
                            "description": "Filter by project"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            ),
            
            Tool(
                name="context_optimize",
                description="Optimize context to reduce token usage",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content to optimize"
                        },
                        "target_reduction": {
                            "type": "number",
                            "description": "Target token reduction percentage (0-100)",
                            "default": 50
                        },
                        "preserve_code": {
                            "type": "boolean",
                            "description": "Preserve code blocks intact",
                            "default": True
                        }
                    },
                    "required": ["content"]
                }
            ),
            
            # Workflow management
            Tool(
                name="workflow_track",
                description="Track project workflow phases",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": "Project identifier"
                        },
                        "phase": {
                            "type": "string",
                            "description": "Current phase (planning, implementing, testing, etc.)"
                        },
                        "context": {
                            "type": "string",
                            "description": "Phase context and notes"
                        }
                    },
                    "required": ["project", "phase"]
                }
            ),
            
            Tool(
                name="board_update",
                description="Update task board state",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": "Project identifier"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Task identifier"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["todo", "in_progress", "blocked", "done"],
                            "description": "Task status"
                        },
                        "notes": {
                            "type": "string",
                            "description": "Task notes"
                        }
                    },
                    "required": ["project", "task_id", "status"]
                }
            ),
            
            Tool(
                name="relate_entities",
                description="Create relationships between memories",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Source entity/memory ID"
                        },
                        "target": {
                            "type": "string",
                            "description": "Target entity/memory ID"
                        },
                        "relationship": {
                            "type": "string",
                            "description": "Relationship type (depends_on, related_to, implements, etc.)"
                        },
                        "strength": {
                            "type": "number",
                            "description": "Relationship strength (0-1)",
                            "default": 1.0
                        }
                    },
                    "required": ["source", "target", "relationship"]
                }
            ),
            
            # Hybrid operations
            Tool(
                name="sync_status",
                description="Check memory sync status",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "detailed": {
                            "type": "boolean",
                            "description": "Include detailed sync information",
                            "default": False
                        }
                    }
                }
            ),
            
            Tool(
                name="cache_stats",
                description="Get cache performance metrics",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "include_predictions": {
                            "type": "boolean",
                            "description": "Include cache prediction analysis",
                            "default": True
                        }
                    }
                }
            ),
            
            Tool(
                name="priority_set",
                description="Set memory priority for sync and caching",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "Memory identifier"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["critical", "high", "normal", "low"],
                            "description": "Memory priority level"
                        }
                    },
                    "required": ["memory_id", "priority"]
                }
            ),
            
            Tool(
                name="memory_analyze",
                description="Analyze memory patterns and clusters",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": "Project to analyze"
                        },
                        "timeframe": {
                            "type": "string",
                            "description": "Analysis timeframe (e.g., '7d', '1m')",
                            "default": "7d"
                        },
                        "include_suggestions": {
                            "type": "boolean",
                            "description": "Include optimization suggestions",
                            "default": True
                        }
                    }
                }
            )
        ]
    
    async def handle_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool execution"""
        
        # Initialize service if needed
        await self.memory_service.initialize()
        
        if tool_name == "quick_store":
            return await self._handle_quick_store(arguments)
            
        elif tool_name == "quick_recall":
            return await self._handle_quick_recall(arguments)
            
        elif tool_name == "context_optimize":
            return await self._handle_context_optimize(arguments)
            
        elif tool_name == "workflow_track":
            return await self._handle_workflow_track(arguments)
            
        elif tool_name == "board_update":
            return await self._handle_board_update(arguments)
            
        elif tool_name == "relate_entities":
            return await self._handle_relate_entities(arguments)
            
        elif tool_name == "sync_status":
            return await self._handle_sync_status(arguments)
            
        elif tool_name == "cache_stats":
            return await self._handle_cache_stats(arguments)
            
        elif tool_name == "priority_set":
            return await self._handle_priority_set(arguments)
            
        elif tool_name == "memory_analyze":
            return await self._handle_memory_analyze(arguments)
            
        else:
            return [TextContent(type="text", text=f"Unknown tool: {tool_name}")]
    
    async def _handle_quick_store(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle quick memory storage"""
        metadata = {}
        if args.get("project"):
            metadata["project"] = args["project"]
        if args.get("tags"):
            metadata["tags"] = args["tags"]
            
        memory_id = await self.memory_service.store(
            user_id="claude",  # TODO: Get from context
            content=args["content"],
            memory_type=args.get("type", "general"),
            metadata=metadata
        )
        
        return [TextContent(
            type="text",
            text=f"✅ Memory stored: {memory_id} (instant local, queued for sync)"
        )]
    
    async def _handle_quick_recall(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle quick memory recall"""
        results = await self.memory_service.recall(
            query=args["query"],
            user_id="claude",  # TODO: Get from context
            memory_type=args.get("type"),
            limit=args.get("limit", 10)
        )
        
        if not results:
            return [TextContent(type="text", text="No memories found matching query")]
        
        # Format results
        output = f"Found {len(results)} memories:\n\n"
        for i, memory in enumerate(results, 1):
            output += f"{i}. [{memory['type']}] {memory['content'][:100]}...\n"
            output += f"   Created: {memory['created_at']}, Accessed: {memory['access_count']} times\n\n"
        
        return [TextContent(type="text", text=output)]
    
    async def _handle_context_optimize(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle context optimization"""
        optimized, metadata = self.token_optimizer.optimize(
            args["content"],
            context={"preserve_code": args.get("preserve_code", True)}
        )
        
        output = f"Token Optimization Results:\n"
        output += f"Original tokens: {metadata['original_tokens']}\n"
        output += f"Optimized tokens: {metadata['optimized_tokens']}\n"
        output += f"Savings: {metadata['token_savings']} tokens ({metadata['savings_percentage']:.1f}%)\n\n"
        output += f"Strategies applied: {', '.join(metadata['strategies_applied'])}\n\n"
        output += f"Optimized content:\n{optimized}"
        
        return [TextContent(type="text", text=output)]
    
    async def _handle_workflow_track(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle workflow tracking"""
        await self.memory_service.track_workflow(
            project_id=args["project"],
            phase=args["phase"],
            context=args.get("context")
        )
        
        return [TextContent(
            type="text",
            text=f"✅ Workflow updated: {args['project']} → {args['phase']}"
        )]
    
    async def _handle_cache_stats(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle cache statistics"""
        metrics = await self.memory_service.get_metrics()
        
        output = "Cache Performance Metrics:\n"
        output += f"Cache hit rate: {metrics['cache_hit_rate']*100:.1f}%\n"
        output += f"Local hit rate: {metrics['local_hit_rate']*100:.1f}%\n"
        output += f"Total queries: {metrics['total_queries']}\n"
        output += f"Token savings: {metrics['token_savings_total']:,} tokens\n"
        
        if args.get("include_predictions"):
            output += "\nPredictions:\n"
            output += "- Consider pre-loading frequently accessed memories\n"
            output += "- Increase cache TTL for stable content\n"
        
        return [TextContent(type="text", text=output)]
    
    async def _handle_board_update(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle task board updates"""
        # Store board update as a special memory type
        board_memory = f"BOARD|{args['project']}|{args['task_id']}|{args['status']}"
        if args.get("notes"):
            board_memory += f"|{args['notes']}"
            
        memory_id = await self.memory_service.store(
            user_id="claude",
            content=board_memory,
            memory_type="board_update",
            metadata={
                "project": args["project"],
                "task_id": args["task_id"],
                "status": args["status"],
                "notes": args.get("notes", "")
            }
        )
        
        return [TextContent(
            type="text",
            text=f"✅ Board updated: Task {args['task_id']} → {args['status']}"
        )]
    
    async def _handle_relate_entities(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle entity relationship mapping"""
        await self.memory_service.map_relationships(
            source_id=args["source"],
            target_id=args["target"],
            relationship=args["relationship"],
            strength=args.get("strength", 1.0)
        )
        
        return [TextContent(
            type="text",
            text=f"✅ Relationship created: {args['source']} -{args['relationship']}→ {args['target']}"
        )]
    
    async def _handle_sync_status(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle sync status check"""
        from ...api.services.memory_sync_service import MemorySyncService
        
        sync_service = MemorySyncService(
            local_db_path=self.memory_service.local_db_path,
            redis_url=self.memory_service.redis_url
        )
        
        status = await sync_service.get_sync_status()
        
        output = "Memory Sync Status:\n"
        output += f"🔄 Syncing: {'Yes' if status['is_syncing'] else 'No'}\n"
        output += f"⏰ Last sync: {status.get('last_sync', 'Never')}\n\n"
        
        output += "Memory counts by status:\n"
        for sync_status, count in status.get('status_counts', {}).items():
            output += f"  {sync_status}: {count}\n"
        
        if args.get("detailed") and status.get("recent_failures"):
            output += "\nRecent sync failures:\n"
            for failure in status["recent_failures"][:5]:
                output += f"  - {failure['id']}: {failure['error']}\n"
        
        return [TextContent(type="text", text=output)]
    
    async def _handle_priority_set(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle memory priority setting"""
        async with aiosqlite.connect(self.memory_service.local_db_path) as db:
            # Add priority column if not exists
            await db.execute("""
                ALTER TABLE memories 
                ADD COLUMN priority TEXT DEFAULT 'normal'
            """)
            
            # Update priority
            await db.execute("""
                UPDATE memories 
                SET priority = ?,
                    sync_status = 'pending'
                WHERE id = ?
            """, (args["priority"], args["memory_id"]))
            
            await db.commit()
        
        return [TextContent(
            type="text",
            text=f"✅ Priority set: {args['memory_id']} → {args['priority']}"
        )]
    
    async def _handle_memory_analyze(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle memory pattern analysis"""
        import aiosqlite
        from collections import Counter
        from datetime import datetime, timedelta
        
        # Parse timeframe
        timeframe = args.get("timeframe", "7d")
        days = int(timeframe.rstrip('d'))
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        async with aiosqlite.connect(self.memory_service.local_db_path) as db:
            # Analyze memory types
            cursor = await db.execute("""
                SELECT type, COUNT(*) as count
                FROM memories
                WHERE created_at > ?
                GROUP BY type
                ORDER BY count DESC
            """, (cutoff.isoformat(),))
            
            type_stats = await cursor.fetchall()
            
            # Analyze access patterns
            cursor = await db.execute("""
                SELECT id, content, access_count
                FROM memories
                WHERE created_at > ?
                ORDER BY access_count DESC
                LIMIT 10
            """, (cutoff.isoformat(),))
            
            hot_memories = await cursor.fetchall()
            
            # Get total metrics
            cursor = await db.execute("""
                SELECT 
                    COUNT(*) as total_memories,
                    AVG(access_count) as avg_access,
                    SUM(tokens) as total_tokens
                FROM memories
                WHERE created_at > ?
            """, (cutoff.isoformat(),))
            
            metrics = await cursor.fetchone()
        
        output = f"Memory Analysis ({timeframe}):\n\n"
        
        output += "📊 Memory Types:\n"
        for mem_type, count in type_stats:
            output += f"  {mem_type}: {count}\n"
        
        output += f"\n📈 Metrics:\n"
        output += f"  Total memories: {metrics[0]}\n"
        output += f"  Avg access count: {metrics[1]:.1f}\n"
        output += f"  Total tokens: {metrics[2] or 0:,}\n"
        
        output += f"\n🔥 Most accessed memories:\n"
        for mem_id, content, access_count in hot_memories[:5]:
            output += f"  [{access_count}x] {content[:50]}...\n"
        
        if args.get("include_suggestions"):
            output += "\n💡 Optimization suggestions:\n"
            
            # Check if certain types are overrepresented
            if type_stats and type_stats[0][1] > metrics[0] * 0.5:
                output += f"  - Consider sub-categorizing '{type_stats[0][0]}' memories\n"
            
            # Check for low access memories
            if metrics[1] < 2:
                output += "  - Many memories have low access; consider archiving\n"
            
            # Token optimization opportunity
            if metrics[2] and metrics[2] > 100000:
                output += "  - High token count; run context optimization\n"
        
        return [TextContent(type="text", text=output)]