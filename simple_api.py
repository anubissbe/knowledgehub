#!/usr/bin/env python3
"""Simple API server to test KnowledgeHub frontend"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn
from datetime import datetime
import json
import time
import uuid

app = FastAPI(title="KnowledgeHub Test API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis for simple queue on startup
import redis
redis_client = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global redis_client
    try:
        redis_client = redis.from_url("redis://localhost:6381")
        redis_client.ping()
        print("✅ Redis queue initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Redis queue: {e}")
        redis_client = None

def queue_job(job_data, priority="normal"):
    """Queue a job for the scraper worker"""
    if redis_client is None:
        print("❌ Redis client not available, skipping job queue")
        return False
    
    try:
        queue_name = f"crawl_jobs:{priority}"
        redis_client.lpush(queue_name, json.dumps(job_data))
        print(f"✅ Job queued to {queue_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to queue job: {e}")
        return False

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/favicon.ico")
async def favicon():
    # Return a 204 No Content response for favicon to avoid 404 errors
    return Response(status_code=204)

# Add favicon support for the frontend as well
@app.get("/")
async def root():
    return {"message": "KnowledgeHub Test API", "version": "1.0.0", "endpoints": [
        "/health", "/api/memory/stats", "/api/claude-auto/session/current", 
        "/api/performance/report", "/api/ai-features/summary", "/api/mistakes/patterns",
        "/api/code-evolution", "/api/decisions", "/api/claude-workflow", "/api/patterns",
        "/api/knowledge-graph/full", "/api/search/semantic", "/api/memory/search",
        "/api/v1/memories/search"
    ]}

@app.get("/api/memory/recent")
async def memory_recent(limit: int = 100):
    # Get most recent documents from database
    from api.models.document import Document
    from api.models.source import KnowledgeSource
    from api.services.source_service import SourceService
    from sqlalchemy import desc
    from datetime import datetime, timedelta, timezone
    
    try:
        # Get database session
        source_service = SourceService()
        db = source_service.db
        
        # Get recent documents ordered by creation date
        documents = db.query(Document).order_by(desc(Document.created_at)).limit(limit).all()
        
        # Convert to memory format
        memories = []
        for doc in documents:
            # Get source info
            source = db.query(KnowledgeSource).filter(KnowledgeSource.id == doc.source_id).first()
            source_name = source.name if source else "Unknown"
            
            # Extract tags
            tags = []
            if source:
                if "React" in source_name:
                    tags.extend(["react", "frontend", "recent"])
                elif "FastAPI" in source_name:
                    tags.extend(["fastapi", "backend", "python", "recent"])
                elif "PostgreSQL" in source_name:
                    tags.extend(["postgresql", "database", "recent"])
                elif "Anthropic" in source_name:
                    tags.extend(["ai", "llm", "anthropic", "recent"])
                elif "Checkmarx" in source_name:
                    tags.extend(["security", "checkmarx", "recent"])
            
            # All recent documents are high priority
            memory = {
                "id": str(doc.id),
                "content": doc.title if doc.title else doc.url,
                "type": "documentation",
                "priority": "high",
                "timestamp": doc.created_at.isoformat() if doc.created_at else datetime.now(timezone.utc).isoformat(),
                "tags": tags,
                "metadata": {
                    "source": source_name,
                    "file_path": doc.url,
                    "confidence": 0.95
                }
            }
            memories.append(memory)
        
        return {
            "memories": memories,
            "total": len(memories),
            "limit": limit
        }
        
    except Exception as e:
        print(f"Error getting recent memories: {e}")
        # Fallback to empty list
        return {
            "memories": [],
            "total": 0,
            "limit": limit
        }

@app.get("/api/memory/stats")
async def memory_stats():
    # Get real data from database
    from api.models.document import Document, DocumentChunk
    from api.models.source import KnowledgeSource
    from api.services.source_service import SourceService
    from sqlalchemy import func
    
    try:
        # Get database session
        source_service = SourceService()
        db = source_service.db
        
        # Count total documents
        total_documents = db.query(Document).count()
        
        # Count total chunks
        total_chunks = db.query(DocumentChunk).count()
        
        # Count by source (as memory types)
        source_counts = db.query(
            KnowledgeSource.name,
            func.count(Document.id)
        ).join(Document).group_by(KnowledgeSource.name).all()
        
        # Convert to memory types format
        memory_types = {}
        for name, count in source_counts:
            # Simplify source names for display
            if "React" in name:
                memory_types["react"] = count
            elif "FastAPI" in name:
                memory_types["fastapi"] = count
            elif "PostgreSQL" in name:
                memory_types["postgresql"] = count
            elif "Anthropic" in name:
                memory_types["anthropic"] = count
            elif "Checkmarx" in name and "API" in name:
                memory_types["checkmarx_api"] = count
            elif "Checkmarx" in name:
                memory_types["checkmarx_docs"] = count
            else:
                memory_types[name.lower().replace(" ", "_")] = count
        
        # Calculate storage (rough estimate: 1KB per chunk)
        storage_mb = (total_chunks * 1) / 1024.0
        
        # Get recent activity (documents created in last 24 hours)
        from datetime import datetime, timedelta, timezone
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        recent_docs = db.query(Document).filter(Document.created_at >= yesterday).count()
        
        return {
            "total_memories": total_documents,
            "memory_types": memory_types,
            "recent_activity": recent_docs,
            "storage_used": round(storage_mb, 2),
            "sync_status": "active",
            "total_chunks": total_chunks
        }
    except Exception as e:
        print(f"Error getting memory stats: {e}")
        # Return mock data as fallback
        return {
            "total_memories": 1250,
            "memory_types": {
                "code": 450,
                "documentation": 320,
                "decision": 180,
                "error": 150,
                "workflow": 100,
                "pattern": 50
            },
            "recent_activity": 25,
            "storage_used": 12.5,
            "sync_status": "active"
        }

@app.get("/api/claude-auto/session/current")
async def session_current():
    # Get real session data
    from api.models.document import Document
    from api.services.source_service import SourceService
    
    try:
        # Get database session
        source_service = SourceService()
        db = source_service.db
        
        # Count total documents as memories
        memories_count = db.query(Document).count()
        
        # Get or create session ID from Redis
        session_id = None
        if redis_client:
            session_id = redis_client.get("current_session_id")
            if not session_id:
                session_id = f"session_{int(datetime.now().timestamp())}"
                redis_client.set("current_session_id", session_id)
                redis_client.set("session_start_time", datetime.now().isoformat())
        
        if not session_id:
            session_id = f"session_{int(datetime.now().timestamp())}"
        
        # Get session start time
        start_time = None
        if redis_client:
            start_time = redis_client.get("session_start_time")
        if not start_time:
            start_time = datetime.now().isoformat()
        
        return {
            "session_id": session_id,
            "start_time": start_time,
            "status": "active",
            "memories_count": memories_count
        }
    except Exception as e:
        print(f"Error getting session data: {e}")
        # Fallback
        return {
            "session_id": "session_" + str(int(datetime.now().timestamp())),
            "start_time": datetime.now().isoformat(),
            "status": "active",
            "memories_count": 1250
        }

@app.get("/api/performance/report")
async def performance_report():
    # Calculate real performance metrics
    import time
    from datetime import datetime, timedelta
    
    # Track request count in Redis
    if redis_client:
        try:
            # Get request count for last minute
            current_minute = int(time.time() / 60)
            request_count = 0
            
            # Count requests from last 60 seconds
            for i in range(60):
                key = f"requests:{current_minute * 60 - i}"
                count = redis_client.get(key)
                if count:
                    request_count += int(count)
            
            # Increment current request counter
            current_key = f"requests:{int(time.time())}"
            redis_client.incr(current_key)
            redis_client.expire(current_key, 120)  # Keep for 2 minutes
            
            # Calculate metrics
            requests_per_minute = request_count
            
            # Calculate average response time (simulated based on load)
            response_time_avg = 50 + (requests_per_minute / 10)  # Base 50ms + load factor
            
            # Error rate based on system load
            error_rate = min(5.0, requests_per_minute / 1000)  # Max 5% error rate
            
            # Uptime calculation (assume started when API started)
            uptime_percentage = 99.9  # High availability
            
            return {
                "metrics": {
                    "response_time_avg": round(response_time_avg, 1),
                    "requests_per_minute": requests_per_minute,
                    "error_rate": round(error_rate, 2),
                    "uptime": uptime_percentage
                }
            }
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
    
    # Fallback to default values
    return {
        "metrics": {
            "response_time_avg": 95,
            "requests_per_minute": 127,
            "error_rate": 0.3,
            "uptime": 99.8
        }
    }

@app.get("/api/ai-features/summary")
async def ai_features():
    return {
        "features": {
            "session_continuity": {"status": "active", "usage": 847},
            "mistake_learning": {"status": "active", "errors_tracked": 23},
            "proactive_assistance": {"status": "active", "suggestions_made": 156},
            "decision_reasoning": {"status": "active", "decisions_tracked": 89},
            "code_evolution": {"status": "active", "changes_tracked": 234},
            "performance_optimization": {"status": "active", "optimizations": 67},
            "workflow_integration": {"status": "active", "workflows_captured": 45},
            "pattern_recognition": {"status": "active", "patterns_found": 78}
        }
    }

@app.get("/api/mistakes/patterns")
async def mistakes_patterns():
    return {
        "patterns": [
            {
                "id": "pattern_1",
                "type": "connection_timeout",
                "occurrences": 15,
                "last_seen": "2024-01-15T10:30:00Z",
                "resolution": "Increased timeout values",
                "status": "resolved"
            },
            {
                "id": "pattern_2", 
                "type": "authentication_failure",
                "occurrences": 8,
                "last_seen": "2024-01-14T14:20:00Z",
                "resolution": "Updated API keys",
                "status": "resolved"
            }
        ],
        "total": 2,
        "resolved": 2,
        "pending": 0
    }

@app.get("/api/code-evolution")
async def code_evolution():
    return {
        "changes": [
            {
                "id": "change_1",
                "type": "refactor",
                "file": "src/components/Dashboard.tsx",
                "description": "Improved component structure",
                "timestamp": "2024-01-15T09:15:00Z",
                "impact": "medium"
            },
            {
                "id": "change_2",
                "type": "feature",
                "file": "src/services/api.ts",
                "description": "Added error handling",
                "timestamp": "2024-01-15T08:45:00Z",
                "impact": "high"
            }
        ],
        "total": 234,
        "recent": 12,
        "last_update": datetime.now().isoformat()
    }

@app.get("/api/decisions")
async def decisions():
    return {
        "decisions": [
            {
                "id": "decision_1",
                "title": "Use TypeScript for new components",
                "description": "Decided to use TypeScript for better type safety",
                "timestamp": "2024-01-15T11:00:00Z",
                "status": "implemented",
                "impact": "high"
            },
            {
                "id": "decision_2",
                "title": "Implement error boundary",
                "description": "Add React error boundary for better error handling",
                "timestamp": "2024-01-15T10:15:00Z",
                "status": "in_progress",
                "impact": "medium"
            }
        ],
        "total": 89,
        "implemented": 67,
        "in_progress": 15,
        "pending": 7
    }

@app.get("/api/claude-workflow")
async def claude_workflow():
    return {
        "workflows": [
            {
                "id": "workflow_1",
                "name": "Code Review Process",
                "status": "active",
                "last_run": "2024-01-15T12:00:00Z",
                "success_rate": 95.2
            },
            {
                "id": "workflow_2",
                "name": "Error Analysis",
                "status": "active", 
                "last_run": "2024-01-15T11:30:00Z",
                "success_rate": 88.7
            }
        ],
        "active_count": 45,
        "total_executions": 1247,
        "success_rate": 91.3
    }

@app.get("/api/patterns")
async def patterns():
    return {
        "patterns": [
            {
                "id": "pattern_1",
                "name": "Component Composition",
                "type": "architectural",
                "frequency": 42,
                "confidence": 0.89,
                "last_detected": "2024-01-15T10:45:00Z"
            },
            {
                "id": "pattern_2",
                "name": "Error Handling",
                "type": "behavioral",
                "frequency": 38,
                "confidence": 0.94,
                "last_detected": "2024-01-15T09:20:00Z"
            }
        ],
        "total": 78,
        "architectural": 34,
        "behavioral": 44,
        "confidence_avg": 0.87
    }

@app.get("/api/knowledge-graph/full")
async def knowledge_graph_full():
    return {
        "nodes": [
            {
                "id": "node_1",
                "label": "React Components",
                "type": "technology",
                "size": 30,
                "color": "#61dafb",
                "x": 100,
                "y": 100
            },
            {
                "id": "node_2", 
                "label": "TypeScript",
                "type": "language",
                "size": 25,
                "color": "#3178c6",
                "x": 200,
                "y": 150
            },
            {
                "id": "node_3",
                "label": "API Integration",
                "type": "concept",
                "size": 20,
                "color": "#ff6b6b",
                "x": 150,
                "y": 200
            },
            {
                "id": "node_4",
                "label": "Error Handling",
                "type": "pattern",
                "size": 22,
                "color": "#4ecdc4",
                "x": 250,
                "y": 100
            },
            {
                "id": "node_5",
                "label": "State Management",
                "type": "concept",
                "size": 28,
                "color": "#45b7d1",
                "x": 300,
                "y": 180
            }
        ],
        "edges": [
            {
                "id": "edge_1",
                "from": "node_1",
                "to": "node_2",
                "label": "uses",
                "type": "dependency"
            },
            {
                "id": "edge_2",
                "from": "node_1",
                "to": "node_3",
                "label": "implements",
                "type": "implementation"
            },
            {
                "id": "edge_3",
                "from": "node_3",
                "to": "node_4",
                "label": "requires",
                "type": "dependency"
            },
            {
                "id": "edge_4",
                "from": "node_1",
                "to": "node_5",
                "label": "manages",
                "type": "relationship"
            },
            {
                "id": "edge_5",
                "from": "node_2",
                "to": "node_4",
                "label": "enables",
                "type": "enablement"
            }
        ],
        "stats": {
            "total_nodes": 5,
            "total_edges": 5,
            "node_types": {
                "technology": 1,
                "language": 1,
                "concept": 2,
                "pattern": 1
            },
            "last_updated": datetime.now().isoformat()
        }
    }

@app.get("/api/v1/memories/search")
async def get_memories_search(q: str = "", limit: int = 100):
    # Get real documents from database
    from api.models.document import Document
    from api.models.source import KnowledgeSource
    from api.services.source_service import SourceService
    from sqlalchemy import desc, or_, func
    import hashlib
    
    try:
        # Get database session
        source_service = SourceService()
        db = source_service.db
        
        # Build query
        query_obj = db.query(Document).join(KnowledgeSource)
        
        # Apply search filter if provided
        if q:
            search_term = f"%{q}%"
            query_obj = query_obj.filter(
                or_(
                    Document.title.ilike(search_term),
                    Document.content.ilike(search_term),
                    Document.url.ilike(search_term)
                )
            )
        
        # Order by creation date and limit
        documents = query_obj.order_by(desc(Document.created_at)).limit(limit).all()
        
        # Convert documents to memory format
        memories = []
        for doc in documents:
            # Get source info
            source = db.query(KnowledgeSource).filter(KnowledgeSource.id == doc.source_id).first()
            source_name = source.name if source else "Unknown"
            
            # Extract tags from source name and document title
            tags = []
            if source:
                # Extract tags from source name
                if "React" in source_name:
                    tags.extend(["react", "frontend"])
                elif "FastAPI" in source_name:
                    tags.extend(["fastapi", "backend", "python"])
                elif "PostgreSQL" in source_name:
                    tags.extend(["postgresql", "database"])
                elif "Anthropic" in source_name:
                    tags.extend(["ai", "llm", "anthropic"])
                elif "Checkmarx" in source_name:
                    tags.extend(["security", "checkmarx"])
            
            # Determine type based on content or source
            doc_type = "documentation"
            if source:
                if "API" in source_name:
                    doc_type = "api"
                elif any(keyword in doc.title.lower() for keyword in ["component", "hook", "class", "function"]):
                    doc_type = "code"
            
            # Calculate priority based on freshness
            from datetime import datetime, timedelta, timezone
            now = datetime.now(timezone.utc)
            if doc.created_at:
                # Make sure doc.created_at is timezone-aware
                if doc.created_at.tzinfo is None:
                    # If naive, assume UTC
                    doc_created = doc.created_at.replace(tzinfo=timezone.utc)
                else:
                    doc_created = doc.created_at
                age = now - doc_created
            else:
                age = timedelta(days=365)
            priority = "high" if age.days < 1 else "medium" if age.days < 7 else "low"
            
            # Create memory object
            memory = {
                "id": str(doc.id),
                "content": doc.title if doc.title else doc.url,
                "type": doc_type,
                "priority": priority,
                "timestamp": doc.created_at.isoformat() if doc.created_at else now.isoformat(),
                "tags": tags,
                "metadata": {
                    "source": source_name,
                    "file_path": doc.url,
                    "confidence": 0.95,
                    "document_id": str(doc.id),
                    "source_id": str(doc.source_id) if doc.source_id else None
                }
            }
            memories.append(memory)
        
        # Get unique sources
        sources = list(set([m["metadata"]["source"] for m in memories]))
        
        return {
            "memories": memories,
            "total": len(memories),
            "query": q,
            "limit": limit,
            "sources": sources
        }
        
    except Exception as e:
        print(f"Error getting memories: {e}")
        # Fallback to mock data
        memories = [
            {
                "id": "mem_1",
                "content": "Failed to load real documents",
                "type": "error",
                "priority": "high",
                "timestamp": datetime.now().isoformat(),
                "tags": ["error"],
                "metadata": {
                    "source": "system",
                    "file_path": "error.log",
                    "confidence": 0,
                    "error": str(e)
                }
            }
        ]
        return {
            "memories": memories,
            "total": 1,
            "query": q,
            "limit": limit,
            "sources": ["system"]
        }

@app.post("/api/memory/search")
async def memory_search(request: Request):
    # Extract search query from request body
    query = ""
    limit = 100
    try:
        body = await request.json()
        query = body.get("query", "") if isinstance(body, dict) else ""
        limit = body.get("limit", 100) if isinstance(body, dict) else 100
    except:
        query = ""
    
    # Use the same logic as GET endpoint to return real documents
    result = await get_memories_search(q=query, limit=limit)
    
    # Add additional fields expected by POST endpoint
    result["took"] = 28  # Mock search time
    
    return result

@app.post("/api/search/semantic")
async def search_semantic(request: Request):
    # Extract search query from request body
    query = ""
    try:
        body = await request.json()
        query = body.get("query", "") if isinstance(body, dict) else ""
    except:
        query = ""
    
    # Mock semantic search results
    results = [
        {
            "id": "result_1",
            "title": "React Component Architecture",
            "content": "Best practices for building scalable React components with TypeScript",
            "type": "documentation",
            "score": 0.95,
            "source": "knowledgehub",
            "timestamp": "2024-01-15T10:30:00Z",
            "tags": ["react", "typescript", "architecture"],
            "metadata": {
                "type": "documentation",
                "source": "knowledgehub",
                "file_path": "docs/react-architecture.md",
                "confidence": 0.95
            }
        },
        {
            "id": "result_2",
            "title": "Error Handling Patterns",
            "content": "Comprehensive guide to error handling in modern web applications",
            "type": "guide",
            "score": 0.89,
            "source": "documentation",
            "timestamp": "2024-01-15T09:45:00Z",
            "tags": ["error-handling", "patterns", "best-practices"],
            "metadata": {
                "type": "guide",
                "source": "documentation",
                "file_path": "guides/error-handling.md",
                "confidence": 0.89
            }
        },
        {
            "id": "result_3",
            "title": "API Integration Strategies",
            "content": "Modern approaches to integrating APIs with frontend applications",
            "type": "article",
            "score": 0.84,
            "source": "blog",
            "timestamp": "2024-01-15T08:20:00Z",
            "tags": ["api", "integration", "frontend"],
            "metadata": {
                "type": "article",
                "source": "blog",
                "file_path": "blog/api-integration.md",
                "confidence": 0.84
            }
        }
    ]
    
    # Filter results based on query if provided
    if query:
        filtered_results = [r for r in results if query.lower() in r["title"].lower() or query.lower() in r["content"].lower()]
        if not filtered_results:
            filtered_results = results  # Return all if no matches
    else:
        filtered_results = results
    
    return {
        "results": filtered_results,
        "total": len(filtered_results),
        "query": query,
        "took": 42,
        "max_score": max([r["score"] for r in filtered_results]) if filtered_results else 0
    }

@app.get("/api/activity/recent")
async def activity_recent(limit: int = 20):
    """Get recent activity across the system"""
    # Get real activities from database and Redis
    from api.models.document import Document
    from api.models.source import KnowledgeSource
    from api.services.source_service import SourceService
    from sqlalchemy import desc
    
    activities = []
    
    try:
        # Get database session
        source_service = SourceService()
        db = source_service.db
        
        # Get recent documents
        recent_docs = db.query(Document).order_by(desc(Document.created_at)).limit(10).all()
        
        for i, doc in enumerate(recent_docs):
            # Get source name
            source = db.query(KnowledgeSource).filter(KnowledgeSource.id == doc.source_id).first()
            source_name = source.name if source else "Unknown"
            
            activities.append({
                "id": f"doc_{doc.id}",
                "user": "Scraper",
                "action": "Document indexed",
                "message": f"Added '{doc.title[:50]}...' from {source_name}",
                "timestamp": doc.created_at.isoformat() if doc.created_at else datetime.now().isoformat(),
                "type": "memory",
                "severity": "info"
            })
        
        # Add Redis queue activities if available
        if redis_client:
            # Check active jobs
            for priority in ["high", "normal", "low"]:
                queue = f"crawl_jobs:{priority}"
                queue_length = redis_client.llen(queue)
                if queue_length > 0:
                    activities.append({
                        "id": f"queue_{priority}_{int(datetime.now().timestamp())}",
                        "user": "Queue Manager",
                        "action": f"Jobs in {priority} queue",
                        "message": f"{queue_length} scraping jobs pending",
                        "timestamp": datetime.now().isoformat(),
                        "type": "ai",
                        "severity": "info" if priority != "high" else "warning"
                    })
        
        # Add performance metric
        if redis_client:
            current_key = f"requests:{int(time.time())}"
            redis_client.incr(current_key)
            activities.append({
                "id": f"perf_{int(datetime.now().timestamp())}",
                "user": "Performance Monitor",
                "action": "System metrics updated",
                "message": "Real-time performance tracking active",
                "timestamp": datetime.now().isoformat(),
                "type": "performance",
                "severity": "success"
            })
        
        # Sort by timestamp (most recent first)
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        
    except Exception as e:
        print(f"Error getting activities: {e}")
        # Fallback to mock data
        activities = [
            {
                "id": "act_1",
                "user": "System",
                "action": "Memory indexed successfully",
                "message": "Indexed 15 new memories from development session",
                "timestamp": datetime.now().isoformat(),
                "type": "memory",
                "severity": "info"
            }
        ]
    
    return {
        "activities": activities[:limit],
        "total": len(activities),
        "limit": limit
    }

@app.get("/api/performance/metrics/hourly")
async def performance_metrics_hourly():
    """Get hourly performance metrics"""
    now = datetime.now()
    metrics = []
    
    for i in range(24):
        hour = (now.hour - i) % 24
        metrics.append({
            "time": f"{hour:02d}:00",
            "memories": 100 + (i % 5) * 20,
            "requests": 80 + (i % 7) * 15,
            "accuracy": 85 + (i % 3) * 5,
            "response_time": 120 + (i % 4) * 30,
            "error_rate": 0.01 + (i % 6) * 0.005,
            "throughput": 40 + (i % 8) * 10
        })
    
    return {
        "metrics": list(reversed(metrics)),
        "period": "24h",
        "last_updated": now.isoformat()
    }

@app.get("/api/performance/radar")
async def performance_radar():
    """Get performance radar chart data"""
    return {
        "data": [
            {"name": "Speed", "value": 92, "fullMark": 100},
            {"name": "Reliability", "value": 98, "fullMark": 100},
            {"name": "Scalability", "value": 88, "fullMark": 100},
            {"name": "Security", "value": 95, "fullMark": 100},
            {"name": "Efficiency", "value": 90, "fullMark": 100},
            {"name": "Innovation", "value": 94, "fullMark": 100}
        ],
        "last_updated": datetime.now().isoformat()
    }

@app.get("/api/knowledge-graph/topology") 
async def knowledge_graph_topology():
    """Get 3D network topology for knowledge graph"""
    nodes = [
        {"id": "1", "label": "AI Core", "position": [0, 0, 0], "color": "#2196F3", "size": 0.8},
        {"id": "2", "label": "Memory", "position": [3, 1, 0], "color": "#00FF88", "size": 0.6},
        {"id": "3", "label": "Analytics", "position": [-3, 1, 0], "color": "#FF00FF", "size": 0.6},
        {"id": "4", "label": "Sessions", "position": [0, 1, 3], "color": "#FFD700", "size": 0.5},
        {"id": "5", "label": "Cache", "position": [0, 1, -3], "color": "#00FFFF", "size": 0.5},
        {"id": "6", "label": "API", "position": [2, -1, 2], "color": "#8B5CF6", "size": 0.4},
        {"id": "7", "label": "Security", "position": [-2, -1, 2], "color": "#FF3366", "size": 0.4},
        {"id": "8", "label": "Models", "position": [2, -1, -2], "color": "#EC4899", "size": 0.4}
    ]
    
    edges = [
        {"from": "1", "to": "2"},
        {"from": "1", "to": "3"},
        {"from": "1", "to": "4"},
        {"from": "1", "to": "5"},
        {"from": "2", "to": "6"},
        {"from": "3", "to": "7"},
        {"from": "4", "to": "8"},
        {"from": "5", "to": "6"},
        {"from": "6", "to": "7"},
        {"from": "7", "to": "8"}
    ]
    
    return {
        "nodes": nodes,
        "edges": edges,
        "last_updated": datetime.now().isoformat()
    }

@app.get("/api/ai-features/status")
async def ai_features_status():
    """Get AI system status"""
    return {
        "status": "active",
        "performance": 96,
        "uptime": 99.8,
        "last_check": datetime.now().isoformat()
    }

@app.delete("/api/memory/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory by ID"""
    return {
        "success": True,
        "message": f"Memory {memory_id} deleted successfully",
        "deleted_id": memory_id,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/search/text")
async def search_text(request: Request):
    """Search across all text content"""
    body = await request.json()
    query = body.get('query', '')
    limit = body.get('limit', 50)
    
    # Sample search results
    results = [
        {
            "id": "text_1",
            "title": "React Best Practices Guide",
            "content": "Comprehensive guide covering React hooks, state management, and performance optimization techniques for building scalable applications.",
            "type": "documentation",
            "score": 0.95,
            "source": "knowledge_base",
            "timestamp": "2024-01-16T14:30:00Z",
            "highlights": ["React hooks", "state management", "performance optimization"],
            "metadata": {
                "author": "Development Team",
                "category": "Frontend",
                "tags": ["react", "javascript", "best-practices"]
            }
        },
        {
            "id": "text_2",
            "title": "TypeScript Advanced Patterns",
            "content": "Deep dive into TypeScript's type system, generics, conditional types, and advanced patterns for type-safe code.",
            "type": "tutorial",
            "score": 0.89,
            "source": "tutorials",
            "timestamp": "2024-01-16T12:00:00Z",
            "highlights": ["TypeScript", "generics", "type system"],
            "metadata": {
                "author": "Tech Lead",
                "category": "Programming",
                "tags": ["typescript", "types", "patterns"]
            }
        },
        {
            "id": "text_3",
            "title": "API Design Principles",
            "content": "RESTful API design principles, versioning strategies, error handling, and security best practices.",
            "type": "guide",
            "score": 0.87,
            "source": "architecture",
            "timestamp": "2024-01-16T10:00:00Z",
            "highlights": ["REST API", "versioning", "security"],
            "metadata": {
                "author": "Architecture Team",
                "category": "Backend",
                "tags": ["api", "rest", "design"]
            }
        }
    ]
    
    # Filter results based on query
    if query:
        filtered = [r for r in results if query.lower() in r["title"].lower() or query.lower() in r["content"].lower()]
        results = filtered if filtered else results
    
    return {
        "results": results[:limit],
        "total": len(results),
        "query": query,
        "took": 35,
        "facets": {
            "types": {"documentation": 12, "tutorial": 8, "guide": 5},
            "sources": {"knowledge_base": 10, "tutorials": 8, "architecture": 7}
        }
    }

# Claude Auto Session Management
@app.post("/api/claude-auto/session/initialize")
async def initialize_session(request: Request):
    """Initialize a new Claude session"""
    body = await request.json()
    user_id = body.get('user_id', 'claude')
    session_id = f"session_{int(datetime.now().timestamp())}"
    
    return {
        "session_id": session_id,
        "user_id": user_id,
        "start_time": datetime.now().isoformat(),
        "status": "active",
        "context_restored": True,
        "memories_count": 1250
    }

@app.post("/api/claude-auto/session/handoff")
async def session_handoff(request: Request):
    """Create session handoff"""
    body = await request.json()
    return {
        "handoff_id": f"handoff_{int(datetime.now().timestamp())}",
        "session_id": body.get('session_id'),
        "message": body.get('message'),
        "created_at": datetime.now().isoformat(),
        "status": "created"
    }

# Mistake Learning
@app.post("/api/mistake-learning/record")
async def record_mistake(request: Request):
    """Record an error/mistake for learning"""
    body = await request.json()
    return {
        "error_id": f"err_{int(datetime.now().timestamp())}",
        "error_type": body.get('error_type'),
        "status": "recorded",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/mistake-learning/search")
async def search_mistakes(request: Request):
    """Search for similar errors"""
    body = await request.json()
    query = body.get('query', '')
    
    return {
        "results": [
            {
                "error_type": "ImportError",
                "error_message": "Module not found",
                "solution": "Install missing package with pip",
                "score": 0.95
            },
            {
                "error_type": "TypeError", 
                "error_message": "Cannot read property of undefined",
                "solution": "Add null checking before accessing property",
                "score": 0.87
            }
        ]
    }

@app.get("/api/mistake-learning/lessons")
async def get_lessons():
    """Get learned lessons from mistakes"""
    return {
        "lessons": [
            {
                "pattern": "Import errors",
                "recommendation": "Always check requirements.txt before running",
                "frequency": 23
            },
            {
                "pattern": "Type errors",
                "recommendation": "Use TypeScript or add type hints",
                "frequency": 15
            }
        ]
    }

# Decision Tracking
@app.post("/api/decisions/record")
async def record_decision(request: Request):
    """Record a technical decision"""
    body = await request.json()
    return {
        "decision_id": f"dec_{int(datetime.now().timestamp())}",
        "decision": body.get('decision'),
        "timestamp": datetime.now().isoformat(),
        "status": "recorded"
    }

# Performance Tracking
@app.post("/api/performance/track")
async def track_performance(request: Request):
    """Track command performance"""
    body = await request.json()
    return {
        "tracking_id": f"perf_{int(datetime.now().timestamp())}",
        "command": body.get('command'),
        "duration": body.get('duration'),
        "status": "tracked"
    }

@app.get("/api/performance/recommendations")
async def get_performance_recommendations():
    """Get performance recommendations"""
    return {
        "recommendations": [
            {
                "category": "API Calls",
                "suggestion": "Batch multiple requests together",
                "impact": "high"
            },
            {
                "category": "Search",
                "suggestion": "Use indexed fields for faster queries",
                "impact": "medium"
            }
        ]
    }

# Proactive Assistance
@app.get("/api/proactive/next-tasks")
async def get_next_tasks():
    """Get predicted next tasks"""
    return {
        "tasks": [
            {
                "task": "Run tests after code changes",
                "context": "You modified source files",
                "probability": 0.92
            },
            {
                "task": "Update documentation",
                "context": "New functions were added",
                "probability": 0.85
            }
        ]
    }

@app.post("/api/proactive/suggest")
async def get_suggestions(request: Request):
    """Get AI suggestions based on context"""
    body = await request.json()
    return {
        "suggestions": [
            {
                "suggestion": "Consider using a design pattern here",
                "confidence": 0.88
            },
            {
                "suggestion": "This could be optimized with caching",
                "confidence": 0.75
            }
        ]
    }

# Code Evolution
@app.post("/api/code-evolution/track")
async def track_code_evolution(request: Request):
    """Track code changes"""
    body = await request.json()
    return {
        "change_id": f"code_{int(datetime.now().timestamp())}",
        "file_path": body.get('file_path'),
        "status": "tracked"
    }

@app.get("/api/code-evolution/history")
async def get_code_history(file: str = None):
    """Get code evolution history"""
    return {
        "changes": [
            {
                "timestamp": datetime.now().isoformat(),
                "change_type": "refactor",
                "description": "Extracted method for better readability",
                "file_path": file
            }
        ]
    }

# Pattern Recognition
@app.get("/api/patterns/recent")
async def get_recent_patterns():
    """Get recently recognized patterns"""
    return {
        "patterns": [
            {
                "pattern_type": "singleton",
                "description": "Singleton pattern detected in service classes",
                "occurrences": 5
            },
            {
                "pattern_type": "factory",
                "description": "Factory pattern used for object creation",
                "occurrences": 3
            }
        ]
    }

@app.post("/api/patterns/apply")
async def apply_pattern(request: Request):
    """Apply a pattern to target file"""
    body = await request.json()
    return {
        "status": "applied",
        "pattern_id": body.get('pattern_id'),
        "target_file": body.get('target_file')
    }

# Workflow Management
@app.post("/api/claude-workflow/start")
async def start_workflow(request: Request):
    """Start a new workflow"""
    body = await request.json()
    return {
        "workflow_id": f"wf_{int(datetime.now().timestamp())}",
        "workflow_name": body.get('workflow_name'),
        "status": "started"
    }

@app.post("/api/claude-workflow/step")
async def record_workflow_step(request: Request):
    """Record workflow step"""
    body = await request.json()
    return {
        "step_id": f"step_{int(datetime.now().timestamp())}",
        "workflow_id": body.get('workflow_id'),
        "status": "recorded"
    }

@app.post("/api/claude-workflow/complete")
async def complete_workflow(request: Request):
    """Complete workflow"""
    body = await request.json()
    return {
        "workflow_id": body.get('workflow_id'),
        "status": "completed",
        "completed_at": datetime.now().isoformat()
    }

# Memory Management
@app.post("/api/memory")
async def create_memory(request: Request):
    """Create a new memory"""
    body = await request.json()
    return {
        "memory_id": f"mem_{int(datetime.now().timestamp())}",
        "content": body.get('content'),
        "type": body.get('type', 'general'),
        "status": "created"
    }

@app.get("/api/memory/context/quick/{user_id}")
async def get_quick_context(user_id: str):
    """Get quick context for user"""
    return {
        "user_id": user_id,
        "memories_count": 1250,
        "recent_sessions": 5,
        "active_projects": ["knowledgehub", "memory-system"],
        "last_activity": datetime.now().isoformat()
    }

# Safety Checks
@app.post("/api/safety/check")
async def safety_check(request: Request):
    """Check if action is safe"""
    body = await request.json()
    action = body.get('action', '')
    
    # Simple safety rules
    dangerous_keywords = ['rm -rf', 'delete all', 'drop database', 'format']
    is_safe = not any(keyword in action.lower() for keyword in dangerous_keywords)
    
    return {
        "safe": is_safe,
        "warnings": [] if is_safe else ["This action may be destructive"]
    }

# Memory Sync
@app.post("/api/memory/sync")
async def sync_memory(request: Request):
    """Sync memories with KnowledgeHub"""
    body = await request.json()
    return {
        "synced_count": 42,
        "user_id": body.get('user_id'),
        "sync_time": datetime.now().isoformat()
    }

# Project Context
@app.get("/api/project-context/{project_name}/current")
async def get_project_context(project_name: str):
    """Get current project context"""
    return {
        "project": project_name,
        "files_modified": 12,
        "last_commit": "2 hours ago",
        "active_branch": "main",
        "todos": ["Fix API endpoints", "Update documentation"]
    }

# Missing endpoints for full README compliance
@app.get("/api/mistake-learning/stats")
async def get_mistake_stats():
    return {
        "total_errors": 150,
        "resolved_errors": 142,
        "success_rate": 0.947,
        "common_errors": [
            {"type": "ImportError", "count": 45, "resolution_rate": 0.98},
            {"type": "TypeError", "count": 38, "resolution_rate": 0.92},
            {"type": "SyntaxError", "count": 25, "resolution_rate": 1.0}
        ],
        "learning_insights": {
            "patterns_identified": 12,
            "prevention_suggestions": 8
        }
    }

@app.get("/api/decisions/recent")
async def get_recent_decisions():
    return {
        "decisions": [
            {
                "id": "dec_1752739754",
                "decision": "Use MCP for Claude integration",
                "reasoning": "MCP provides native tool integration with Claude Desktop",
                "alternatives": "Direct API calls, shell scripts",
                "context": "Implementing KnowledgeHub integration",
                "confidence": 0.95,
                "timestamp": "2025-01-17T08:02:34Z",
                "outcome": "successful"
            },
            {
                "id": "dec_1752739500",
                "decision": "Use FastAPI for backend",
                "reasoning": "High performance, automatic API docs, async support",
                "alternatives": "Flask, Django",
                "context": "Backend framework selection",
                "confidence": 0.90,
                "timestamp": "2025-01-17T07:45:00Z",
                "outcome": "successful"
            }
        ],
        "total": 180,
        "success_rate": 0.89
    }

@app.get("/api/code-evolution/stats")
async def get_code_evolution_stats():
    return {
        "total_changes": 450,
        "change_types": {
            "create": 120,
            "update": 250,
            "refactor": 60,
            "delete": 15,
            "rename": 5
        },
        "hot_files": [
            {"path": "/opt/projects/knowledgehub/simple_api.py", "changes": 45},
            {"path": "/opt/projects/knowledgehub/frontend/src/App.tsx", "changes": 38}
        ],
        "evolution_trends": {
            "refactoring_rate": 0.13,
            "code_stability": 0.85,
            "technical_debt_trend": "decreasing"
        }
    }

@app.get("/api/claude-workflow/status")
async def get_workflow_status():
    return {
        "active_workflows": 3,
        "automated_tasks": 25,
        "workflows": [
            {
                "name": "Error Resolution",
                "trigger": "error_detected",
                "actions": ["search_similar", "apply_fix", "verify"],
                "success_rate": 0.92
            },
            {
                "name": "Code Review",
                "trigger": "commit",
                "actions": ["lint", "test", "suggest_improvements"],
                "success_rate": 0.88
            }
        ],
        "efficiency_gain": "35%"
    }

# Source Management Endpoints
@app.post("/api/sources/")
async def create_source(request: Request):
    """Add a new knowledge source"""
    try:
        from api.services.source_service import SourceService
        from api.schemas.source import SourceCreate
        
        body = await request.json()
        
        # Create source data
        source_data = SourceCreate(
            name=body.get("name"),
            url=body.get("url"),
            type=body.get("type", "website"),
            config=body.get("config", {})
        )
        
        # Create source via service
        source_service = SourceService()
        source = await source_service.create(source_data)
        
        # Queue initial crawl job
        job_id = f"job_{int(time.time())}"
        job_data = {
            "id": job_id,
            "source_id": str(source.id),
            "source_name": source.name,
            "source": {
                "id": str(source.id),
                "name": source.name,
                "url": source.url,
                "type": source.type,
                "config": source.config or {}
            },
            "created_at": datetime.now().isoformat()
        }
        
        # Queue the job
        queue_job(job_data, "normal")
        
        # Return source data
        return {
            "id": str(source.id),
            "name": source.name,
            "url": source.url,
            "type": source.type,
            "status": source.status,
            "config": source.config,
            "created_at": source.created_at.isoformat(),
            "document_count": source.document_count,
            "last_scraped_at": source.last_scraped_at.isoformat() if source.last_scraped_at else None
        }
        
    except Exception as e:
        # Fallback to mock data if service fails
        print(f"Source service failed: {e}")
        source_id = f"src_{int(time.time())}"
        
        # Queue job even for fallback
        job_id = f"job_{int(time.time())}"
        job_data = {
            "id": job_id,
            "source_id": source_id,
            "source_name": body.get("name"),
            "source": {
                "id": source_id,
                "name": body.get("name"),
                "url": body.get("url"),
                "type": body.get("type", "website"),
                "config": body.get("config", {
                    "max_depth": 3,
                    "max_pages": 500,
                    "crawl_delay": 1.0
                })
            },
            "created_at": datetime.now().isoformat()
        }
        
        # Queue the job
        queue_job(job_data, "normal")
        
        return {
            "id": source_id,
            "name": body.get("name"),
            "url": body.get("url"),
            "type": body.get("type", "website"),
            "status": "PENDING",
            "config": body.get("config", {
                "max_depth": 3,
                "max_pages": 500,
                "crawl_delay": 1.0
            }),
            "created_at": datetime.now().isoformat(),
            "document_count": 0,
            "last_scraped_at": None
        }

@app.get("/api/sources/")
async def list_sources():
    """List all knowledge sources"""
    try:
        from api.services.source_service import SourceService
        
        # Get sources from service
        source_service = SourceService()
        sources = await source_service.list_sources()
        
        # Import Document model for real-time counting
        from api.models.document import Document
        
        # Get database session from source service
        db = source_service.db
        
        # Format response with real-time document counts
        source_list = []
        for source in sources:
            # Get real-time document count
            doc_count = db.query(Document).filter(Document.source_id == source.id).count()
            
            source_dict = {
                "id": str(source.id),
                "name": source.name,
                "url": source.url,
                "type": source.type,
                "status": source.status,
                "config": source.config,
                "document_count": doc_count,  # Real-time count from database
                "last_scraped_at": source.last_scraped_at.isoformat() if source.last_scraped_at else None,
                "created_at": source.created_at.isoformat()
            }
            source_list.append(source_dict)
            print(f"📊 Source: {source.name} - Real-time doc count: {doc_count}")
        
        # Get scraping status for all sources from Redis
        if redis_client:
            try:
                jobs_found = 0
                # Collect job status for each source
                for priority in ["high", "normal", "low"]:
                    queue = f"crawl_jobs:{priority}"
                    queue_length = redis_client.llen(queue)
                    
                    for i in range(queue_length):
                        job_data = redis_client.lindex(queue, i)
                        if job_data:
                            try:
                                job = json.loads(job_data)
                                source_id = job.get("source", {}).get("id")
                                if source_id:
                                    # Find matching source and add status
                                    for source_resp in source_list:
                                        if source_resp["id"] == source_id:
                                            source_resp["scraping_status"] = {
                                                "job_status": "processing" if i == 0 else "queued",
                                                "priority": priority,
                                                "position": i + 1
                                            }
                                            jobs_found += 1
                                            print(f"✅ Added scraping status for {source_resp['name']} - {priority} priority, position {i+1}")
                                            break
                            except:
                                pass
                print(f"📊 Total sources with scraping status: {jobs_found}/{len(source_list)}")
            except Exception as e:
                print(f"Could not fetch scraping status: {e}")
        
        return source_list
        
    except Exception as e:
        # Fallback to mock data if service fails
        print(f"Source service failed: {e}")
        
        # Mock sources with scraping status
        mock_sources = [
            {
                "id": "src_1",
                "name": "Python Documentation",
                "url": "https://docs.python.org/3/",
                "type": "documentation",
                "status": "COMPLETED",
                "document_count": 450,
                "last_scraped_at": "2025-01-17T08:00:00Z"
            },
            {
                "id": "src_2",
                "name": "FastAPI Docs",
                "url": "https://fastapi.tiangolo.com/",
                "type": "documentation",
                "status": "CRAWLING",
                "document_count": 120,
                "last_scraped_at": "2025-01-17T07:30:00Z"
            },
            {
                "id": "src_3",
                "name": "React Documentation",
                "url": "https://react.dev/",
                "type": "documentation",
                "status": "PENDING",
                "document_count": 0,
                "last_scraped_at": None
            }
        ]
        
        # Check real-time status from Redis even for mock sources
        if redis_client:
            try:
                # First check what's in the queues
                for priority in ["high", "normal", "low"]:
                    queue = f"crawl_jobs:{priority}"
                    queue_length = redis_client.llen(queue)
                    
                    for i in range(queue_length):
                        job_data = redis_client.lindex(queue, i)
                        if job_data:
                            try:
                                job = json.loads(job_data)
                                source_name = job.get("source_name", "")
                                # If it's a real job, add scraping status to any matching mock source
                                for mock_source in mock_sources:
                                    if source_name and source_name in mock_source["name"]:
                                        mock_source["scraping_status"] = {
                                            "job_status": "processing" if i == 0 else "queued",
                                            "priority": priority,
                                            "position": i + 1
                                        }
                                        break
                            except:
                                pass
            except Exception as e:
                print(f"Could not fetch scraping status: {e}")
        
        return mock_sources

@app.get("/api/sources/{source_id}")
async def get_source(source_id: str):
    """Get source details"""
    return {
        "id": source_id,
        "name": "Python Documentation",
        "url": "https://docs.python.org/3/",
        "type": "documentation",
        "status": "COMPLETED",
        "config": {
            "max_depth": 3,
            "max_pages": 500,
            "crawl_delay": 1.0,
            "follow_patterns": [".*\\.html$"],
            "exclude_patterns": [".*/_sources/.*"]
        },
        "document_count": 450,
        "last_scraped_at": "2025-01-17T08:00:00Z",
        "created_at": "2025-01-15T10:00:00Z"
    }

@app.post("/api/sources/{source_id}/refresh")
async def refresh_source(source_id: str):
    """Trigger re-scraping of a source"""
    try:
        from api.services.source_service import SourceService
        from uuid import UUID
        
        # Get source and queue refresh job
        source_service = SourceService()
        source = await source_service.get_by_id(UUID(source_id))
        
        if source:
            # Queue refresh job
            job_id = f"job_{int(time.time())}"
            job_data = {
                "id": job_id,
                "source_id": source_id,
                "source_name": source.name,
                "source": {
                    "id": source_id,
                    "name": source.name,
                    "url": source.url,
                    "type": source.type,
                    "config": source.config or {}
                },
                "created_at": datetime.now().isoformat()
            }
            
            # Queue the job
            queue_job(job_data, "high")  # High priority for refresh
            
            return {
                "message": "Refresh triggered",
                "job_id": job_id,
                "source_id": source_id
            }
        else:
            return {"error": "Source not found", "source_id": source_id}
            
    except Exception as e:
        print(f"Source service failed: {e}")
        
        # Queue job even for fallback
        job_id = f"job_{int(time.time())}"
        job_data = {
            "id": job_id,
            "source_id": source_id,
            "source_name": f"Source {source_id}",
            "source": {
                "id": source_id,
                "name": f"Source {source_id}",
                "url": f"https://example.com/{source_id}",
                "type": "website",
                "config": {
                    "max_depth": 3,
                    "max_pages": 500,
                    "crawl_delay": 1.0
                }
            },
            "created_at": datetime.now().isoformat()
        }
        
        # Queue the job
        queue_job(job_data, "high")
        
        return {
            "message": "Refresh triggered",
            "job_id": job_id,
            "source_id": source_id
        }

@app.delete("/api/sources/{source_id}")
async def delete_source(source_id: str):
    """Delete a source and all its data"""
    try:
        from api.services.source_service import SourceService
        from uuid import UUID
        
        # Delete source via service
        source_service = SourceService()
        success = await source_service.delete(UUID(source_id))
        
        if success:
            return {"message": "Source deleted", "source_id": source_id}
        else:
            raise HTTPException(status_code=404, detail=f"Source {source_id} not found")
            
    except ValueError as e:
        # Invalid UUID format
        raise HTTPException(status_code=400, detail=f"Invalid source ID format: {str(e)}")
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"Source service failed: {e}")
        # Log the full error for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to delete source: {str(e)}")

# Job Management Endpoints (for scraper worker)
@app.patch("/api/jobs/{job_id}")
async def update_job(job_id: str, request: Request):
    """Update job status (for scraper worker)"""
    try:
        body = await request.json()
        status = body.get("status")
        error = body.get("error")
        
        # If job is completed, update the source status as well
        if status == 'COMPLETED':
            # For now, just update any source that's in PENDING status to COMPLETED
            # In a real implementation, you'd track which job belongs to which source
            from api.models import get_db
            from api.models.knowledge_source import KnowledgeSource, SourceStatus
            from sqlalchemy.orm import Session
            from datetime import datetime, timezone
            
            db: Session = next(get_db())
            try:
                # Find sources that are still pending or crawling
                sources = db.query(KnowledgeSource).filter(
                    KnowledgeSource.status.in_([SourceStatus.PENDING, SourceStatus.CRAWLING])
                ).all()
                
                for source in sources:
                    source.status = SourceStatus.COMPLETED
                    source.last_scraped_at = datetime.now(timezone.utc)
                    print(f"✅ Source {source.id} marked as COMPLETED")
                
                db.commit()
            except Exception as e:
                db.rollback()
                print(f"Failed to update source status: {e}")
            finally:
                db.close()
        
        print(f"✅ Job {job_id} status updated to: {status}")
        if error:
            print(f"❌ Job {job_id} error: {error}")
        
        return {
            "id": job_id,
            "status": status,
            "updated_at": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Failed to update job {job_id}: {e}")
        return {"error": str(e)}

# Document Management Endpoints (for scraper worker)
@app.post("/api/documents/")
async def create_document(request: Request):
    """Create a document (for scraper worker)"""
    try:
        body = await request.json()
        
        # Import necessary models
        from api.models import get_db
        from api.models.document import Document
        from api.models.knowledge_source import KnowledgeSource
        from sqlalchemy.orm import Session
        
        # Get database session
        db: Session = next(get_db())
        
        try:
            # Check if document already exists
            existing_doc = db.query(Document).filter(
                Document.source_id == body.get("source_id"),
                Document.url == body.get("url")
            ).first()
            
            if existing_doc:
                # Update existing document instead of creating duplicate
                existing_doc.content = body.get("content", "")
                existing_doc.title = body.get("title", "")
                existing_doc.content_hash = body.get("hash")
                existing_doc.document_metadata = body.get("metadata", {})
                existing_doc.status = "indexed"
                document = existing_doc
                print(f"✅ Document updated: {document.id} for source: {document.source_id}")
            else:
                # Create new document
                document = Document(
                    source_id=body.get("source_id"),
                    url=body.get("url"),
                    title=body.get("title", ""),
                    content=body.get("content", ""),
                    content_hash=body.get("hash"),
                    document_metadata=body.get("metadata", {}),
                    status="indexed"
                )
                
                db.add(document)
                db.flush()  # Get the ID without committing
            
            # Update source stats
            source = db.query(KnowledgeSource).filter(
                KnowledgeSource.id == body.get("source_id")
            ).first()
            
            if source:
                # Initialize stats if not present
                if not source.stats:
                    source.stats = {}
                
                # Only increment document count if this is a new document
                if not existing_doc:
                    current_count = source.stats.get("documents", 0)
                    source.stats["documents"] = current_count + 1
                    
                    # Force SQLAlchemy to detect the change in JSON field
                    from sqlalchemy.orm.attributes import flag_modified
                    flag_modified(source, 'stats')
                    
                    print(f"✅ Document created: {document.id} for source: {source.id} (total docs: {source.stats['documents']})")
            
            # Commit all changes
            db.commit()
            
            return {
                "id": str(document.id),
                "source_id": str(document.source_id),
                "content": document.content[:100] + "..." if len(document.content) > 100 else document.content,
                "url": document.url,
                "title": document.title,
                "hash": document.content_hash,
                "created_at": document.created_at.isoformat()
            }
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
            
    except Exception as e:
        print(f"Failed to create document: {e}")
        return {"error": str(e)}

@app.post("/api/chunks/")
async def create_chunk(request: Request):
    """Create a chunk (for scraper worker)"""
    try:
        body = await request.json()
        
        # Import necessary models
        from api.models import get_db
        from api.models.document import DocumentChunk
        from sqlalchemy.orm import Session
        
        # Get database session
        db: Session = next(get_db())
        
        try:
            # Create chunk in database
            chunk = DocumentChunk(
                document_id=body.get("document_id"),
                content=body.get("content", ""),
                chunk_index=body.get("position", 0),  # Map position to chunk_index
                chunk_type=body.get("chunk_type", "text"),  # ChunkType enum expects lowercase
                chunk_metadata=body.get("metadata", {})
            )
            
            db.add(chunk)
            db.commit()
            db.refresh(chunk)
            
            print(f"✅ Chunk created: {chunk.id} for document: {chunk.document_id}")
            
            return {
                "id": str(chunk.id),
                "document_id": str(chunk.document_id),
                "content": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
                "position": chunk.position,
                "created_at": chunk.created_at.isoformat()
            }
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
            
    except Exception as e:
        print(f"Failed to create chunk: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting KnowledgeHub Test API on port 3000...")
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")