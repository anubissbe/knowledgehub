"""
Memory System Seed Data Generator

Creates realistic test data for the memory system including:
- Sample sessions with various contexts
- Diverse memory types and content
- Different importance levels and patterns
- Realistic conversation flows
- Technical knowledge samples
- User preferences and decisions
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..models.memory import MemorySession, Memory, MemoryType
from ..core.session_manager import SessionManager
from ..core.memory_manager import MemoryManager
from ..core.persistent_context import PersistentContextManager
from ...models import get_db


class MemorySystemSeedData:
    """Generate comprehensive seed data for memory system testing"""
    
    def __init__(self, db: Session):
        self.db = db
        self.session_manager = SessionManager(db)
        self.memory_manager = MemoryManager(db)
        self.persistent_context_manager = PersistentContextManager(db)
        
        # Sample data templates
        self.sample_sessions = self._create_sample_sessions()
        self.sample_memories = self._create_sample_memories()
        self.sample_contexts = self._create_sample_contexts()
    
    def _create_sample_sessions(self) -> List[Dict[str, Any]]:
        """Create sample session data with realistic metadata"""
        return [
            {
                "user_id": "claude-user-1",
                "project_id": "knowledgehub-project",
                "session_metadata": {
                    "session_type": "development",
                    "primary_focus": "memory_system_implementation",
                    "tools_used": ["python", "fastapi", "sqlalchemy"],
                    "complexity_level": "high"
                },
                "tags": ["memory-system", "development", "api-implementation"],
                "duration_minutes": 120,
                "memory_count": 15
            },
            {
                "user_id": "claude-user-1", 
                "project_id": "knowledgehub-project",
                "session_metadata": {
                    "session_type": "debugging",
                    "primary_focus": "authentication_issues",
                    "tools_used": ["curl", "postman", "database"],
                    "complexity_level": "medium"
                },
                "tags": ["debugging", "authentication", "api-security"],
                "duration_minutes": 45,
                "memory_count": 8
            },
            {
                "user_id": "claude-user-2",
                "project_id": "security-audit",
                "session_metadata": {
                    "session_type": "security_review",
                    "primary_focus": "cors_and_rate_limiting",
                    "tools_used": ["security-scanner", "penetration-testing"],
                    "complexity_level": "high"
                },
                "tags": ["security", "cors", "rate-limiting", "penetration-testing"],
                "duration_minutes": 180,
                "memory_count": 22
            },
            {
                "user_id": "claude-user-1",
                "project_id": "documentation-project",
                "session_metadata": {
                    "session_type": "documentation",
                    "primary_focus": "api_documentation",
                    "tools_used": ["markdown", "swagger", "postman"],
                    "complexity_level": "low"
                },
                "tags": ["documentation", "api-specs", "user-guides"],
                "duration_minutes": 90,
                "memory_count": 12
            },
            {
                "user_id": "claude-user-3",
                "project_id": "performance-optimization",
                "session_metadata": {
                    "session_type": "optimization",
                    "primary_focus": "database_query_optimization",
                    "tools_used": ["sql-profiler", "database-analyzer"],
                    "complexity_level": "high"
                },
                "tags": ["performance", "database", "optimization", "sql"],
                "duration_minutes": 150,
                "memory_count": 18
            }
        ]
    
    def _create_sample_memories(self) -> List[Dict[str, Any]]:
        """Create diverse sample memories with different types and importance"""
        return [
            # Technical Knowledge
            {
                "memory_type": MemoryType.TECHNICAL_KNOWLEDGE,
                "content": "FastAPI middleware must be added in reverse order - the last middleware added runs first. This is crucial for security middleware like authentication and CORS.",
                "importance_score": 0.9,
                "entities": ["FastAPI", "middleware", "security", "authentication", "CORS"],
                "facts": ["Middleware execution order is reverse of addition order", "Security middleware should be added last"],
                "metadata": {
                    "source": "fastapi_documentation",
                    "verified": True,
                    "applies_to": ["python", "fastapi", "web-development"]
                }
            },
            {
                "memory_type": MemoryType.TECHNICAL_KNOWLEDGE,
                "content": "SQLAlchemy async sessions require explicit commit() calls and proper session management with async context managers to prevent connection leaks.",
                "importance_score": 0.85,
                "entities": ["SQLAlchemy", "async", "sessions", "database", "connection-management"],
                "facts": ["Async sessions need explicit commits", "Use async context managers", "Prevent connection leaks"],
                "metadata": {
                    "source": "sqlalchemy_documentation",
                    "verified": True,
                    "applies_to": ["python", "sqlalchemy", "database", "async"]
                }
            },
            
            # User Preferences
            {
                "memory_type": MemoryType.USER_PREFERENCE,
                "content": "User prefers comprehensive error handling with detailed logging and user-friendly error messages. Always include both technical details for debugging and user-facing messages.",
                "importance_score": 0.7,
                "entities": ["error-handling", "logging", "user-experience"],
                "facts": ["Comprehensive error handling preferred", "Include both technical and user-friendly messages"],
                "metadata": {
                    "preference_type": "coding_style",
                    "consistency_level": "high"
                }
            },
            {
                "memory_type": MemoryType.USER_PREFERENCE,
                "content": "User prefers Pydantic models for all API validation with strict mode enabled. No raw dictionaries for request/response handling.",
                "importance_score": 0.75,
                "entities": ["Pydantic", "validation", "API", "strict-mode"],
                "facts": ["Use Pydantic for all validation", "Enable strict mode", "No raw dictionaries"],
                "metadata": {
                    "preference_type": "validation_approach",
                    "consistency_level": "strict"
                }
            },
            
            # Decisions
            {
                "memory_type": MemoryType.DECISION,
                "content": "Decided to use Redis for caching and session storage instead of in-memory storage for better scalability and persistence across restarts.",
                "importance_score": 0.8,
                "entities": ["Redis", "caching", "session-storage", "scalability"],
                "facts": ["Redis chosen over in-memory storage", "Better scalability", "Persistence across restarts"],
                "metadata": {
                    "decision_date": "2025-07-08",
                    "alternatives_considered": ["in-memory", "database-only"],
                    "rationale": "scalability_and_persistence"
                }
            },
            {
                "memory_type": MemoryType.DECISION,
                "content": "Decided to implement rate limiting with multiple strategies (sliding window, token bucket, fixed window) to handle different attack patterns effectively.",
                "importance_score": 0.85,
                "entities": ["rate-limiting", "security", "sliding-window", "token-bucket"],
                "facts": ["Multiple rate limiting strategies", "Handle different attack patterns", "Comprehensive security approach"],
                "metadata": {
                    "decision_date": "2025-07-08",
                    "alternatives_considered": ["single-strategy", "third-party-service"],
                    "rationale": "comprehensive_protection"
                }
            },
            
            # Patterns
            {
                "memory_type": MemoryType.PATTERN,
                "content": "Authentication pattern: Always implement middleware for auth, use dependency injection for protected endpoints, cache validated tokens, and provide both API key and Bearer token support.",
                "importance_score": 0.9,
                "entities": ["authentication", "middleware", "dependency-injection", "token-caching"],
                "facts": ["Use middleware for auth", "Dependency injection for protected endpoints", "Cache validated tokens", "Support multiple token formats"],
                "metadata": {
                    "pattern_type": "security_implementation",
                    "reusability": "high",
                    "applies_to": ["fastapi", "authentication", "api-security"]
                }
            },
            {
                "memory_type": MemoryType.PATTERN,
                "content": "Database model pattern: Use SQLAlchemy with async sessions, implement proper relationship mapping, add created_at/updated_at timestamps, and include soft delete functionality.",
                "importance_score": 0.8,
                "entities": ["SQLAlchemy", "async-sessions", "relationships", "timestamps", "soft-delete"],
                "facts": ["Use async sessions", "Proper relationship mapping", "Include timestamps", "Soft delete functionality"],
                "metadata": {
                    "pattern_type": "database_design",
                    "reusability": "high",
                    "applies_to": ["sqlalchemy", "database", "data-modeling"]
                }
            },
            
            # Workflows
            {
                "memory_type": MemoryType.WORKFLOW,
                "content": "Memory system workflow: 1) Create/update session, 2) Extract entities and facts, 3) Score importance, 4) Store with embeddings, 5) Update context graph, 6) Compress if needed, 7) Update session metadata.",
                "importance_score": 0.95,
                "entities": ["memory-system", "workflow", "entities", "facts", "importance-scoring", "embeddings"],
                "facts": ["7-step memory processing workflow", "Includes entity extraction", "Importance scoring", "Context graph updates"],
                "metadata": {
                    "workflow_type": "memory_processing",
                    "steps": 7,
                    "automation_level": "high"
                }
            },
            {
                "memory_type": MemoryType.WORKFLOW,
                "content": "Security implementation workflow: 1) Identify threat vectors, 2) Implement preventive measures, 3) Add monitoring, 4) Test attack scenarios, 5) Document security measures, 6) Regular security audits.",
                "importance_score": 0.9,
                "entities": ["security", "threat-vectors", "prevention", "monitoring", "testing", "documentation"],
                "facts": ["6-step security workflow", "Includes threat identification", "Preventive measures", "Regular audits"],
                "metadata": {
                    "workflow_type": "security_implementation",
                    "steps": 6,
                    "automation_level": "medium"
                }
            },
            
            # Problem-Solution Pairs
            {
                "memory_type": MemoryType.PROBLEM_SOLUTION,
                "content": "Problem: Circular import errors in memory system routers. Solution: Create simplified router files and use dependency injection instead of direct imports for complex dependencies.",
                "importance_score": 0.85,
                "entities": ["circular-imports", "routers", "dependency-injection", "memory-system"],
                "facts": ["Circular imports cause errors", "Use simplified routers", "Dependency injection prevents imports"],
                "metadata": {
                    "problem_type": "import_management",
                    "solution_effectiveness": "high",
                    "recurrence_likelihood": "medium"
                }
            },
            {
                "memory_type": MemoryType.PROBLEM_SOLUTION,
                "content": "Problem: API rate limiting bypass attempts. Solution: Implement multiple rate limiting strategies with IP blacklisting, user agent analysis, and adaptive thresholds.",
                "importance_score": 0.9,
                "entities": ["rate-limiting", "bypass-attempts", "IP-blacklisting", "user-agent-analysis"],
                "facts": ["Multiple strategies prevent bypass", "IP blacklisting effective", "User agent analysis detects tools"],
                "metadata": {
                    "problem_type": "security_bypass",
                    "solution_effectiveness": "high",
                    "recurrence_likelihood": "high"
                }
            },
            
            # Context and Insights
            {
                "memory_type": MemoryType.CONTEXT,
                "content": "KnowledgeHub project context: Building AI-powered knowledge management system with memory capabilities, security hardening, and multi-service architecture. Primary focus on production-ready deployment.",
                "importance_score": 0.8,
                "entities": ["KnowledgeHub", "AI", "knowledge-management", "memory-system", "security", "production"],
                "facts": ["AI-powered knowledge management", "Memory capabilities", "Security hardening", "Multi-service architecture"],
                "metadata": {
                    "context_type": "project_overview",
                    "scope": "system_wide",
                    "relevance_duration": "long_term"
                }
            },
            {
                "memory_type": MemoryType.INSIGHT,
                "content": "Insight: Memory system performance improves significantly with proper indexing on importance_score and session_id columns. Vector similarity searches need careful threshold tuning.",
                "importance_score": 0.75,
                "entities": ["performance", "indexing", "importance-score", "session-id", "vector-similarity"],
                "facts": ["Indexing improves performance", "Index importance_score and session_id", "Vector similarity needs threshold tuning"],
                "metadata": {
                    "insight_type": "performance_optimization",
                    "validation_status": "tested",
                    "applies_to": ["database", "memory-system", "search"]
                }
            }
        ]
    
    def _create_sample_contexts(self) -> List[Dict[str, Any]]:
        """Create sample persistent context data"""
        return [
            {
                "content": "FastAPI development best practices include proper middleware ordering, async/await patterns, Pydantic validation, and comprehensive error handling.",
                "context_type": "technical_knowledge",
                "scope": "project",
                "importance": 0.9,
                "related_entities": ["FastAPI", "middleware", "async", "Pydantic", "error-handling"],
                "metadata": {
                    "source": "development_experience",
                    "validation_level": "high",
                    "applicability": "web_development"
                }
            },
            {
                "content": "User prefers comprehensive documentation with code examples, API specifications, and troubleshooting guides for all implemented features.",
                "context_type": "preferences",
                "scope": "user",
                "importance": 0.7,
                "related_entities": ["documentation", "code-examples", "API-specs", "troubleshooting"],
                "metadata": {
                    "preference_strength": "strong",
                    "consistency": "high"
                }
            },
            {
                "content": "Security-first approach: Always implement authentication, authorization, input validation, rate limiting, and monitoring before deploying to production.",
                "context_type": "patterns",
                "scope": "global",
                "importance": 0.95,
                "related_entities": ["security", "authentication", "authorization", "validation", "rate-limiting", "monitoring"],
                "metadata": {
                    "pattern_type": "security_implementation",
                    "priority": "critical"
                }
            }
        ]
    
    async def generate_seed_data(self, num_sessions: int = 5, num_memories_per_session: int = 10) -> Dict[str, Any]:
        """Generate comprehensive seed data for testing"""
        print(f"🌱 Generating seed data: {num_sessions} sessions, ~{num_memories_per_session} memories per session")
        
        results = {
            "sessions_created": 0,
            "memories_created": 0,
            "contexts_created": 0,
            "session_ids": [],
            "memory_ids": [],
            "context_ids": []
        }
        
        try:
            # Create sessions
            for i in range(min(num_sessions, len(self.sample_sessions))):
                session_data = self.sample_sessions[i]
                
                # Create session
                session = await self._create_test_session(session_data)
                results["sessions_created"] += 1
                results["session_ids"].append(str(session.id))
                
                # Create memories for this session
                session_memories = self.sample_memories[i*num_memories_per_session:(i+1)*num_memories_per_session]
                for memory_data in session_memories:
                    memory = await self._create_test_memory(session.id, memory_data)
                    results["memories_created"] += 1
                    results["memory_ids"].append(str(memory.id))
                
                print(f"✅ Created session {i+1}/{num_sessions} with {len(session_memories)} memories")
            
            # Create persistent context data
            for context_data in self.sample_contexts:
                context_id = await self._create_test_context(context_data)
                results["contexts_created"] += 1
                results["context_ids"].append(str(context_id))
            
            print(f"🎯 Seed data generation complete:")
            print(f"  - Sessions: {results['sessions_created']}")
            print(f"  - Memories: {results['memories_created']}")
            print(f"  - Contexts: {results['contexts_created']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error generating seed data: {e}")
            raise
    
    async def _create_test_session(self, session_data: Dict[str, Any]) -> MemorySession:
        """Create a test session with realistic data"""
        session = MemorySession(
            id=uuid.uuid4(),
            user_id=session_data["user_id"],
            project_id=session_data["project_id"],
            session_metadata=session_data["session_metadata"],
            tags=session_data["tags"],
            started_at=datetime.now(timezone.utc) - timedelta(minutes=session_data["duration_minutes"]),
            ended_at=datetime.now(timezone.utc),
            duration=session_data["duration_minutes"] * 60,  # Convert to seconds
            memory_count=session_data["memory_count"],
            is_active=False
        )
        
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        
        return session
    
    async def _create_test_memory(self, session_id: uuid.UUID, memory_data: Dict[str, Any]) -> Memory:
        """Create a test memory with realistic content"""
        memory = Memory(
            id=uuid.uuid4(),
            session_id=session_id,
            memory_type=memory_data["memory_type"],
            content=memory_data["content"],
            importance_score=memory_data["importance_score"],
            entities=memory_data["entities"],
            facts=memory_data["facts"],
            metadata=memory_data["metadata"],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        self.db.add(memory)
        self.db.commit()
        self.db.refresh(memory)
        
        return memory
    
    async def _create_test_context(self, context_data: Dict[str, Any]) -> uuid.UUID:
        """Create persistent context data"""
        context_id = await self.persistent_context_manager.add_context(
            content=context_data["content"],
            context_type=context_data["context_type"],
            scope=context_data["scope"],
            importance=context_data["importance"],
            related_entities=context_data["related_entities"],
            metadata=context_data["metadata"]
        )
        
        return context_id
    
    async def clear_seed_data(self) -> Dict[str, int]:
        """Clear all seed data for clean testing"""
        print("🧹 Clearing existing seed data...")
        
        results = {
            "sessions_deleted": 0,
            "memories_deleted": 0,
            "contexts_deleted": 0
        }
        
        try:
            # Delete memories first (foreign key constraint)
            memory_count = self.db.query(Memory).count()
            self.db.query(Memory).delete()
            results["memories_deleted"] = memory_count
            
            # Delete sessions
            session_count = self.db.query(MemorySession).count()
            self.db.query(MemorySession).delete()
            results["sessions_deleted"] = session_count
            
            # Clear persistent context (if implemented)
            try:
                await self.persistent_context_manager.clear_all_contexts()
                results["contexts_deleted"] = 1  # Placeholder
            except Exception as e:
                print(f"⚠️ Could not clear persistent contexts: {e}")
            
            self.db.commit()
            
            print(f"✅ Cleared seed data:")
            print(f"  - Sessions: {results['sessions_deleted']}")
            print(f"  - Memories: {results['memories_deleted']}")
            print(f"  - Contexts: {results['contexts_deleted']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error clearing seed data: {e}")
            self.db.rollback()
            raise
    
    async def validate_seed_data(self) -> Dict[str, Any]:
        """Validate that seed data was created correctly"""
        print("🔍 Validating seed data...")
        
        validation_results = {
            "sessions": {
                "count": 0,
                "has_metadata": 0,
                "has_tags": 0,
                "duration_set": 0
            },
            "memories": {
                "count": 0,
                "by_type": {},
                "with_entities": 0,
                "with_facts": 0,
                "avg_importance": 0.0
            },
            "contexts": {
                "count": 0,
                "by_type": {},
                "by_scope": {}
            },
            "validation_passed": True,
            "errors": []
        }
        
        try:
            # Validate sessions
            sessions = self.db.query(MemorySession).all()
            validation_results["sessions"]["count"] = len(sessions)
            
            for session in sessions:
                if session.session_metadata:
                    validation_results["sessions"]["has_metadata"] += 1
                if session.tags:
                    validation_results["sessions"]["has_tags"] += 1
                if session.duration and session.duration > 0:
                    validation_results["sessions"]["duration_set"] += 1
            
            # Validate memories
            memories = self.db.query(Memory).all()
            validation_results["memories"]["count"] = len(memories)
            
            importance_scores = []
            for memory in memories:
                # Count by type
                memory_type = memory.memory_type.value if memory.memory_type else "unknown"
                validation_results["memories"]["by_type"][memory_type] = validation_results["memories"]["by_type"].get(memory_type, 0) + 1
                
                # Count with entities/facts
                if memory.entities and len(memory.entities) > 0:
                    validation_results["memories"]["with_entities"] += 1
                if memory.facts and len(memory.facts) > 0:
                    validation_results["memories"]["with_facts"] += 1
                
                # Track importance scores
                if memory.importance_score is not None:
                    importance_scores.append(memory.importance_score)
            
            if importance_scores:
                validation_results["memories"]["avg_importance"] = sum(importance_scores) / len(importance_scores)
            
            # Validate contexts (if available)
            try:
                context_summary = await self.persistent_context_manager.get_context_summary()
                validation_results["contexts"]["count"] = context_summary.get("total_vectors", 0)
            except Exception as e:
                validation_results["errors"].append(f"Context validation error: {e}")
            
            # Check for validation issues
            if validation_results["sessions"]["count"] == 0:
                validation_results["errors"].append("No sessions created")
                validation_results["validation_passed"] = False
            
            if validation_results["memories"]["count"] == 0:
                validation_results["errors"].append("No memories created")
                validation_results["validation_passed"] = False
            
            if validation_results["memories"]["avg_importance"] == 0.0:
                validation_results["errors"].append("No importance scores set")
                validation_results["validation_passed"] = False
            
            status = "✅ PASSED" if validation_results["validation_passed"] else "❌ FAILED"
            print(f"🔍 Validation {status}")
            
            if validation_results["errors"]:
                print("⚠️ Validation errors:")
                for error in validation_results["errors"]:
                    print(f"  - {error}")
            
            return validation_results
            
        except Exception as e:
            print(f"❌ Validation error: {e}")
            validation_results["validation_passed"] = False
            validation_results["errors"].append(str(e))
            return validation_results


async def main():
    """Main function to run seed data generation"""
    print("🌱 Memory System Seed Data Generator")
    print("=" * 50)
    
    # Get database session
    db = next(get_db())
    
    try:
        # Create seed data generator
        seed_generator = MemorySystemSeedData(db)
        
        # Clear existing data
        await seed_generator.clear_seed_data()
        
        # Generate new seed data
        results = await seed_generator.generate_seed_data(
            num_sessions=5,
            num_memories_per_session=3
        )
        
        # Validate the generated data
        validation = await seed_generator.validate_seed_data()
        
        print("\n" + "=" * 50)
        print("🎯 Seed Data Generation Summary:")
        print(f"  Sessions: {results['sessions_created']}")
        print(f"  Memories: {results['memories_created']}")
        print(f"  Contexts: {results['contexts_created']}")
        print(f"  Validation: {'PASSED' if validation['validation_passed'] else 'FAILED'}")
        
        if validation['errors']:
            print("\n⚠️ Validation Issues:")
            for error in validation['errors']:
                print(f"  - {error}")
        
        print("\n✅ Seed data generation complete!")
        
    except Exception as e:
        print(f"❌ Error in main: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())