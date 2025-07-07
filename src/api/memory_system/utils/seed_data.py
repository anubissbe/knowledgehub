"""Seed data generator for memory system testing"""

import random
from datetime import datetime, timedelta
from typing import List
from uuid import uuid4
from sqlalchemy.orm import Session

from ..models import MemorySession, Memory, MemoryType


class SeedDataGenerator:
    """Generate realistic seed data for testing"""
    
    # Sample content templates
    FACT_TEMPLATES = [
        "User prefers {preference} for {topic}",
        "Project uses {technology} version {version}",
        "The {component} is configured with {setting}",
        "Database connection uses {protocol} on port {port}",
        "API endpoint {endpoint} requires {auth_type} authentication"
    ]
    
    PREFERENCE_TEMPLATES = [
        "User prefers functional components over class components",
        "User likes to use TypeScript for type safety",
        "User prefers async/await over promises",
        "User wants detailed comments in code",
        "User prefers REST APIs over GraphQL"
    ]
    
    CODE_TEMPLATES = [
        "Custom hook useAuth implemented with JWT refresh logic",
        "Error boundary component wraps the main application",
        "Database connection pool configured with 20 connections",
        "Redis cache TTL set to 3600 seconds",
        "Webpack configured with code splitting"
    ]
    
    DECISION_TEMPLATES = [
        "Decided to use PostgreSQL for the main database",
        "Chose React Query for server state management",
        "Will implement authentication using JWT tokens",
        "Selected Docker for containerization",
        "Agreed to follow conventional commits"
    ]
    
    ERROR_TEMPLATES = [
        "CORS error when accessing API from frontend",
        "Database connection timeout after 30 seconds",
        "Memory leak detected in useEffect hook",
        "TypeScript error: Property does not exist on type",
        "WebSocket connection failed with 404"
    ]
    
    ENTITIES = [
        "React", "TypeScript", "PostgreSQL", "Redis", "Docker",
        "FastAPI", "SQLAlchemy", "Pydantic", "JWT", "OAuth",
        "useAuth", "useQuery", "useState", "useEffect", "useMemo",
        "frontend", "backend", "database", "cache", "API"
    ]
    
    USERS = ["alice", "bob", "charlie", "david", "eve"]
    
    def __init__(self, db: Session):
        self.db = db
    
    def generate_sessions(self, count: int = 10) -> List[MemorySession]:
        """Generate sample sessions"""
        sessions = []
        
        for i in range(count):
            user = random.choice(self.USERS)
            
            # Create session
            session = MemorySession(
                user_id=f"{user}@example.com",
                project_id=uuid4() if random.random() > 0.3 else None,
                started_at=datetime.utcnow() - timedelta(
                    days=random.randint(0, 30),
                    hours=random.randint(0, 23)
                ),
                session_metadata={
                    "client": "claude-code",
                    "version": "1.0.0",
                    "environment": random.choice(["development", "staging", "production"])
                },
                tags=self._generate_tags()
            )
            
            # Some sessions should be ended
            if random.random() > 0.3:
                duration_hours = random.randint(1, 8)
                session.ended_at = session.started_at + timedelta(hours=duration_hours)
            
            # Link some sessions
            if i > 0 and random.random() > 0.5:
                session.parent_session_id = sessions[-1].id
                session.add_tag("continued")
            
            self.db.add(session)
            sessions.append(session)
        
        self.db.commit()
        return sessions
    
    def generate_memories(self, sessions: List[MemorySession], 
                          avg_per_session: int = 20) -> List[Memory]:
        """Generate memories for sessions"""
        memories = []
        
        for session in sessions:
            memory_count = random.randint(5, avg_per_session * 2)
            
            for _ in range(memory_count):
                memory_type = random.choice(list(MemoryType))
                content = self._generate_content(memory_type)
                
                memory = Memory(
                    session_id=session.id,
                    content=content,
                    summary=content[:100] if len(content) > 100 else None,
                    memory_type=memory_type,
                    importance=self._generate_importance(memory_type),
                    confidence=random.uniform(0.6, 1.0),
                    entities=self._extract_entities(content),
                    metadata=self._generate_metadata(memory_type)
                )
                
                # Add access patterns
                if random.random() > 0.7:
                    memory.access_count = random.randint(1, 10)
                    memory.last_accessed = datetime.utcnow() - timedelta(
                        hours=random.randint(1, 72)
                    )
                
                self.db.add(memory)
                memories.append(memory)
        
        self.db.commit()
        
        # Add some relationships
        for memory in memories:
            if random.random() > 0.8:
                # Find related memories
                related = [m for m in memories 
                           if m.id != memory.id 
                           and m.session_id == memory.session_id
                           and random.random() > 0.7][:3]
                
                for related_memory in related:
                    memory.add_related_memory(related_memory.id)
        
        self.db.commit()
        return memories
    
    def _generate_tags(self) -> List[str]:
        """Generate random tags"""
        possible_tags = [
            "development", "debugging", "feature", "bugfix",
            "refactoring", "testing", "documentation", "deployment"
        ]
        return random.sample(possible_tags, k=random.randint(1, 3))
    
    def _generate_content(self, memory_type: MemoryType) -> str:
        """Generate content based on memory type"""
        if memory_type == MemoryType.FACT:
            template = random.choice(self.FACT_TEMPLATES)
            return template.format(
                preference=random.choice(["async/await", "hooks", "TypeScript"]),
                topic=random.choice(["error handling", "state management", "routing"]),
                technology=random.choice(["React", "Node.js", "PostgreSQL"]),
                version=f"{random.randint(1, 20)}.{random.randint(0, 9)}.{random.randint(0, 20)}",
                component=random.choice(["API server", "frontend", "database"]),
                setting=random.choice(["connection pooling", "caching", "rate limiting"]),
                protocol=random.choice(["TCP", "HTTP", "WebSocket"]),
                port=random.choice([3000, 5432, 6379, 8080]),
                endpoint=random.choice(["/api/auth", "/api/users", "/api/search"]),
                auth_type=random.choice(["JWT", "API key", "OAuth"])
            )
        
        elif memory_type == MemoryType.PREFERENCE:
            return random.choice(self.PREFERENCE_TEMPLATES)
        
        elif memory_type == MemoryType.CODE:
            return random.choice(self.CODE_TEMPLATES)
        
        elif memory_type == MemoryType.DECISION:
            return random.choice(self.DECISION_TEMPLATES)
        
        elif memory_type == MemoryType.ERROR:
            return random.choice(self.ERROR_TEMPLATES)
        
        elif memory_type == MemoryType.PATTERN:
            return f"Pattern detected: {random.choice(['Repository pattern', 'Factory pattern', 'Observer pattern'])} used in {random.choice(['API layer', 'service layer', 'data layer'])}"
        
        else:  # ENTITY
            entity = random.choice(self.ENTITIES)
            return f"Entity '{entity}' is a {random.choice(['component', 'service', 'module', 'library'])} used for {random.choice(['authentication', 'data processing', 'UI rendering', 'state management'])}"
    
    def _generate_importance(self, memory_type: MemoryType) -> float:
        """Generate importance score based on type"""
        if memory_type in [MemoryType.DECISION, MemoryType.ERROR]:
            return random.uniform(0.6, 1.0)
        elif memory_type in [MemoryType.PREFERENCE, MemoryType.PATTERN]:
            return random.uniform(0.5, 0.9)
        else:
            return random.uniform(0.3, 0.8)
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract entities from content"""
        found_entities = []
        for entity in self.ENTITIES:
            if entity.lower() in content.lower():
                found_entities.append(entity)
        
        return found_entities[:5]  # Limit to 5 entities
    
    def _generate_metadata(self, memory_type: MemoryType) -> dict:
        """Generate metadata based on type"""
        metadata = {
            "source": random.choice(["conversation", "code_analysis", "error_log"]),
            "confidence_reason": random.choice(["explicit_statement", "inferred", "pattern_match"])
        }
        
        if memory_type == MemoryType.CODE:
            metadata["language"] = random.choice(["python", "typescript", "javascript"])
            metadata["line_count"] = random.randint(5, 50)
        
        elif memory_type == MemoryType.ERROR:
            metadata["severity"] = random.choice(["low", "medium", "high"])
            metadata["resolved"] = random.choice([True, False])
        
        return metadata


def generate_test_data(db: Session, session_count: int = 10, avg_memories: int = 20):
    """Generate test data for memory system"""
    generator = SeedDataGenerator(db)
    
    print(f"Generating {session_count} sessions...")
    sessions = generator.generate_sessions(session_count)
    
    print(f"Generating ~{session_count * avg_memories} memories...")
    memories = generator.generate_memories(sessions, avg_memories)
    
    print(f"Generated {len(sessions)} sessions and {len(memories)} memories")
    
    # Print summary
    active_sessions = sum(1 for s in sessions if s.is_active)
    high_importance = sum(1 for m in memories if m.importance >= 0.7)
    
    print(f"Active sessions: {active_sessions}")
    print(f"High importance memories: {high_importance}")
    print(f"Memory types distribution:")
    
    type_counts = {}
    for memory in memories:
        type_name = memory.memory_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    for memory_type, count in sorted(type_counts.items()):
        print(f"  {memory_type}: {count}")
    
    return sessions, memories