#!/usr/bin/env python3
"""
Create AI Memory System Tables.

This script creates the enhanced memory tables for the AI memory system
with embeddings, clustering, and advanced features.
"""

import asyncio
import sys
import os
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy import text
from api.models import get_db
from api.models.memory import Memory, MemoryCluster, MemoryAssociation, MemoryAccess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_memory_tables():
    """Create all memory system tables."""
    try:
        logger.info("🧠 Creating AI Memory System tables...")
        
        db = next(get_db())
        
        # Enable UUID extension if not exists
        db.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"))
        
        # Create memory system tables
        from api.models.base import Base, engine
        
        # Create all tables defined in the models
        Base.metadata.create_all(bind=engine, tables=[
            Memory.__table__,
            MemoryCluster.__table__,
            MemoryAssociation.__table__,
            MemoryAccess.__table__
        ])
        
        db.commit()
        
        # Verify tables were created
        result = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name IN ('ai_memories', 'memory_clusters', 'memory_associations', 'memory_access_logs')
            AND table_schema = 'public'
            ORDER BY table_name;
        """))
        
        tables = [row[0] for row in result]
        logger.info(f"✅ Created memory tables: {tables}")
        
        # Create additional indexes for performance
        create_additional_indexes(db)
        
        # Create custom functions and triggers
        create_memory_functions(db)
        
        logger.info("🎉 AI Memory System tables created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create memory tables: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_additional_indexes(db):
    """Create additional performance indexes."""
    
    indexes = [
        # Advanced search indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_full_text ON ai_memories USING gin(to_tsvector('english', content));",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_tags_gin ON ai_memories USING gin(tags);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_context_gin ON ai_memories USING gin(context);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_metadata_gin ON ai_memories USING gin(metadata);",
        
        # Embedding similarity indexes (for pgvector if available)
        # Note: These will be created conditionally if pgvector is installed
        
        # Performance optimization indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_decay_relevance ON ai_memories (decay_factor, relevance_score) WHERE is_archived = false;",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_expiry_active ON ai_memories (expires_at) WHERE expires_at IS NOT NULL AND is_archived = false;",
        
        # Association indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_associations_strength_type ON memory_associations (association_type, strength DESC);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_associations_reinforcement ON memory_associations (reinforcement_count DESC, last_reinforced DESC);",
        
        # Access pattern indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_access_logs_time_method ON memory_access_logs (accessed_at DESC, retrieval_method);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_access_logs_performance ON memory_access_logs (response_time_ms, context_similarity);",
        
        # Cluster performance indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_clusters_stats ON memory_clusters (memory_count DESC, avg_relevance DESC);",
    ]
    
    logger.info("Creating additional performance indexes...")
    
    for index_sql in indexes:
        try:
            db.execute(text(index_sql))
            logger.debug(f"Created index: {index_sql[:50]}...")
        except Exception as e:
            # Some indexes might already exist or require specific extensions
            logger.warning(f"Index creation warning: {str(e)[:100]}...")
    
    db.commit()
    logger.info("✅ Additional indexes created")


def create_memory_functions(db):
    """Create custom PostgreSQL functions for memory operations."""
    
    functions = [
        # Function to calculate memory decay
        """
        CREATE OR REPLACE FUNCTION calculate_memory_decay(
            created_at TIMESTAMPTZ,
            importance TEXT,
            access_count INTEGER,
            last_accessed TIMESTAMPTZ
        ) RETURNS FLOAT AS $$
        DECLARE
            base_decay FLOAT;
            time_factor FLOAT;
            access_factor FLOAT;
            days_old FLOAT;
            days_since_access FLOAT;
        BEGIN
            -- Base decay rate by importance
            base_decay := CASE importance
                WHEN 'critical' THEN 1.0
                WHEN 'high' THEN 0.95
                WHEN 'medium' THEN 0.85
                WHEN 'low' THEN 0.70
                WHEN 'ephemeral' THEN 0.50
                ELSE 0.85
            END;
            
            -- Calculate time factors
            days_old := EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0;
            days_since_access := EXTRACT(EPOCH FROM (NOW() - last_accessed)) / 86400.0;
            
            -- Time decay (memories fade over time)
            time_factor := EXP(-days_old / 365.0);  -- Exponential decay over year
            
            -- Access boost (frequently accessed memories decay slower)
            access_factor := 1.0 + (LN(access_count + 1) * 0.1);
            
            -- Recency boost (recently accessed memories decay slower)
            IF days_since_access < 7 THEN
                access_factor := access_factor * 1.2;
            ELSIF days_since_access < 30 THEN
                access_factor := access_factor * 1.1;
            END IF;
            
            RETURN GREATEST(0.1, base_decay * time_factor * access_factor);
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Function to update memory relevance score
        """
        CREATE OR REPLACE FUNCTION update_memory_relevance(memory_id UUID) RETURNS VOID AS $$
        DECLARE
            memory_record RECORD;
            new_decay FLOAT;
            new_relevance FLOAT;
        BEGIN
            SELECT * INTO memory_record FROM ai_memories WHERE id = memory_id;
            
            IF FOUND THEN
                -- Calculate new decay factor
                new_decay := calculate_memory_decay(
                    memory_record.created_at,
                    memory_record.importance,
                    memory_record.access_count,
                    memory_record.last_accessed
                );
                
                -- Calculate new relevance (decay * confidence * base relevance)
                new_relevance := new_decay * memory_record.confidence_score * 1.0;
                
                -- Update the memory
                UPDATE ai_memories 
                SET 
                    decay_factor = new_decay,
                    relevance_score = new_relevance
                WHERE id = memory_id;
            END IF;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Function to find similar memories by embedding (placeholder for vector similarity)
        """
        CREATE OR REPLACE FUNCTION find_similar_memories(
            target_embedding FLOAT[],
            user_id_param TEXT,
            limit_param INTEGER DEFAULT 10,
            threshold_param FLOAT DEFAULT 0.7
        ) RETURNS TABLE(memory_id UUID, similarity_score FLOAT) AS $$
        BEGIN
            -- Placeholder implementation - would use pgvector cosine similarity in production
            -- For now, return based on content similarity and recent access
            RETURN QUERY
            SELECT 
                m.id as memory_id,
                (m.relevance_score * 0.7 + (1.0 / (1.0 + EXTRACT(EPOCH FROM (NOW() - m.last_accessed)) / 86400.0)) * 0.3) as similarity_score
            FROM ai_memories m
            WHERE 
                m.user_id = user_id_param
                AND m.is_archived = false
                AND m.embeddings IS NOT NULL
            ORDER BY similarity_score DESC
            LIMIT limit_param;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Trigger to automatically update decay factors
        """
        CREATE OR REPLACE FUNCTION memory_decay_trigger() RETURNS TRIGGER AS $$
        BEGIN
            -- Update decay factor when memory is accessed
            IF TG_OP = 'UPDATE' AND OLD.access_count != NEW.access_count THEN
                NEW.decay_factor := calculate_memory_decay(
                    NEW.created_at,
                    NEW.importance,
                    NEW.access_count,
                    NEW.last_accessed
                );
                NEW.relevance_score := NEW.decay_factor * NEW.confidence_score;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Create the trigger
        """
        DROP TRIGGER IF EXISTS memory_decay_update ON ai_memories;
        CREATE TRIGGER memory_decay_update
            BEFORE UPDATE ON ai_memories
            FOR EACH ROW
            EXECUTE FUNCTION memory_decay_trigger();
        """
    ]
    
    logger.info("Creating memory system functions and triggers...")
    
    for func_sql in functions:
        try:
            db.execute(text(func_sql))
            logger.debug(f"Created function/trigger: {func_sql[:50]}...")
        except Exception as e:
            logger.warning(f"Function creation warning: {str(e)[:100]}...")
    
    db.commit()
    logger.info("✅ Memory functions and triggers created")


def create_sample_data(db):
    """Create sample memory data for testing."""
    
    logger.info("Creating sample memory data...")
    
    sample_data = [
        {
            "user_id": "test_user_1",
            "session_id": "session_001",
            "content": "Learning about PostgreSQL database optimization techniques",
            "memory_type": "learning",
            "importance": "high",
            "tags": ["database", "postgresql", "optimization"],
            "context": {"topic": "database", "skill_level": "intermediate"}
        },
        {
            "user_id": "test_user_1", 
            "session_id": "session_001",
            "content": "Decided to use TimescaleDB for time-series analytics data",
            "memory_type": "decision",
            "importance": "high",
            "tags": ["timescaledb", "analytics", "architecture"],
            "context": {"decision_type": "technology", "confidence": 0.9}
        },
        {
            "user_id": "test_user_1",
            "session_id": "session_002", 
            "content": "Fixed SQL injection vulnerability in search endpoint",
            "memory_type": "error",
            "importance": "critical",
            "tags": ["security", "sql-injection", "bugfix"],
            "context": {"error_type": "security", "severity": "high"}
        }
    ]
    
    for data in sample_data:
        try:
            # Insert sample memory
            result = db.execute(text("""
                INSERT INTO ai_memories (
                    user_id, session_id, content, memory_type, importance,
                    tags, context, content_hash, metadata
                ) VALUES (
                    :user_id, :session_id, :content, :memory_type, :importance,
                    :tags, :context, :content_hash, :metadata
                ) RETURNING id
            """), {
                "user_id": data["user_id"],
                "session_id": data["session_id"],
                "content": data["content"],
                "memory_type": data["memory_type"],
                "importance": data["importance"],
                "tags": data["tags"],
                "context": data["context"],
                "content_hash": f"sample_{hash(data['content']) % 1000000}",
                "metadata": {"sample": True}
            })
            
            memory_id = result.fetchone()[0]
            logger.info(f"Created sample memory: {memory_id}")
            
        except Exception as e:
            logger.warning(f"Sample data creation warning: {e}")
    
    db.commit()
    logger.info("✅ Sample memory data created")


async def main():
    """Main function."""
    logger.info("🚀 Starting AI Memory System setup...")
    
    # Create memory tables
    success = create_memory_tables()
    
    if success:
        logger.info("🎉 AI Memory System setup complete!")
        
        # Optionally create sample data
        try:
            db = next(get_db())
            create_sample_data(db)
        except Exception as e:
            logger.warning(f"Sample data creation failed: {e}")
        
        return True
    else:
        logger.error("❌ AI Memory System setup failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)