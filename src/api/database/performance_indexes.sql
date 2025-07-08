-- Performance Optimization Indexes for Memory System
-- This migration adds optimized indexes based on actual query patterns

-- Check if indexes already exist to avoid errors
DO $$
BEGIN
    -- Drop existing basic indexes if they're not optimal
    -- We'll recreate them with better structure
    
    -- Performance Indexes for Memory Queries
    
    -- 1. PRIMARY: Session-based memory filtering (most critical - 80% of queries)
    -- Covers: session_id, memory_type, importance with optimal ordering
    CREATE INDEX IF NOT EXISTS idx_memories_session_type_importance_optimized 
    ON memories(session_id, memory_type, importance DESC, created_at DESC);
    
    -- 2. Time-based queries with importance ranking
    -- Covers: Recent memories with importance scoring
    CREATE INDEX IF NOT EXISTS idx_memories_created_importance_desc 
    ON memories(created_at DESC, importance DESC) 
    INCLUDE (session_id, memory_type);
    
    -- 3. Importance range queries (for context compression)
    -- Covers: importance >= threshold queries
    CREATE INDEX IF NOT EXISTS idx_memories_importance_range 
    ON memories(importance DESC, created_at DESC) 
    WHERE importance >= 0.7;
    
    -- 4. Memory type filtering with performance
    -- Covers: memory_type queries with sorting
    CREATE INDEX IF NOT EXISTS idx_memories_type_performance 
    ON memories(memory_type, importance DESC, created_at DESC) 
    INCLUDE (session_id);
    
    -- 5. Vector search eligibility and performance
    -- Covers: WHERE embedding IS NOT NULL with session context
    CREATE INDEX IF NOT EXISTS idx_memories_embedding_ready 
    ON memories(session_id, importance DESC) 
    WHERE embedding IS NOT NULL;
    
    -- 6. Cross-session memory search (for user-wide searches)
    -- Covers: memory queries across multiple sessions for a user
    CREATE INDEX IF NOT EXISTS idx_memories_cross_session_search 
    ON memories(memory_type, importance DESC, created_at DESC) 
    INCLUDE (session_id, content, summary);
    
    -- Session Performance Indexes
    
    -- 7. User session lookup optimization
    -- Covers: user_id with project and time filtering
    CREATE INDEX IF NOT EXISTS idx_memory_sessions_user_optimized 
    ON memory_sessions(user_id, project_id, started_at DESC, ended_at) 
    INCLUDE (id, metadata, tags);
    
    -- 8. Active sessions lookup (ended_at IS NULL)
    -- Covers: Finding active sessions for users
    CREATE INDEX IF NOT EXISTS idx_memory_sessions_active_users 
    ON memory_sessions(user_id, updated_at DESC) 
    WHERE ended_at IS NULL;
    
    -- 9. Session hierarchy navigation
    -- Covers: parent_session_id lookups with timing
    CREATE INDEX IF NOT EXISTS idx_memory_sessions_hierarchy 
    ON memory_sessions(parent_session_id, started_at DESC) 
    WHERE parent_session_id IS NOT NULL;
    
    -- 10. Session cleanup and maintenance
    -- Covers: Old session cleanup queries
    CREATE INDEX IF NOT EXISTS idx_memory_sessions_cleanup_optimized 
    ON memory_sessions(ended_at, created_at) 
    WHERE ended_at IS NOT NULL;
    
    -- Full-Text Search Indexes
    
    -- 11. Content full-text search (PostgreSQL native)
    CREATE INDEX IF NOT EXISTS idx_memories_content_fulltext 
    ON memories USING gin(to_tsvector('english', content));
    
    -- 12. Summary full-text search
    CREATE INDEX IF NOT EXISTS idx_memories_summary_fulltext 
    ON memories USING gin(to_tsvector('english', coalesce(summary, '')));
    
    -- 13. Combined content and summary search
    CREATE INDEX IF NOT EXISTS idx_memories_combined_text_search 
    ON memories USING gin(to_tsvector('english', 
        coalesce(summary, '') || ' ' || content));
    
    -- Array and JSON Optimization
    
    -- 14. Entity array searches (already exists but ensure optimal)
    -- This supports: entities @> ARRAY['entity'] and entity containment
    DROP INDEX IF EXISTS idx_memories_entities;
    CREATE INDEX idx_memories_entities_optimized 
    ON memories USING gin(entities);
    
    -- 15. Related memories array searches
    CREATE INDEX IF NOT EXISTS idx_memories_related_gin 
    ON memories USING gin(related_memories);
    
    -- 16. Metadata JSON search optimization
    DROP INDEX IF EXISTS idx_memories_metadata;
    CREATE INDEX idx_memories_metadata_optimized 
    ON memories USING gin(metadata);
    
    -- 17. Session metadata optimization
    DROP INDEX IF EXISTS idx_memory_sessions_metadata;
    CREATE INDEX idx_memory_sessions_metadata_optimized 
    ON memory_sessions USING gin(metadata);
    
    -- 18. Session tags optimization
    DROP INDEX IF EXISTS idx_memory_sessions_tags;
    CREATE INDEX idx_memory_sessions_tags_optimized 
    ON memory_sessions USING gin(tags);
    
    -- Context Compression Specific Indexes
    
    -- 19. Compression strategy: importance-based
    CREATE INDEX IF NOT EXISTS idx_memories_compression_importance 
    ON memories(session_id, importance DESC) 
    WHERE importance >= 0.5;
    
    -- 20. Compression strategy: recency-weighted
    CREATE INDEX IF NOT EXISTS idx_memories_compression_recency 
    ON memories(session_id, created_at DESC, importance DESC);
    
    -- 21. Compression strategy: entity consolidation
    CREATE INDEX IF NOT EXISTS idx_memories_compression_entities 
    ON memories(session_id) 
    WHERE entities IS NOT NULL AND array_length(entities, 1) > 0;
    
    -- Advanced Composite Indexes
    
    -- 22. Memory search with session context (covers most complex queries)
    CREATE INDEX IF NOT EXISTS idx_memories_advanced_search 
    ON memories(session_id, memory_type, importance DESC, created_at DESC) 
    INCLUDE (content, summary, entities);
    
    -- 23. User memory timeline (for memory system analytics)
    CREATE INDEX IF NOT EXISTS idx_memories_user_timeline 
    ON memories(session_id, created_at DESC) 
    INCLUDE (memory_type, importance, summary);
    
    -- 24. Memory access tracking
    CREATE INDEX IF NOT EXISTS idx_memories_access_tracking 
    ON memories(last_accessed DESC, access_count DESC) 
    WHERE last_accessed IS NOT NULL;
    
    -- Session Analytics Indexes
    
    -- 25. Session duration and activity analysis
    CREATE INDEX IF NOT EXISTS idx_memory_sessions_analytics 
    ON memory_sessions(user_id, started_at, ended_at) 
    INCLUDE (created_at, updated_at);
    
    -- 26. Project-based session analysis
    CREATE INDEX IF NOT EXISTS idx_memory_sessions_project_analytics 
    ON memory_sessions(project_id, started_at DESC) 
    WHERE project_id IS NOT NULL;
    
    -- Vector Search Preparation (if using pgvector extension)
    
    -- 27. Vector similarity search support
    -- Note: This requires pgvector extension
    -- CREATE INDEX IF NOT EXISTS idx_memories_embedding_cosine 
    -- ON memories USING ivfflat (embedding vector_cosine_ops) 
    -- WITH (lists = 1000);
    
    -- Performance Statistics Update
    ANALYZE memories;
    ANALYZE memory_sessions;
    
    RAISE NOTICE 'Performance indexes created successfully. Total indexes: 26';
    
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Error creating indexes: %', SQLERRM;
        RAISE;
END $$;