-- Backup Script: Document Current Index State Before Performance Optimization
-- Run this BEFORE applying performance_indexes.sql to allow rollback

-- Generate a complete list of current indexes for backup
DO $$
DECLARE
    rec RECORD;
    backup_sql TEXT := '';
BEGIN
    -- Header comment
    backup_sql := '-- Index Backup Generated: ' || CURRENT_TIMESTAMP || E'\n';
    backup_sql := backup_sql || '-- Use this script to restore original indexes if needed' || E'\n\n';
    
    -- Get all current indexes on memory tables
    FOR rec IN 
        SELECT 
            schemaname,
            tablename,
            indexname,
            indexdef
        FROM pg_indexes 
        WHERE schemaname = 'public' 
        AND (tablename = 'memories' OR tablename = 'memory_sessions')
        AND indexname NOT LIKE '%_pkey'  -- Skip primary key indexes
        ORDER BY tablename, indexname
    LOOP
        backup_sql := backup_sql || '-- Original index: ' || rec.indexname || E'\n';
        backup_sql := backup_sql || rec.indexdef || ';' || E'\n\n';
    END LOOP;
    
    -- Create backup file content
    RAISE NOTICE 'Index backup script generated. Save this content to restore original indexes:';
    RAISE NOTICE '%', backup_sql;
    
    -- Also log current index count
    SELECT COUNT(*) INTO rec FROM pg_indexes 
    WHERE schemaname = 'public' 
    AND (tablename = 'memories' OR tablename = 'memory_sessions')
    AND indexname NOT LIKE '%_pkey';
    
    RAISE NOTICE 'Total indexes to backup: %', rec.count;
    
END $$;

-- Also create a rollback script template
DO $$
BEGIN
    RAISE NOTICE 'ROLLBACK TEMPLATE - Copy and modify as needed:';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_session_type_importance_optimized;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_created_importance_desc;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_importance_range;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_type_performance;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_embedding_ready;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_cross_session_search;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memory_sessions_user_optimized;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memory_sessions_active_users;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memory_sessions_hierarchy;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memory_sessions_cleanup_optimized;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_content_fulltext;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_summary_fulltext;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_combined_text_search;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_entities_optimized;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_related_gin;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_metadata_optimized;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memory_sessions_metadata_optimized;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memory_sessions_tags_optimized;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_compression_importance;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_compression_recency;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_compression_entities;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_advanced_search;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_user_timeline;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memories_access_tracking;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memory_sessions_analytics;';
    RAISE NOTICE 'DROP INDEX IF EXISTS idx_memory_sessions_project_analytics;';
END $$;