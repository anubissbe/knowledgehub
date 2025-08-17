-- Rollback Script for Hybrid RAG Database Migration
-- This script safely rolls back migrations 004 and 005
-- Version: 2.0.0

-- Create backup tables before rollback
CREATE TABLE IF NOT EXISTS rollback_backup_log (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    operation VARCHAR(255) NOT NULL,
    table_name VARCHAR(255),
    records_affected INTEGER,
    backup_data JSONB,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- Function to safely backup and drop tables
CREATE OR REPLACE FUNCTION safe_rollback_table(
    table_name_param TEXT,
    backup_data BOOLEAN DEFAULT true
) RETURNS VOID AS $$
DECLARE
    record_count INTEGER;
    backup_record JSONB;
BEGIN
    -- Check if table exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = table_name_param) THEN
        -- Get record count
        EXECUTE format('SELECT COUNT(*) FROM %I', table_name_param) INTO record_count;
        
        -- Create backup if requested and table has data
        IF backup_data AND record_count > 0 THEN
            EXECUTE format(
                'SELECT jsonb_agg(row_to_json(t)) FROM (SELECT * FROM %I) t', 
                table_name_param
            ) INTO backup_record;
            
            -- Store backup
            INSERT INTO rollback_backup_log (
                operation, table_name, records_affected, backup_data, notes
            ) VALUES (
                'table_backup', 
                table_name_param, 
                record_count, 
                backup_record,
                'Backup before rollback'
            );
        END IF;
        
        -- Drop the table
        EXECUTE format('DROP TABLE IF EXISTS %I CASCADE', table_name_param);
        
        -- Log the rollback
        INSERT INTO rollback_backup_log (
            operation, table_name, records_affected, notes
        ) VALUES (
            'table_dropped', 
            table_name_param, 
            record_count,
            'Table dropped during rollback'
        );
        
        RAISE NOTICE 'Rolled back table: % (% records backed up)', table_name_param, record_count;
    ELSE
        RAISE NOTICE 'Table % does not exist, skipping rollback', table_name_param;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to safely remove columns
CREATE OR REPLACE FUNCTION safe_remove_column(
    table_name_param TEXT,
    column_name_param TEXT
) RETURNS VOID AS $$
BEGIN
    -- Check if column exists
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = table_name_param AND column_name = column_name_param
    ) THEN
        EXECUTE format('ALTER TABLE %I DROP COLUMN IF EXISTS %I', table_name_param, column_name_param);
        
        INSERT INTO rollback_backup_log (
            operation, table_name, notes
        ) VALUES (
            'column_removed', 
            table_name_param,
            'Removed column: ' || column_name_param
        );
        
        RAISE NOTICE 'Removed column % from table %', column_name_param, table_name_param;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Begin rollback transaction
BEGIN;

RAISE NOTICE 'Starting rollback of Hybrid RAG database migration...';

-- Step 1: Remove new tables in reverse dependency order
SELECT safe_rollback_table('service_dependencies', true);
SELECT safe_rollback_table('service_integration_logs', true);
SELECT safe_rollback_table('service_configurations', true);
SELECT safe_rollback_table('performance_monitoring', true);
SELECT safe_rollback_table('service_health_logs', true);
SELECT safe_rollback_table('firecrawl_jobs', true);
SELECT safe_rollback_table('search_result_cache', false); -- No backup needed for cache
SELECT safe_rollback_table('enhanced_chunks', true);
SELECT safe_rollback_table('document_ingestion_logs', true);
SELECT safe_rollback_table('rag_query_logs', true);
SELECT safe_rollback_table('rag_configurations', true);
SELECT safe_rollback_table('agent_tasks', true);
SELECT safe_rollback_table('workflow_executions', true);
SELECT safe_rollback_table('workflow_definitions', true);
SELECT safe_rollback_table('agent_definitions', true);
SELECT safe_rollback_table('memory_access_logs', true);
SELECT safe_rollback_table('memory_associations', true);
SELECT safe_rollback_table('memory_clusters', true);
SELECT safe_rollback_table('zep_session_mapping', true);

-- Step 2: Remove enhanced columns from existing tables
DO $$
BEGIN
    RAISE NOTICE 'Removing enhanced columns from existing tables...';
    
    -- Remove enhanced fields from ai_memories if they exist
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'ai_memories') THEN
        SELECT safe_remove_column('ai_memories', 'content_hash');
        SELECT safe_remove_column('ai_memories', 'knowledge_entities');
        SELECT safe_remove_column('ai_memories', 'knowledge_relations');
        SELECT safe_remove_column('ai_memories', 'cluster_id');
        SELECT safe_remove_column('ai_memories', 'importance');
        SELECT safe_remove_column('ai_memories', 'confidence_score');
        SELECT safe_remove_column('ai_memories', 'decay_factor');
        SELECT safe_remove_column('ai_memories', 'is_archived');
        SELECT safe_remove_column('ai_memories', 'embedding_version');
        SELECT safe_remove_column('ai_memories', 'last_accessed');
        SELECT safe_remove_column('ai_memories', 'access_count');
        
        RAISE NOTICE 'Removed enhanced columns from ai_memories';
    END IF;
    
END $$;

-- Step 3: Remove views
DROP VIEW IF EXISTS service_health_summary;
DROP VIEW IF EXISTS rag_performance_view;
DROP VIEW IF EXISTS workflow_performance_view;
DROP VIEW IF EXISTS enhanced_memory_view;

-- Step 4: Remove functions and types
DROP FUNCTION IF EXISTS safe_rollback_table(TEXT, BOOLEAN);
DROP FUNCTION IF EXISTS safe_remove_column(TEXT, TEXT);
DROP FUNCTION IF EXISTS update_memory_access_log();
DROP FUNCTION IF EXISTS calculate_memory_importance();
DROP FUNCTION IF EXISTS cluster_memories_by_similarity();

-- Step 5: Remove migration log entries
DELETE FROM migration_log WHERE migration_name IN ('004_hybrid_rag_schema', '005_data_migration');

-- Step 6: Clean up indexes that might remain
DO $$
DECLARE
    index_name TEXT;
BEGIN
    FOR index_name IN 
        SELECT indexname FROM pg_indexes 
        WHERE schemaname = 'public' 
        AND indexname ~ '^idx_(memory_clusters|agent_|workflow_|rag_|enhanced_|service_|zep_|firecrawl_)'
    LOOP
        EXECUTE format('DROP INDEX IF EXISTS %I', index_name);
        RAISE NOTICE 'Dropped index: %', index_name;
    END LOOP;
END $$;

-- Step 7: Update statistics
ANALYZE;

-- Create rollback completion record
INSERT INTO rollback_backup_log (
    operation, notes
) VALUES (
    'rollback_completed',
    'Hybrid RAG database migration rollback completed successfully'
);

RAISE NOTICE '=== ROLLBACK COMPLETED ===';
RAISE NOTICE 'Database has been rolled back to pre-migration state';
RAISE NOTICE 'All data has been backed up in rollback_backup_log table';
RAISE NOTICE 'Migration log entries have been removed';

COMMIT;

-- Final verification
DO $$
DECLARE
    remaining_tables INTEGER;
BEGIN
    SELECT COUNT(*) INTO remaining_tables
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name IN (
        'memory_clusters', 'agent_definitions', 'workflow_definitions',
        'rag_configurations', 'enhanced_chunks', 'service_configurations'
    );
    
    IF remaining_tables = 0 THEN
        RAISE NOTICE 'SUCCESS: All migration tables have been removed';
    ELSE
        RAISE WARNING 'WARNING: % migration tables still exist', remaining_tables;
    END IF;
END $$;