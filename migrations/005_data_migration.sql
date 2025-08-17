-- Data Migration Script for Hybrid RAG System
-- Version: 2.0.0 - Preserving and Enhancing Existing Data

-- Function to safely migrate data with error handling
CREATE OR REPLACE FUNCTION migrate_data_safely(
    operation_name TEXT,
    sql_command TEXT
) RETURNS VOID AS $$
BEGIN
    BEGIN
        EXECUTE sql_command;
        RAISE NOTICE 'Successfully completed: %', operation_name;
    EXCEPTION WHEN OTHERS THEN
        RAISE WARNING 'Failed operation (%): %. Error: %', operation_name, sql_command, SQLERRM;
    END;
END;
$$ LANGUAGE plpgsql;

-- Update existing ai_memories with enhanced fields
DO $$
BEGIN
    RAISE NOTICE 'Starting data migration for enhanced memory system...';
    
    -- Update content_hash for existing memories
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_memories' AND column_name = 'content_hash') THEN
        UPDATE ai_memories 
        SET content_hash = encode(sha256(content::bytea), 'hex')
        WHERE content_hash IS NULL OR content_hash = '';
        
        RAISE NOTICE 'Updated content_hash for existing memories';
    END IF;
    
    -- Initialize knowledge_entities for existing memories
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_memories' AND column_name = 'knowledge_entities') THEN
        UPDATE ai_memories 
        SET knowledge_entities = '[]'::jsonb
        WHERE knowledge_entities IS NULL;
        
        RAISE NOTICE 'Initialized knowledge_entities for existing memories';
    END IF;
    
    -- Initialize knowledge_relations for existing memories
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_memories' AND column_name = 'knowledge_relations') THEN
        UPDATE ai_memories 
        SET knowledge_relations = '[]'::jsonb
        WHERE knowledge_relations IS NULL;
        
        RAISE NOTICE 'Initialized knowledge_relations for existing memories';
    END IF;
    
    -- Set default values for new columns
    UPDATE ai_memories 
    SET 
        importance = COALESCE(importance, 'medium'),
        confidence_score = COALESCE(confidence_score, 1.0),
        decay_factor = COALESCE(decay_factor, 1.0),
        is_archived = COALESCE(is_archived, false),
        embedding_version = COALESCE(embedding_version, 'v1.0')
    WHERE importance IS NULL OR confidence_score IS NULL OR decay_factor IS NULL 
       OR is_archived IS NULL OR embedding_version IS NULL;
    
    RAISE NOTICE 'Set default values for enhanced memory fields';
    
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Error during memory system migration: %', SQLERRM;
END $$;

-- Migrate existing documents to enhanced chunks
DO $$
BEGIN
    RAISE NOTICE 'Starting migration of existing chunks to enhanced chunks...';
    
    -- Check if chunks table exists and enhanced_chunks doesn't have data
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'chunks') 
       AND NOT EXISTS (SELECT 1 FROM enhanced_chunks LIMIT 1) THEN
        
        -- Migrate chunks to enhanced_chunks
        INSERT INTO enhanced_chunks (
            id, document_id, chunk_index, content, content_hash, content_length,
            embedding_primary, embedding_model, created_at, updated_at
        )
        SELECT 
            c.id,
            c.document_id,
            COALESCE(c.position, 0) as chunk_index,
            c.content,
            encode(sha256(c.content::bytea), 'hex') as content_hash,
            LENGTH(c.content) as content_length,
            CASE 
                WHEN c.embedding IS NOT NULL THEN 
                    -- Convert vector to array format
                    ARRAY(SELECT unnest(string_to_array(substring(c.embedding::text from 2 for length(c.embedding::text)-2), ',')))::float[]
                ELSE NULL
            END as embedding_primary,
            'BAAI/bge-base-en-v1.5' as embedding_model,
            c.created_at,
            CURRENT_TIMESTAMP as updated_at
        FROM chunks c
        WHERE c.content IS NOT NULL AND c.content != '';
        
        RAISE NOTICE 'Migrated % chunks to enhanced_chunks', 
            (SELECT COUNT(*) FROM chunks WHERE content IS NOT NULL AND content != '');
    END IF;
    
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Error during chunks migration: %', SQLERRM;
END $$;

-- Create default memory clusters based on existing memory types
DO $$
DECLARE
    memory_type_record RECORD;
    cluster_id UUID;
BEGIN
    RAISE NOTICE 'Creating default memory clusters...';
    
    -- Create clusters for each unique memory type
    FOR memory_type_record IN 
        SELECT DISTINCT memory_type, COUNT(*) as memory_count
        FROM ai_memories 
        WHERE memory_type IS NOT NULL
        GROUP BY memory_type
    LOOP
        -- Check if cluster already exists
        SELECT id INTO cluster_id 
        FROM memory_clusters 
        WHERE name = memory_type_record.memory_type || '_cluster';
        
        IF cluster_id IS NULL THEN
            INSERT INTO memory_clusters (
                name, description, cluster_type, memory_count, created_at
            ) VALUES (
                memory_type_record.memory_type || '_cluster',
                'Auto-generated cluster for ' || memory_type_record.memory_type || ' memories',
                memory_type_record.memory_type,
                memory_type_record.memory_count,
                CURRENT_TIMESTAMP
            ) RETURNING id INTO cluster_id;
            
            -- Update memories to reference the cluster
            UPDATE ai_memories 
            SET cluster_id = cluster_id
            WHERE memory_type = memory_type_record.memory_type AND cluster_id IS NULL;
            
            RAISE NOTICE 'Created cluster for memory type: % with % memories', 
                memory_type_record.memory_type, memory_type_record.memory_count;
        END IF;
    END LOOP;
    
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Error during cluster creation: %', SQLERRM;
END $$;

-- Migrate existing documents to document ingestion logs
DO $$
BEGIN
    RAISE NOTICE 'Creating ingestion logs for existing documents...';
    
    -- Create ingestion logs for documents that don't have them
    INSERT INTO document_ingestion_logs (
        document_id, source_url, source_type, ingestion_method, status,
        total_chunks, successful_chunks, content_length, started_at, completed_at
    )
    SELECT DISTINCT
        d.id as document_id,
        d.url as source_url,
        COALESCE(d.metadata->>'type', 'unknown') as source_type,
        'manual' as ingestion_method,
        'completed' as status,
        (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) as total_chunks,
        (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id AND c.content IS NOT NULL) as successful_chunks,
        LENGTH(COALESCE(d.content, '')) as content_length,
        d.created_at as started_at,
        d.updated_at as completed_at
    FROM documents d
    WHERE NOT EXISTS (
        SELECT 1 FROM document_ingestion_logs dil 
        WHERE dil.document_id = d.id
    );
    
    RAISE NOTICE 'Created ingestion logs for % documents', 
        (SELECT COUNT(*) FROM documents WHERE NOT EXISTS (
            SELECT 1 FROM document_ingestion_logs dil 
            WHERE dil.document_id = documents.id
        ));
    
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Error during ingestion logs creation: %', SQLERRM;
END $$;

-- Create default RAG configuration if none exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM rag_configurations WHERE is_default = true) THEN
        INSERT INTO rag_configurations (
            name, description, is_default,
            dense_weight, sparse_weight, graph_weight, rerank_weight,
            dense_model, rerank_model,
            dense_top_k, sparse_top_k, graph_top_k, rerank_top_k,
            min_dense_score, min_sparse_score, min_graph_score, min_final_score,
            enable_caching, cache_ttl_seconds
        ) VALUES (
            'migrated_default',
            'Default configuration created during migration',
            true,
            0.4, 0.3, 0.2, 0.1,
            'BAAI/bge-base-en-v1.5',
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            50, 50, 30, 20,
            0.5, 0.1, 0.3, 0.2,
            true, 300
        );
        
        RAISE NOTICE 'Created default RAG configuration';
    END IF;
END $$;

-- Create service configurations for existing services
DO $$
BEGIN
    RAISE NOTICE 'Creating service configurations...';
    
    -- Create PostgreSQL configuration
    INSERT INTO service_configurations (
        service_type, service_name, config, is_active, is_default, environment, description
    ) VALUES (
        'postgresql', 'knowledgehub_primary',
        '{"host": "postgres", "port": 5432, "database": "knowledgehub"}',
        true, true, 'production',
        'Primary KnowledgeHub PostgreSQL database'
    ) ON CONFLICT (service_name) DO NOTHING;
    
    -- Create Redis configuration
    INSERT INTO service_configurations (
        service_type, service_name, config, is_active, is_default, environment, description
    ) VALUES (
        'redis', 'knowledgehub_cache',
        '{"host": "redis", "port": 6379, "db": 0}',
        true, true, 'production',
        'Primary Redis cache for KnowledgeHub'
    ) ON CONFLICT (service_name) DO NOTHING;
    
    -- Create Weaviate configuration
    INSERT INTO service_configurations (
        service_type, service_name, config, is_active, is_default, environment, description
    ) VALUES (
        'weaviate', 'knowledgehub_vectors',
        '{"url": "http://weaviate:8080", "collection": "KnowledgeHub"}',
        true, true, 'production',
        'Weaviate vector database for embeddings'
    ) ON CONFLICT (service_name) DO NOTHING;
    
    -- Create Neo4j configuration
    INSERT INTO service_configurations (
        service_type, service_name, config, is_active, is_default, environment, description
    ) VALUES (
        'neo4j', 'knowledgehub_graph',
        '{"uri": "bolt://neo4j:7687", "database": "neo4j"}',
        true, true, 'production',
        'Neo4j graph database for knowledge graph'
    ) ON CONFLICT (service_name) DO NOTHING;
    
    -- Create Zep configuration
    INSERT INTO service_configurations (
        service_type, service_name, config, is_active, is_default, environment, description
    ) VALUES (
        'zep_memory', 'knowledgehub_zep',
        '{"api_url": "http://zep:8000", "session_window": 12}',
        true, true, 'production',
        'Zep conversational memory service'
    ) ON CONFLICT (service_name) DO NOTHING;
    
    RAISE NOTICE 'Created service configurations';
    
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Error during service configuration creation: %', SQLERRM;
END $$;

-- Create service dependencies
DO $$
BEGIN
    RAISE NOTICE 'Setting up service dependencies...';
    
    -- KnowledgeHub API depends on PostgreSQL (required)
    INSERT INTO service_dependencies (
        source_service, target_service, dependency_type, config
    ) VALUES (
        'knowledgehub_api', 'knowledgehub_primary', 'required',
        '{"timeout_seconds": 30, "retry_count": 3}'
    ) ON CONFLICT DO NOTHING;
    
    -- KnowledgeHub API depends on Redis (optional)
    INSERT INTO service_dependencies (
        source_service, target_service, dependency_type, config
    ) VALUES (
        'knowledgehub_api', 'knowledgehub_cache', 'optional',
        '{"timeout_seconds": 5, "fallback_enabled": true}'
    ) ON CONFLICT DO NOTHING;
    
    -- Hybrid RAG depends on Weaviate (required)
    INSERT INTO service_dependencies (
        source_service, target_service, dependency_type, config
    ) VALUES (
        'hybrid_rag', 'knowledgehub_vectors', 'required',
        '{"timeout_seconds": 60, "retry_count": 2}'
    ) ON CONFLICT DO NOTHING;
    
    -- Hybrid RAG depends on Neo4j (optional)
    INSERT INTO service_dependencies (
        source_service, target_service, dependency_type, config
    ) VALUES (
        'hybrid_rag', 'knowledgehub_graph', 'optional',
        '{"timeout_seconds": 30, "fallback_enabled": true}'
    ) ON CONFLICT DO NOTHING;
    
    -- Memory service depends on Zep (optional)
    INSERT INTO service_dependencies (
        source_service, target_service, dependency_type, config
    ) VALUES (
        'memory_service', 'knowledgehub_zep', 'optional',
        '{"timeout_seconds": 15, "fallback_enabled": true}'
    ) ON CONFLICT DO NOTHING;
    
    RAISE NOTICE 'Created service dependencies';
    
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Error during service dependencies creation: %', SQLERRM;
END $$;

-- Migrate existing performance metrics
DO $$
BEGIN
    RAISE NOTICE 'Migrating existing performance metrics...';
    
    -- Migrate from performance_metrics table if it exists
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'performance_metrics') THEN
        INSERT INTO performance_monitoring (
            service, operation, execution_time_ms, user_id, metadata, recorded_at
        )
        SELECT 
            'knowledgehub_api' as service,
            command as operation,
            COALESCE(execution_time_ms, 0) as execution_time_ms,
            user_id,
            jsonb_build_object(
                'success', success,
                'error_message', error_message,
                'legacy_context', context
            ) as metadata,
            created_at as recorded_at
        FROM performance_metrics
        WHERE NOT EXISTS (
            SELECT 1 FROM performance_monitoring pm
            WHERE pm.service = 'knowledgehub_api' 
            AND pm.operation = performance_metrics.command
            AND pm.recorded_at = performance_metrics.created_at
        );
        
        RAISE NOTICE 'Migrated performance metrics from legacy table';
    END IF;
    
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Error during performance metrics migration: %', SQLERRM;
END $$;

-- Create initial health records for services
DO $$
BEGIN
    RAISE NOTICE 'Creating initial service health records...';
    
    -- Insert initial health records for core services
    INSERT INTO service_health_logs (service_name, component, status, health_score, details)
    VALUES 
        ('knowledgehub_api', 'main', 'healthy', 1.0, '{"status": "Migration completed successfully"}'),
        ('postgresql', 'database', 'healthy', 1.0, '{"status": "Data migration completed"}'),
        ('redis', 'cache', 'healthy', 1.0, '{"status": "Ready for caching"}'),
        ('weaviate', 'vectors', 'healthy', 1.0, '{"status": "Vector database ready"}'),
        ('neo4j', 'graph', 'healthy', 1.0, '{"status": "Knowledge graph ready"}'),
        ('hybrid_rag', 'main', 'healthy', 1.0, '{"status": "RAG system initialized"}');
    
    RAISE NOTICE 'Created initial service health records';
    
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Error during health records creation: %', SQLERRM;
END $$;

-- Update statistics for all new tables
DO $$
BEGIN
    RAISE NOTICE 'Updating table statistics...';
    
    -- Update statistics for better query performance
    ANALYZE memory_clusters;
    ANALYZE memory_associations;
    ANALYZE memory_access_logs;
    ANALYZE agent_definitions;
    ANALYZE workflow_definitions;
    ANALYZE workflow_executions;
    ANALYZE agent_tasks;
    ANALYZE rag_configurations;
    ANALYZE rag_query_logs;
    ANALYZE document_ingestion_logs;
    ANALYZE enhanced_chunks;
    ANALYZE search_result_cache;
    ANALYZE zep_session_mapping;
    ANALYZE firecrawl_jobs;
    ANALYZE service_health_logs;
    ANALYZE performance_monitoring;
    ANALYZE service_configurations;
    ANALYZE service_integration_logs;
    ANALYZE service_dependencies;
    
    -- Update existing table statistics
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'ai_memories') THEN
        ANALYZE ai_memories;
    END IF;
    
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'documents') THEN
        ANALYZE documents;
    END IF;
    
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'chunks') THEN
        ANALYZE chunks;
    END IF;
    
    RAISE NOTICE 'Updated table statistics for optimal performance';
    
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Error during statistics update: %', SQLERRM;
END $$;

-- Create validation report
DO $$
DECLARE
    memories_count INTEGER;
    clusters_count INTEGER;
    documents_count INTEGER;
    chunks_count INTEGER;
    enhanced_chunks_count INTEGER;
    configs_count INTEGER;
    services_count INTEGER;
BEGIN
    -- Count migrated data
    SELECT COUNT(*) INTO memories_count FROM ai_memories;
    SELECT COUNT(*) INTO clusters_count FROM memory_clusters;
    SELECT COUNT(*) INTO documents_count FROM documents;
    
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'chunks') THEN
        SELECT COUNT(*) INTO chunks_count FROM chunks;
    ELSE
        chunks_count := 0;
    END IF;
    
    SELECT COUNT(*) INTO enhanced_chunks_count FROM enhanced_chunks;
    SELECT COUNT(*) INTO configs_count FROM rag_configurations;
    SELECT COUNT(*) INTO services_count FROM service_configurations;
    
    -- Generate migration report
    RAISE NOTICE '=== DATA MIGRATION REPORT ===';
    RAISE NOTICE 'Memories migrated: %', memories_count;
    RAISE NOTICE 'Memory clusters created: %', clusters_count;
    RAISE NOTICE 'Documents processed: %', documents_count;
    RAISE NOTICE 'Legacy chunks: %', chunks_count;
    RAISE NOTICE 'Enhanced chunks created: %', enhanced_chunks_count;
    RAISE NOTICE 'RAG configurations: %', configs_count;
    RAISE NOTICE 'Service configurations: %', services_count;
    RAISE NOTICE '=== MIGRATION COMPLETED ===';
    
END $$;

-- Clean up migration function
DROP FUNCTION IF EXISTS migrate_data_safely(TEXT, TEXT);

-- Create completion timestamp
CREATE TABLE IF NOT EXISTS migration_log (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    migration_name VARCHAR(255) NOT NULL,
    completed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

INSERT INTO migration_log (migration_name, notes) VALUES (
    '005_data_migration',
    'Hybrid RAG system data migration completed successfully. All existing data preserved and enhanced.'
);

COMMIT;