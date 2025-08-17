-- Hybrid RAG System Database Migration
-- Version: 2.0.0 - Enhanced Memory and Multi-Agent Workflow Support

-- Create extensions for enhanced functionality
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Enhanced Memory System Tables

-- Update existing ai_memories table to support enhanced memory model
DO $$
BEGIN
    -- Check if ai_memories table exists and migrate to new structure
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'ai_memories') THEN
        -- Add new columns if they don't exist
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_memories' AND column_name = 'content_hash') THEN
            ALTER TABLE ai_memories ADD COLUMN content_hash VARCHAR(64);
            CREATE INDEX idx_ai_memories_content_hash ON ai_memories(content_hash);
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_memories' AND column_name = 'embeddings') THEN
            ALTER TABLE ai_memories ADD COLUMN embeddings vector(768);
            CREATE INDEX idx_ai_memories_embeddings ON ai_memories USING ivfflat (embeddings vector_cosine_ops);
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_memories' AND column_name = 'cluster_id') THEN
            ALTER TABLE ai_memories ADD COLUMN cluster_id UUID;
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_memories' AND column_name = 'knowledge_entities') THEN
            ALTER TABLE ai_memories ADD COLUMN knowledge_entities JSONB DEFAULT '[]';
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_memories' AND column_name = 'knowledge_relations') THEN
            ALTER TABLE ai_memories ADD COLUMN knowledge_relations JSONB DEFAULT '[]';
        END IF;
    END IF;
END
$$;

-- Memory Clusters for enhanced organization
CREATE TABLE IF NOT EXISTS memory_clusters (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    cluster_type VARCHAR(50) NOT NULL,
    centroid_embedding vector(768),
    topic_keywords JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Statistics
    memory_count INTEGER DEFAULT 0,
    avg_relevance FLOAT DEFAULT 0.0,
    last_accessed TIMESTAMPTZ
);

-- Memory Associations for graph-like relationships
CREATE TABLE IF NOT EXISTS memory_associations (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    source_memory_id UUID NOT NULL,
    target_memory_id UUID NOT NULL,
    
    association_type VARCHAR(50) NOT NULL,
    strength FLOAT DEFAULT 1.0,
    confidence FLOAT DEFAULT 1.0,
    
    context JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_reinforced TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    reinforcement_count INTEGER DEFAULT 0
);

-- Memory Access Logs for analytics
CREATE TABLE IF NOT EXISTS memory_access_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    memory_id UUID NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    
    access_type VARCHAR(50) NOT NULL,
    context_similarity FLOAT,
    retrieval_method VARCHAR(50),
    
    query_context JSONB DEFAULT '{}',
    response_time_ms FLOAT,
    result_rank INTEGER,
    
    accessed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Agent Workflow System Tables

-- Agent Definitions and Configurations
CREATE TABLE IF NOT EXISTS agent_definitions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    role VARCHAR(50) NOT NULL,
    description TEXT,
    
    -- Capabilities and Configuration
    capabilities JSONB DEFAULT '[]',
    tools_available JSONB DEFAULT '[]',
    model_config JSONB DEFAULT '{}',
    system_prompt TEXT,
    
    -- Performance and Limits
    max_concurrent_tasks INTEGER DEFAULT 1,
    timeout_seconds INTEGER DEFAULT 300,
    rate_limit_per_minute INTEGER DEFAULT 60,
    
    -- State
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Workflow Definitions
CREATE TABLE IF NOT EXISTS workflow_definitions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    workflow_type VARCHAR(50) NOT NULL,
    
    -- Graph Definition
    graph_definition JSONB NOT NULL,
    entry_point VARCHAR(100) NOT NULL,
    exit_points JSONB DEFAULT '[]',
    
    -- Configuration
    config JSONB DEFAULT '{}',
    agents_required JSONB DEFAULT '[]',
    tools_required JSONB DEFAULT '[]',
    
    -- Metadata
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    created_by UUID,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Workflow Executions
CREATE TABLE IF NOT EXISTS workflow_executions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    workflow_id UUID NOT NULL,
    session_id VARCHAR(255),
    user_id VARCHAR(255) NOT NULL,
    
    -- Execution Details
    execution_type VARCHAR(50) DEFAULT 'synchronous',
    status VARCHAR(50) DEFAULT 'pending',
    
    -- Input/Output
    input_data JSONB NOT NULL,
    output_data JSONB,
    error_details JSONB,
    
    -- Execution Metadata
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    execution_time_ms INTEGER,
    
    -- State Management (for LangGraph checkpoints)
    checkpoint_data JSONB,
    current_state JSONB DEFAULT '{}',
    
    -- Performance Tracking
    steps_completed INTEGER DEFAULT 0,
    agents_involved JSONB DEFAULT '[]',
    tools_used JSONB DEFAULT '[]'
);

-- Agent Tasks (individual tasks within workflows)
CREATE TABLE IF NOT EXISTS agent_tasks (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    workflow_execution_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    
    -- Task Details
    task_type VARCHAR(50) NOT NULL,
    step_name VARCHAR(100),
    input_data JSONB NOT NULL,
    output_data JSONB,
    
    -- Execution
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    execution_time_ms INTEGER,
    
    -- Error Handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Dependencies
    depends_on_tasks JSONB DEFAULT '[]',
    blocked_by_tasks JSONB DEFAULT '[]'
);

-- Hybrid RAG System Tables

-- RAG Configurations
CREATE TABLE IF NOT EXISTS rag_configurations (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    
    -- Retrieval Configuration
    dense_weight FLOAT DEFAULT 0.4,
    sparse_weight FLOAT DEFAULT 0.3,
    graph_weight FLOAT DEFAULT 0.2,
    rerank_weight FLOAT DEFAULT 0.1,
    
    -- Model Configuration
    dense_model VARCHAR(200) DEFAULT 'BAAI/bge-base-en-v1.5',
    rerank_model VARCHAR(200) DEFAULT 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    
    -- Performance Tuning
    dense_top_k INTEGER DEFAULT 50,
    sparse_top_k INTEGER DEFAULT 50,
    graph_top_k INTEGER DEFAULT 30,
    rerank_top_k INTEGER DEFAULT 20,
    
    -- Quality Thresholds
    min_dense_score FLOAT DEFAULT 0.5,
    min_sparse_score FLOAT DEFAULT 0.1,
    min_graph_score FLOAT DEFAULT 0.3,
    min_final_score FLOAT DEFAULT 0.2,
    
    -- Caching and Performance
    enable_caching BOOLEAN DEFAULT true,
    cache_ttl_seconds INTEGER DEFAULT 300,
    
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- RAG Query Logs
CREATE TABLE IF NOT EXISTS rag_query_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    
    -- Query Details
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64) NOT NULL,
    retrieval_mode VARCHAR(50) NOT NULL,
    config_id UUID,
    
    -- Results Metadata
    total_results INTEGER DEFAULT 0,
    dense_results_count INTEGER DEFAULT 0,
    sparse_results_count INTEGER DEFAULT 0,
    graph_results_count INTEGER DEFAULT 0,
    final_results_count INTEGER DEFAULT 0,
    
    -- Performance Metrics
    total_time_ms INTEGER NOT NULL,
    dense_retrieval_time_ms INTEGER DEFAULT 0,
    sparse_retrieval_time_ms INTEGER DEFAULT 0,
    graph_retrieval_time_ms INTEGER DEFAULT 0,
    rerank_time_ms INTEGER DEFAULT 0,
    
    -- Quality Metrics
    avg_relevance_score FLOAT,
    cache_hit BOOLEAN DEFAULT false,
    user_feedback_score INTEGER, -- 1-5 rating if provided
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Enhanced Document and Chunk Management

-- Document Ingestion Tracking
CREATE TABLE IF NOT EXISTS document_ingestion_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    document_id UUID NOT NULL,
    source_url TEXT,
    source_type VARCHAR(50),
    
    -- Ingestion Process
    ingestion_method VARCHAR(50) NOT NULL, -- 'firecrawl', 'manual', 'api'
    status VARCHAR(50) DEFAULT 'pending',
    
    -- Processing Results
    total_chunks INTEGER DEFAULT 0,
    successful_chunks INTEGER DEFAULT 0,
    failed_chunks INTEGER DEFAULT 0,
    
    -- Content Analysis
    entities_extracted JSONB DEFAULT '[]',
    content_length INTEGER,
    language VARCHAR(10),
    content_type VARCHAR(100),
    
    -- Performance
    processing_time_ms INTEGER,
    
    -- Error Handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

-- Enhanced Chunk Metadata
CREATE TABLE IF NOT EXISTS enhanced_chunks (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    document_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    
    -- Content
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    content_length INTEGER NOT NULL,
    
    -- Embeddings (multiple models support)
    embedding_primary vector(768),
    embedding_secondary vector(384), -- For smaller/faster models
    embedding_model VARCHAR(200) DEFAULT 'BAAI/bge-base-en-v1.5',
    
    -- Semantic Metadata
    entities JSONB DEFAULT '[]',
    keywords JSONB DEFAULT '[]',
    topics JSONB DEFAULT '[]',
    sentiment_score FLOAT,
    complexity_score FLOAT,
    
    -- Graph Relationships
    related_chunks JSONB DEFAULT '[]',
    graph_node_id VARCHAR(100),
    
    -- Quality Metrics
    coherence_score FLOAT,
    informativeness_score FLOAT,
    
    -- Chunk Hierarchy
    parent_chunk_id UUID,
    child_chunks JSONB DEFAULT '[]',
    hierarchy_level INTEGER DEFAULT 0,
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Search Result Caching for Performance
CREATE TABLE IF NOT EXISTS search_result_cache (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    query_hash VARCHAR(64) NOT NULL,
    config_hash VARCHAR(64) NOT NULL,
    user_context_hash VARCHAR(64),
    
    -- Cached Results
    results_data JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    
    -- Cache Management
    hit_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Service Integration Tables

-- Zep Memory Integration
CREATE TABLE IF NOT EXISTS zep_session_mapping (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    knowledgehub_session_id VARCHAR(255) NOT NULL UNIQUE,
    zep_session_id VARCHAR(255) NOT NULL UNIQUE,
    user_id VARCHAR(255) NOT NULL,
    
    -- Sync State
    last_sync_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    sync_status VARCHAR(50) DEFAULT 'active',
    
    -- Configuration
    zep_config JSONB DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Firecrawl Integration
CREATE TABLE IF NOT EXISTS firecrawl_jobs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    job_id VARCHAR(255) NOT NULL UNIQUE,
    url TEXT NOT NULL,
    crawl_type VARCHAR(50) NOT NULL, -- 'single', 'map', 'crawl'
    
    -- Job Configuration
    config JSONB NOT NULL,
    
    -- Status Tracking
    status VARCHAR(50) DEFAULT 'pending',
    progress FLOAT DEFAULT 0.0,
    
    -- Results
    pages_found INTEGER DEFAULT 0,
    pages_processed INTEGER DEFAULT 0,
    documents_created INTEGER DEFAULT 0,
    
    -- Error Handling
    error_message TEXT,
    
    -- Timing
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    
    -- Integration
    created_by_user_id VARCHAR(255)
);

-- System Health and Monitoring Tables

-- Service Health Monitoring
CREATE TABLE IF NOT EXISTS service_health_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL,
    component VARCHAR(100),
    
    -- Health Status
    status VARCHAR(50) NOT NULL, -- 'healthy', 'degraded', 'unhealthy'
    health_score FLOAT, -- 0.0 to 1.0
    
    -- Metrics
    response_time_ms FLOAT,
    memory_usage_mb FLOAT,
    cpu_usage_percent FLOAT,
    active_connections INTEGER,
    
    -- Details
    details JSONB DEFAULT '{}',
    error_message TEXT,
    
    checked_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Performance Monitoring
CREATE TABLE IF NOT EXISTS performance_monitoring (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    service VARCHAR(100) NOT NULL,
    operation VARCHAR(100) NOT NULL,
    
    -- Performance Metrics
    execution_time_ms INTEGER NOT NULL,
    memory_used_mb FLOAT,
    cpu_usage_percent FLOAT,
    
    -- Context
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create comprehensive indexes for performance

-- Memory system indexes
CREATE INDEX IF NOT EXISTS idx_memory_clusters_type ON memory_clusters(cluster_type);
CREATE INDEX IF NOT EXISTS idx_memory_clusters_updated ON memory_clusters(updated_at);

CREATE INDEX IF NOT EXISTS idx_memory_associations_source ON memory_associations(source_memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_associations_target ON memory_associations(target_memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_associations_type_strength ON memory_associations(association_type, strength);

CREATE INDEX IF NOT EXISTS idx_memory_access_logs_memory ON memory_access_logs(memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_access_logs_user_session ON memory_access_logs(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_memory_access_logs_accessed_at ON memory_access_logs(accessed_at);

-- Agent workflow indexes
CREATE INDEX IF NOT EXISTS idx_agent_definitions_role ON agent_definitions(role);
CREATE INDEX IF NOT EXISTS idx_agent_definitions_active ON agent_definitions(is_active);

CREATE INDEX IF NOT EXISTS idx_workflow_definitions_type ON workflow_definitions(workflow_type);
CREATE INDEX IF NOT EXISTS idx_workflow_definitions_active ON workflow_definitions(is_active);

CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_id ON workflow_executions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_user_id ON workflow_executions(user_id);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON workflow_executions(status);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_started_at ON workflow_executions(started_at);

CREATE INDEX IF NOT EXISTS idx_agent_tasks_workflow_execution ON agent_tasks(workflow_execution_id);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_agent_id ON agent_tasks(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_status ON agent_tasks(status);

-- RAG system indexes
CREATE INDEX IF NOT EXISTS idx_rag_configurations_default ON rag_configurations(is_default);

CREATE INDEX IF NOT EXISTS idx_rag_query_logs_user_id ON rag_query_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_rag_query_logs_query_hash ON rag_query_logs(query_hash);
CREATE INDEX IF NOT EXISTS idx_rag_query_logs_created_at ON rag_query_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_rag_query_logs_performance ON rag_query_logs(total_time_ms, total_results);

-- Document processing indexes
CREATE INDEX IF NOT EXISTS idx_document_ingestion_document_id ON document_ingestion_logs(document_id);
CREATE INDEX IF NOT EXISTS idx_document_ingestion_status ON document_ingestion_logs(status);
CREATE INDEX IF NOT EXISTS idx_document_ingestion_method ON document_ingestion_logs(ingestion_method);

CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_document_id ON enhanced_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_content_hash ON enhanced_chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_embedding_primary ON enhanced_chunks USING ivfflat (embedding_primary vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_embedding_secondary ON enhanced_chunks USING ivfflat (embedding_secondary vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_parent ON enhanced_chunks(parent_chunk_id);
CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_hierarchy ON enhanced_chunks(hierarchy_level);

-- Cache indexes
CREATE INDEX IF NOT EXISTS idx_search_result_cache_query_hash ON search_result_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_search_result_cache_expires_at ON search_result_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_search_result_cache_hit_count ON search_result_cache(hit_count);

-- Service integration indexes  
CREATE INDEX IF NOT EXISTS idx_zep_session_mapping_kh_session ON zep_session_mapping(knowledgehub_session_id);
CREATE INDEX IF NOT EXISTS idx_zep_session_mapping_zep_session ON zep_session_mapping(zep_session_id);
CREATE INDEX IF NOT EXISTS idx_zep_session_mapping_user_id ON zep_session_mapping(user_id);

CREATE INDEX IF NOT EXISTS idx_firecrawl_jobs_job_id ON firecrawl_jobs(job_id);
CREATE INDEX IF NOT EXISTS idx_firecrawl_jobs_status ON firecrawl_jobs(status);
CREATE INDEX IF NOT EXISTS idx_firecrawl_jobs_started_at ON firecrawl_jobs(started_at);

-- Monitoring indexes
CREATE INDEX IF NOT EXISTS idx_service_health_logs_service ON service_health_logs(service_name);
CREATE INDEX IF NOT EXISTS idx_service_health_logs_status ON service_health_logs(status);
CREATE INDEX IF NOT EXISTS idx_service_health_logs_checked_at ON service_health_logs(checked_at);

CREATE INDEX IF NOT EXISTS idx_performance_monitoring_service_operation ON performance_monitoring(service, operation);
CREATE INDEX IF NOT EXISTS idx_performance_monitoring_recorded_at ON performance_monitoring(recorded_at);
CREATE INDEX IF NOT EXISTS idx_performance_monitoring_execution_time ON performance_monitoring(execution_time_ms);

-- Add foreign key constraints where appropriate
DO $$
BEGIN
    -- Memory system constraints
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'ai_memories') THEN
        IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_ai_memories_cluster_id') THEN
            ALTER TABLE ai_memories ADD CONSTRAINT fk_ai_memories_cluster_id 
                FOREIGN KEY (cluster_id) REFERENCES memory_clusters(id) ON DELETE SET NULL;
        END IF;
    END IF;
    
    -- Only add constraints if referenced tables exist
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'ai_memories') THEN
        IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_memory_associations_source') THEN
            ALTER TABLE memory_associations ADD CONSTRAINT fk_memory_associations_source 
                FOREIGN KEY (source_memory_id) REFERENCES ai_memories(id) ON DELETE CASCADE;
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_memory_associations_target') THEN
            ALTER TABLE memory_associations ADD CONSTRAINT fk_memory_associations_target 
                FOREIGN KEY (target_memory_id) REFERENCES ai_memories(id) ON DELETE CASCADE;
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_memory_access_logs_memory') THEN
            ALTER TABLE memory_access_logs ADD CONSTRAINT fk_memory_access_logs_memory 
                FOREIGN KEY (memory_id) REFERENCES ai_memories(id) ON DELETE CASCADE;
        END IF;
    END IF;
    
    -- Workflow system constraints
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_workflow_executions_workflow') THEN
        ALTER TABLE workflow_executions ADD CONSTRAINT fk_workflow_executions_workflow 
            FOREIGN KEY (workflow_id) REFERENCES workflow_definitions(id) ON DELETE CASCADE;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_agent_tasks_workflow_execution') THEN
        ALTER TABLE agent_tasks ADD CONSTRAINT fk_agent_tasks_workflow_execution 
            FOREIGN KEY (workflow_execution_id) REFERENCES workflow_executions(id) ON DELETE CASCADE;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_agent_tasks_agent') THEN
        ALTER TABLE agent_tasks ADD CONSTRAINT fk_agent_tasks_agent 
            FOREIGN KEY (agent_id) REFERENCES agent_definitions(id) ON DELETE CASCADE;
    END IF;
    
    -- RAG system constraints
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_rag_query_logs_config') THEN
        ALTER TABLE rag_query_logs ADD CONSTRAINT fk_rag_query_logs_config 
            FOREIGN KEY (config_id) REFERENCES rag_configurations(id) ON DELETE SET NULL;
    END IF;
    
    -- Document processing constraints
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'documents') THEN
        IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_document_ingestion_logs_document') THEN
            ALTER TABLE document_ingestion_logs ADD CONSTRAINT fk_document_ingestion_logs_document 
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE;
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_enhanced_chunks_document') THEN
            ALTER TABLE enhanced_chunks ADD CONSTRAINT fk_enhanced_chunks_document 
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE;
        END IF;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_enhanced_chunks_parent') THEN
        ALTER TABLE enhanced_chunks ADD CONSTRAINT fk_enhanced_chunks_parent 
            FOREIGN KEY (parent_chunk_id) REFERENCES enhanced_chunks(id) ON DELETE SET NULL;
    END IF;
    
EXCEPTION WHEN OTHERS THEN
    -- Log error but continue - some constraints might fail if referenced tables don't exist yet
    RAISE NOTICE 'Some foreign key constraints could not be added: %', SQLERRM;
END
$$;

-- Insert default configurations

-- Default RAG configuration
INSERT INTO rag_configurations (
    name, description, is_default,
    dense_weight, sparse_weight, graph_weight, rerank_weight,
    dense_model, rerank_model,
    dense_top_k, sparse_top_k, graph_top_k, rerank_top_k
) VALUES (
    'default_hybrid_rag',
    'Default hybrid RAG configuration balancing all retrieval methods',
    true,
    0.4, 0.3, 0.2, 0.1,
    'BAAI/bge-base-en-v1.5',
    'cross-encoder/ms-marco-MiniLM-L-6-v2',
    50, 50, 30, 20
) ON CONFLICT (name) DO NOTHING;

-- Default agent definitions
INSERT INTO agent_definitions (name, role, description, capabilities, tools_available) VALUES
('researcher', 'RESEARCHER', 'Information gathering specialist', 
 '["information_gathering", "source_identification", "context_expansion"]',
 '["search_knowledge", "search_memory", "web_search"]'),
('analyst', 'ANALYST', 'Data analysis and pattern recognition specialist',
 '["pattern_recognition", "data_interpretation", "insight_generation"]', 
 '["validate_information", "pattern_analysis", "data_processing"]'),
('synthesizer', 'SYNTHESIZER', 'Information integration and response generation specialist',
 '["information_integration", "response_structuring", "content_organization"]',
 '["content_generation", "structure_optimization", "formatting"]'),
('validator', 'VALIDATOR', 'Quality assurance and fact-checking specialist',
 '["fact_checking", "source_verification", "accuracy_assessment"]',
 '["validate_information", "fact_check", "source_verification"]')
ON CONFLICT (name) DO NOTHING;

-- Default workflow definitions
INSERT INTO workflow_definitions (name, description, workflow_type, graph_definition, entry_point) VALUES
('simple_qa', 'Simple question-answering workflow',
 'SIMPLE_QA',
 '{"nodes": ["researcher", "synthesizer"], "edges": [["researcher", "synthesizer"]], "conditional_edges": {}}',
 'researcher'),
('multi_step_research', 'Multi-step research and analysis workflow',
 'MULTI_STEP_RESEARCH', 
 '{"nodes": ["researcher", "analyst", "synthesizer", "validator"], "edges": [["researcher", "analyst"], ["analyst", "synthesizer"], ["synthesizer", "validator"]], "conditional_edges": {}}',
 'researcher')
ON CONFLICT (name) DO NOTHING;

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables with updated_at columns
DO $$
DECLARE
    tables_with_updated_at TEXT[] := ARRAY[
        'memory_clusters', 'agent_definitions', 'workflow_definitions', 
        'enhanced_chunks', 'zep_session_mapping', 'rag_configurations'
    ];
    table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY tables_with_updated_at
    LOOP
        IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = table_name) THEN
            EXECUTE format('DROP TRIGGER IF EXISTS update_%s_updated_at ON %s', table_name, table_name);
            EXECUTE format('CREATE TRIGGER update_%s_updated_at BEFORE UPDATE ON %s FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column()', table_name, table_name);
        END IF;
    END LOOP;
END
$$;

-- Create views for common queries

-- Enhanced Memory View
CREATE OR REPLACE VIEW enhanced_memory_view AS
SELECT 
    m.id,
    m.user_id,
    m.session_id,
    m.content,
    m.memory_type,
    m.relevance_score,
    m.importance,
    m.access_count,
    m.created_at,
    m.last_accessed,
    c.name as cluster_name,
    c.cluster_type,
    array_length(m.knowledge_entities::jsonb, 1) as entity_count,
    array_length(m.knowledge_relations::jsonb, 1) as relation_count
FROM ai_memories m
LEFT JOIN memory_clusters c ON m.cluster_id = c.id
WHERE m.is_archived = false;

-- Workflow Performance View
CREATE OR REPLACE VIEW workflow_performance_view AS
SELECT 
    wd.name as workflow_name,
    wd.workflow_type,
    COUNT(we.id) as total_executions,
    AVG(we.execution_time_ms) as avg_execution_time,
    COUNT(CASE WHEN we.status = 'completed' THEN 1 END) as successful_executions,
    COUNT(CASE WHEN we.status = 'failed' THEN 1 END) as failed_executions,
    (COUNT(CASE WHEN we.status = 'completed' THEN 1 END) * 100.0 / COUNT(we.id)) as success_rate,
    MAX(we.started_at) as last_execution
FROM workflow_definitions wd
LEFT JOIN workflow_executions we ON wd.id = we.workflow_id
WHERE wd.is_active = true
GROUP BY wd.id, wd.name, wd.workflow_type;

-- RAG Performance View
CREATE OR REPLACE VIEW rag_performance_view AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    COUNT(*) as total_queries,
    AVG(total_time_ms) as avg_response_time,
    AVG(total_results) as avg_results_count,
    COUNT(CASE WHEN cache_hit THEN 1 END) as cache_hits,
    (COUNT(CASE WHEN cache_hit THEN 1 END) * 100.0 / COUNT(*)) as cache_hit_rate,
    AVG(avg_relevance_score) as avg_relevance
FROM rag_query_logs
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY hour DESC;

-- Service Health Summary View
CREATE OR REPLACE VIEW service_health_summary AS
SELECT 
    service_name,
    component,
    status,
    AVG(health_score) as avg_health_score,
    AVG(response_time_ms) as avg_response_time,
    COUNT(*) as check_count,
    MAX(checked_at) as last_check
FROM service_health_logs
WHERE checked_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
GROUP BY service_name, component, status
ORDER BY service_name, component;

-- Grant permissions
DO $$
DECLARE
    username TEXT := 'knowledgehub';
BEGIN
    -- Grant permissions on all tables
    EXECUTE format('GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO %I', username);
    
    -- Grant specific permissions on new tables
    EXECUTE format('GRANT ALL PRIVILEGES ON memory_clusters TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON memory_associations TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON memory_access_logs TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON agent_definitions TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON workflow_definitions TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON workflow_executions TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON agent_tasks TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON rag_configurations TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON rag_query_logs TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON document_ingestion_logs TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON enhanced_chunks TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON search_result_cache TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON zep_session_mapping TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON firecrawl_jobs TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON service_health_logs TO %I', username);
    EXECUTE format('GRANT ALL PRIVILEGES ON performance_monitoring TO %I', username);
    
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Could not grant all permissions: %', SQLERRM;
END
$$;

COMMIT;