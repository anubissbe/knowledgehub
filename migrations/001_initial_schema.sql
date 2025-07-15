-- KnowledgeHub Initial Database Schema
-- Version: 1.0.0

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create base tables
CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Memory System Tables
CREATE TABLE IF NOT EXISTS memory_sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    context JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true
);

CREATE TABLE IF NOT EXISTS memories (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES memory_sessions(id),
    user_id UUID REFERENCES users(id),
    content TEXT NOT NULL,
    embedding vector(768),
    importance_score FLOAT DEFAULT 0.5,
    context JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Learning System Tables
CREATE TABLE IF NOT EXISTS mistake_patterns (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    error_type VARCHAR(255) NOT NULL,
    error_message TEXT,
    solution TEXT,
    context JSONB DEFAULT '{}',
    frequency INTEGER DEFAULT 1,
    last_occurrence TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decisions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    decision_type VARCHAR(255),
    chosen_option TEXT NOT NULL,
    alternatives JSONB DEFAULT '[]',
    reasoning TEXT,
    confidence_score FLOAT DEFAULT 0.5,
    outcome VARCHAR(50),
    context JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    command TEXT NOT NULL,
    execution_time_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    context JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Code Evolution Tables
CREATE TABLE IF NOT EXISTS code_changes (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    file_path TEXT NOT NULL,
    change_type VARCHAR(50),
    before_content TEXT,
    after_content TEXT,
    reasoning TEXT,
    impact_score FLOAT DEFAULT 0.5,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Project Context Tables
CREATE TABLE IF NOT EXISTS project_contexts (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    project_path TEXT NOT NULL,
    project_type VARCHAR(100),
    technologies JSONB DEFAULT '[]',
    patterns JSONB DEFAULT '{}',
    preferences JSONB DEFAULT '{}',
    statistics JSONB DEFAULT '{}',
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, project_path)
);

-- Workflow Tables
CREATE TABLE IF NOT EXISTS workflow_captures (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    workflow_name VARCHAR(255),
    steps JSONB DEFAULT '[]',
    frequency INTEGER DEFAULT 1,
    last_executed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge Source Tables
CREATE TABLE IF NOT EXISTS knowledge_sources (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    url TEXT,
    configuration JSONB DEFAULT '{}',
    last_synced TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    source_id UUID REFERENCES knowledge_sources(id),
    title VARCHAR(500),
    content TEXT,
    url TEXT,
    metadata JSONB DEFAULT '{}',
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    content TEXT NOT NULL,
    embedding vector(768),
    position INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Job Management Tables
CREATE TABLE IF NOT EXISTS jobs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    payload JSONB DEFAULT '{}',
    result JSONB,
    error TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_memories_user_id ON memories(user_id);
CREATE INDEX idx_memories_session_id ON memories(session_id);
CREATE INDEX idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_memories_created_at ON memories(created_at DESC);

CREATE INDEX idx_mistake_patterns_user_id ON mistake_patterns(user_id);
CREATE INDEX idx_mistake_patterns_error_type ON mistake_patterns(error_type);

CREATE INDEX idx_decisions_user_id ON decisions(user_id);
CREATE INDEX idx_decisions_created_at ON decisions(created_at DESC);

CREATE INDEX idx_performance_metrics_user_id ON performance_metrics(user_id);
CREATE INDEX idx_performance_metrics_command ON performance_metrics(command);

CREATE INDEX idx_code_changes_user_id ON code_changes(user_id);
CREATE INDEX idx_code_changes_file_path ON code_changes(file_path);

CREATE INDEX idx_project_contexts_user_project ON project_contexts(user_id, project_path);

CREATE INDEX idx_documents_source_id ON documents(source_id);
CREATE INDEX idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_type ON jobs(type);

-- Create default user
INSERT INTO users (id, username, email) 
VALUES ('00000000-0000-0000-0000-000000000000', 'default', 'default@knowledgehub.local')
ON CONFLICT (username) DO NOTHING;