-- AI Knowledge Hub Database Schema
-- PostgreSQL 15+

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create custom types
CREATE TYPE source_status AS ENUM ('pending', 'crawling', 'indexing', 'completed', 'error', 'paused');
CREATE TYPE job_status AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled');
CREATE TYPE job_type AS ENUM ('scraping', 'reindexing', 'deletion');
CREATE TYPE chunk_type AS ENUM ('TEXT', 'CODE', 'TABLE', 'LIST', 'HEADING');

-- Knowledge sources table
CREATE TABLE IF NOT EXISTS knowledge_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    url TEXT NOT NULL UNIQUE,
    status source_status DEFAULT 'pending',
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_scraped_at TIMESTAMP WITH TIME ZONE,
    stats JSONB DEFAULT '{"documents": 0, "chunks": 0, "errors": 0}'
);

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES knowledge_sources(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title TEXT,
    content TEXT,
    content_hash VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'pending',
    indexed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, url)
);

-- Document chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_type chunk_type NOT NULL,
    content TEXT NOT NULL,
    embedding_id VARCHAR(255),
    parent_heading TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, chunk_index)
);

-- Scraping jobs table
CREATE TABLE IF NOT EXISTS scraping_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES knowledge_sources(id) ON DELETE CASCADE,
    job_type job_type NOT NULL DEFAULT 'scraping',
    status job_status DEFAULT 'pending',
    config JSONB DEFAULT '{}',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error TEXT,
    stats JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Memory items table (for MCP memory storage)
CREATE TABLE IF NOT EXISTS memory_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    content_hash VARCHAR(64) UNIQUE,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    embedding_id VARCHAR(255),
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- API keys table (for authentication)
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    permissions JSONB DEFAULT '["read"]',
    last_used_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Search history table (for analytics and suggestions)
CREATE TABLE IF NOT EXISTS search_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    results_count INTEGER DEFAULT 0,
    search_type VARCHAR(50) DEFAULT 'hybrid',
    filters JSONB DEFAULT '{}',
    execution_time_ms FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_sources_status ON knowledge_sources(status);
CREATE INDEX idx_sources_updated_at ON knowledge_sources(updated_at DESC);

CREATE INDEX idx_documents_source_id ON documents(source_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_content_hash ON documents(content_hash);
CREATE INDEX idx_documents_url ON documents(url);

CREATE INDEX idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_chunks_type ON document_chunks(chunk_type);
CREATE INDEX idx_chunks_embedding_id ON document_chunks(embedding_id);

CREATE INDEX idx_jobs_source_id ON scraping_jobs(source_id);
CREATE INDEX idx_jobs_status ON scraping_jobs(status);
CREATE INDEX idx_jobs_created_at ON scraping_jobs(created_at DESC);

CREATE INDEX idx_memory_tags ON memory_items USING GIN(tags);
CREATE INDEX idx_memory_content_hash ON memory_items(content_hash);
CREATE INDEX idx_memory_accessed_at ON memory_items(accessed_at DESC);

CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_active ON api_keys(is_active);

CREATE INDEX idx_search_query ON search_history USING GIN(to_tsvector('english', query));
CREATE INDEX idx_search_created_at ON search_history(created_at DESC);

-- Create update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to relevant tables
CREATE TRIGGER update_knowledge_sources_updated_at BEFORE UPDATE
    ON knowledge_sources FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE
    ON documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_memory_items_updated_at BEFORE UPDATE
    ON memory_items FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE OR REPLACE VIEW source_statistics AS
SELECT 
    ks.id,
    ks.name,
    ks.url,
    ks.status,
    COUNT(DISTINCT d.id) as document_count,
    COUNT(DISTINCT dc.id) as chunk_count,
    MAX(d.created_at) as last_document_added,
    ks.last_scraped_at
FROM knowledge_sources ks
LEFT JOIN documents d ON ks.id = d.source_id
LEFT JOIN document_chunks dc ON d.id = dc.document_id
GROUP BY ks.id;

-- Insert default API key for development (remove in production)
INSERT INTO api_keys (name, key_hash, permissions) 
VALUES (
    'Development Key',
    '$2b$12$X3K9LJM7qK.Uh9wD5YK.8e7xvU5p9Yvhfc5SQ4XNjK7R9bMZ5XNK2', -- hash of 'dev-api-key-123'
    '["read", "write", "admin"]'
);