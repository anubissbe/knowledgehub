-- Memory System Schema for KnowledgeHub
-- Creates tables for persistent Claude-Code memory

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create memory_sessions table
CREATE TABLE IF NOT EXISTS memory_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    project_id UUID,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    parent_session_id UUID REFERENCES memory_sessions(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT ck_sessions_valid_duration CHECK (ended_at IS NULL OR ended_at >= started_at)
);

-- Create indexes for memory_sessions
CREATE INDEX IF NOT EXISTS idx_memory_sessions_user_id ON memory_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_sessions_project_id ON memory_sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_memory_sessions_started_at ON memory_sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_memory_sessions_parent_id ON memory_sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_memory_sessions_tags ON memory_sessions USING gin(tags);
CREATE INDEX IF NOT EXISTS idx_memory_sessions_metadata ON memory_sessions USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_memory_sessions_user_project_time ON memory_sessions(user_id, project_id, started_at);

-- Create memory type enum
CREATE TYPE memory_type AS ENUM ('fact', 'preference', 'code', 'decision', 'error', 'pattern', 'entity');

-- Create memories table
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES memory_sessions(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    summary TEXT,
    memory_type memory_type NOT NULL,
    importance FLOAT NOT NULL DEFAULT 0.5,
    confidence FLOAT NOT NULL DEFAULT 0.8,
    entities TEXT[] DEFAULT '{}',
    related_memories UUID[] DEFAULT '{}',
    embedding FLOAT[],
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT ck_memories_importance_range CHECK (importance >= 0 AND importance <= 1),
    CONSTRAINT ck_memories_confidence_range CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT ck_memories_access_count_positive CHECK (access_count >= 0)
);

-- Create indexes for memories
CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_entities ON memories USING gin(entities);
CREATE INDEX IF NOT EXISTS idx_memories_metadata ON memories USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed);
CREATE INDEX IF NOT EXISTS idx_memories_session_type_importance ON memories(session_id, memory_type, importance);
CREATE INDEX IF NOT EXISTS idx_memories_type_importance_created ON memories(memory_type, importance, created_at);

-- Create update trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_memory_sessions_updated_at 
    BEFORE UPDATE ON memory_sessions
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_memories_updated_at 
    BEFORE UPDATE ON memories
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Add comments
COMMENT ON TABLE memory_sessions IS 'Stores Claude-Code conversation sessions for memory continuity';
COMMENT ON TABLE memories IS 'Stores extracted memories from Claude-Code conversations';
COMMENT ON COLUMN memory_sessions.user_id IS 'Unique identifier for the user';
COMMENT ON COLUMN memory_sessions.project_id IS 'Optional project association';
COMMENT ON COLUMN memories.memory_type IS 'Type of memory for categorization';
COMMENT ON COLUMN memories.importance IS 'Importance score from 0.0 to 1.0';
COMMENT ON COLUMN memories.embedding IS 'Vector embedding for semantic search';