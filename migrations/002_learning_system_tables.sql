-- Learning System Additional Tables
-- Version: 1.0.0

-- Learning patterns table
CREATE TABLE IF NOT EXISTS learning_patterns (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    pattern_type VARCHAR(100) NOT NULL,
    pattern_data JSONB NOT NULL,
    frequency INTEGER DEFAULT 1,
    confidence_score FLOAT DEFAULT 0.5,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User learning profiles
CREATE TABLE IF NOT EXISTS user_learning_profiles (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES users(id) UNIQUE,
    preferences JSONB DEFAULT '{}',
    strengths JSONB DEFAULT '[]',
    improvement_areas JSONB DEFAULT '[]',
    learning_style VARCHAR(50),
    statistics JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pattern evolution tracking
CREATE TABLE IF NOT EXISTS pattern_evolution (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    pattern_id UUID REFERENCES learning_patterns(id),
    version INTEGER DEFAULT 1,
    changes JSONB DEFAULT '{}',
    effectiveness_score FLOAT DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge transfer records
CREATE TABLE IF NOT EXISTS knowledge_transfers (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    from_context VARCHAR(255),
    to_context VARCHAR(255),
    knowledge_type VARCHAR(100),
    knowledge_data JSONB NOT NULL,
    success_rate FLOAT DEFAULT 0.0,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User feedback tracking
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    session_id UUID REFERENCES memory_sessions(id),
    feedback_type VARCHAR(50),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    context JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Learning sessions
CREATE TABLE IF NOT EXISTS learning_sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    session_type VARCHAR(100),
    objectives JSONB DEFAULT '[]',
    outcomes JSONB DEFAULT '[]',
    insights JSONB DEFAULT '[]',
    duration_minutes INTEGER,
    effectiveness_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Decision outcomes tracking
CREATE TABLE IF NOT EXISTS decision_outcomes (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    decision_id UUID REFERENCES decisions(id),
    outcome_type VARCHAR(50),
    success BOOLEAN,
    impact_score FLOAT,
    lessons_learned TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_learning_patterns_user_id ON learning_patterns(user_id);
CREATE INDEX idx_learning_patterns_type ON learning_patterns(pattern_type);
CREATE INDEX idx_pattern_evolution_pattern_id ON pattern_evolution(pattern_id);
CREATE INDEX idx_knowledge_transfers_contexts ON knowledge_transfers(from_context, to_context);
CREATE INDEX idx_user_feedback_user_id ON user_feedback(user_id);
CREATE INDEX idx_learning_sessions_user_id ON learning_sessions(user_id);
CREATE INDEX idx_decision_outcomes_decision_id ON decision_outcomes(decision_id);