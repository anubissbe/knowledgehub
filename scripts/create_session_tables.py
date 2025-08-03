#!/usr/bin/env python3
"""
Create AI Session Management Tables.

This script creates the enhanced session tables for the AI session management system
with state preservation, context windows, and cross-session continuity.
"""

import asyncio
import sys
import os
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy import text
from api.models import get_db
from api.models.session import Session, SessionHandoff, SessionCheckpoint, SessionMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_session_tables():
    """Create all session management tables."""
    try:
        logger.info("🧠 Creating AI Session Management tables...")
        
        db = next(get_db())
        
        # Enable UUID extension if not exists
        db.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"))
        
        # Check if old ai_sessions table exists and drop it
        result = db.execute(text("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'ai_sessions' AND table_schema = 'public' 
            AND column_name = 'session_id'
        """))
        old_table_exists = len(list(result)) > 0
        
        if old_table_exists:
            logger.info("Dropping old ai_sessions table to recreate with new schema...")
            db.execute(text("DROP TABLE IF EXISTS ai_sessions CASCADE;"))
            db.commit()
        
        # Create session management tables
        from api.models.base import Base, engine
        
        # Create tables in dependency order (Session first, then dependent tables)
        Base.metadata.create_all(bind=engine, tables=[Session.__table__])
        db.commit()
        
        # Create dependent tables
        Base.metadata.create_all(bind=engine, tables=[
            SessionHandoff.__table__,
            SessionCheckpoint.__table__,
            SessionMetrics.__table__
        ])
        
        db.commit()
        
        # Verify tables were created
        result = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name IN ('ai_sessions', 'session_handoffs', 'session_checkpoints', 'session_metrics')
            AND table_schema = 'public'
            ORDER BY table_name;
        """))
        
        tables = [row[0] for row in result]
        logger.info(f"✅ Created session tables: {tables}")
        
        # Create additional indexes for performance
        create_additional_indexes(db)
        
        # Create custom functions and triggers
        create_session_functions(db)
        
        logger.info("🎉 AI Session Management tables created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create session tables: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_additional_indexes(db):
    """Create additional performance indexes."""
    
    indexes = [
        # Session performance indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_user_project_active ON ai_sessions (user_id, project_id, state, last_active) WHERE state = 'active';",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_performance_metrics ON ai_sessions (success_rate DESC, avg_response_time ASC) WHERE state = 'active';",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_duration_analysis ON ai_sessions (total_duration DESC, interaction_count DESC);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_context_size ON ai_sessions (context_size, max_context_size) WHERE state = 'active';",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_recovery_ready ON ai_sessions (state, last_checkpoint DESC) WHERE recovery_data IS NOT NULL;",
        
        # Session chain and linking indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_parent_chain ON ai_sessions (parent_session_id, session_chain) WHERE parent_session_id IS NOT NULL;",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_related_gin ON ai_sessions USING gin(related_sessions);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_chain_gin ON ai_sessions USING gin(session_chain);",
        
        # Handoff performance indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_handoffs_pending_active ON session_handoffs (status, created_at) WHERE status = 'pending';",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_handoffs_source_target ON session_handoffs (source_session_id, target_session_id);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_handoffs_completion_time ON session_handoffs (completed_at DESC) WHERE status = 'completed';",
        
        # Checkpoint recovery indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_checkpoints_recovery_priority ON session_checkpoints (session_id, is_recovery_point DESC, recovery_priority DESC);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_checkpoints_auto_cleanup ON session_checkpoints (checkpoint_type, created_at) WHERE checkpoint_type = 'auto';",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_checkpoints_session_timeline ON session_checkpoints (session_id, created_at DESC, interaction_count);",
        
        # Metrics analysis indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_type_time ON session_metrics (metric_type, recorded_at DESC);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_performance_analysis ON session_metrics (session_id, metric_type, metric_value) WHERE metric_type IN ('response_time', 'memory_usage');",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_task_context ON session_metrics (task_context, metric_type, recorded_at);",
        
        # JSON field indexes for complex queries
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_context_window_gin ON ai_sessions USING gin(context_window);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_active_tasks_gin ON ai_sessions USING gin(active_tasks);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_preferences_gin ON ai_sessions USING gin(preferences);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_goals_gin ON ai_sessions USING gin(goals_achieved, goals_pending);",
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


def create_session_functions(db):
    """Create custom PostgreSQL functions for session operations."""
    
    functions = [
        # Function to calculate session health score
        """
        CREATE OR REPLACE FUNCTION calculate_session_health(
            success_rate FLOAT,
            avg_response_time FLOAT,
            error_count INTEGER,
            interaction_count INTEGER,
            total_duration INTEGER
        ) RETURNS FLOAT AS $$
        DECLARE
            performance_score FLOAT;
            reliability_score FLOAT;
            efficiency_score FLOAT;
            health_score FLOAT;
        BEGIN
            -- Performance score (0-1 based on response time)
            performance_score := CASE 
                WHEN avg_response_time <= 1.0 THEN 1.0
                WHEN avg_response_time <= 3.0 THEN 0.8
                WHEN avg_response_time <= 5.0 THEN 0.6
                WHEN avg_response_time <= 10.0 THEN 0.4
                ELSE 0.2
            END;
            
            -- Reliability score (based on success rate and error frequency)
            reliability_score := success_rate * (1.0 - LEAST(error_count::FLOAT / GREATEST(interaction_count, 1), 0.5));
            
            -- Efficiency score (interactions per time)
            efficiency_score := CASE
                WHEN total_duration > 0 THEN 
                    LEAST(interaction_count::FLOAT / (total_duration / 3600.0), 10.0) / 10.0
                ELSE 0.5
            END;
            
            -- Combined health score
            health_score := (performance_score * 0.3 + reliability_score * 0.5 + efficiency_score * 0.2);
            
            RETURN GREATEST(0.0, LEAST(1.0, health_score));
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Function to determine optimal context window size
        """
        CREATE OR REPLACE FUNCTION calculate_optimal_context_size(
            session_id_param UUID,
            current_size INTEGER,
            max_size INTEGER,
            avg_response_time FLOAT
        ) RETURNS INTEGER AS $$
        DECLARE
            recent_performance FLOAT;
            memory_pressure FLOAT;
            optimal_size INTEGER;
        BEGIN
            -- Analyze recent performance trends
            SELECT AVG(metric_value) INTO recent_performance
            FROM session_metrics 
            WHERE session_id = session_id_param 
                AND metric_type = 'response_time'
                AND recorded_at > NOW() - INTERVAL '1 hour';
            
            recent_performance := COALESCE(recent_performance, avg_response_time);
            
            -- Calculate memory pressure
            memory_pressure := current_size::FLOAT / max_size::FLOAT;
            
            -- Determine optimal size
            IF recent_performance > 5.0 AND memory_pressure > 0.8 THEN
                -- High response time and high memory usage - reduce context
                optimal_size := GREATEST(current_size - 10, max_size / 4);
            ELSIF recent_performance < 2.0 AND memory_pressure < 0.6 THEN
                -- Good performance and low memory usage - can increase context
                optimal_size := LEAST(current_size + 5, max_size);
            ELSE
                -- Maintain current size
                optimal_size := current_size;
            END IF;
            
            RETURN optimal_size;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Function to identify sessions ready for handoff
        """
        CREATE OR REPLACE FUNCTION find_handoff_candidates(
            user_id_param TEXT,
            min_duration INTEGER DEFAULT 3600,
            max_inactive INTEGER DEFAULT 1800
        ) RETURNS TABLE(session_id UUID, handoff_reason TEXT, priority INTEGER) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                s.id as session_id,
                CASE 
                    WHEN EXTRACT(EPOCH FROM (NOW() - s.last_active)) > max_inactive THEN 'timeout'
                    WHEN s.context_size >= s.max_context_size * 0.9 THEN 'context_limit'
                    WHEN s.total_duration > min_duration AND s.success_rate < 0.7 THEN 'performance'
                    WHEN s.error_count > 10 THEN 'error_recovery'
                    ELSE 'scheduled'
                END as handoff_reason,
                CASE 
                    WHEN s.error_count > 10 THEN 1
                    WHEN EXTRACT(EPOCH FROM (NOW() - s.last_active)) > max_inactive THEN 2
                    WHEN s.context_size >= s.max_context_size * 0.9 THEN 3
                    WHEN s.success_rate < 0.7 THEN 4
                    ELSE 5
                END as priority
            FROM ai_sessions s
            WHERE 
                s.user_id = user_id_param
                AND s.state = 'active'
                AND (
                    EXTRACT(EPOCH FROM (NOW() - s.last_active)) > max_inactive
                    OR s.context_size >= s.max_context_size * 0.8
                    OR (s.total_duration > min_duration AND s.success_rate < 0.7)
                    OR s.error_count > 5
                )
            ORDER BY priority ASC, s.last_active ASC;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Function to automatically clean up old sessions
        """
        CREATE OR REPLACE FUNCTION cleanup_old_sessions(
            retention_days INTEGER DEFAULT 30,
            checkpoint_retention_days INTEGER DEFAULT 7
        ) RETURNS INTEGER AS $$
        DECLARE
            sessions_cleaned INTEGER := 0;
            checkpoints_cleaned INTEGER := 0;
        BEGIN
            -- Archive old completed sessions
            UPDATE ai_sessions 
            SET state = 'archived'
            WHERE state IN ('completed', 'abandoned')
                AND ended_at < NOW() - INTERVAL '1 day' * retention_days;
            
            GET DIAGNOSTICS sessions_cleaned = ROW_COUNT;
            
            -- Clean up old auto checkpoints (keep manual and recovery checkpoints)
            DELETE FROM session_checkpoints
            WHERE checkpoint_type = 'auto'
                AND is_recovery_point = false
                AND created_at < NOW() - INTERVAL '1 day' * checkpoint_retention_days;
            
            GET DIAGNOSTICS checkpoints_cleaned = ROW_COUNT;
            
            RETURN sessions_cleaned + checkpoints_cleaned;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Trigger to automatically update session activity
        """
        CREATE OR REPLACE FUNCTION session_activity_trigger() RETURNS TRIGGER AS $$
        BEGIN
            -- Update last_active when certain fields change
            IF TG_OP = 'UPDATE' AND (
                OLD.interaction_count != NEW.interaction_count
                OR OLD.context_size != NEW.context_size
                OR OLD.active_tasks::text != NEW.active_tasks::text
            ) THEN
                NEW.last_active := NOW();
                
                -- Update total_duration for active sessions
                IF NEW.state = 'active' THEN
                    NEW.total_duration := EXTRACT(EPOCH FROM (NOW() - NEW.started_at));
                END IF;
            END IF;
            
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Create the trigger
        """
        DROP TRIGGER IF EXISTS session_activity_update ON ai_sessions;
        CREATE TRIGGER session_activity_update
            BEFORE UPDATE ON ai_sessions
            FOR EACH ROW
            EXECUTE FUNCTION session_activity_trigger();
        """,
        
        # Trigger to create automatic checkpoints
        """
        CREATE OR REPLACE FUNCTION auto_checkpoint_trigger() RETURNS TRIGGER AS $$
        DECLARE
            checkpoint_interval INTEGER := 50; -- Every 50 interactions
            last_checkpoint_count INTEGER;
        BEGIN
            -- Create automatic checkpoint every N interactions
            IF TG_OP = 'UPDATE' 
                AND OLD.interaction_count != NEW.interaction_count 
                AND NEW.interaction_count % checkpoint_interval = 0 THEN
                
                -- Check if we already have a checkpoint at this interaction count
                SELECT interaction_count INTO last_checkpoint_count
                FROM session_checkpoints
                WHERE session_id = NEW.id
                    AND checkpoint_type = 'auto'
                ORDER BY created_at DESC
                LIMIT 1;
                
                -- Create checkpoint if needed
                IF last_checkpoint_count IS NULL OR last_checkpoint_count < NEW.interaction_count THEN
                    INSERT INTO session_checkpoints (
                        session_id,
                        checkpoint_name,
                        description,
                        checkpoint_type,
                        session_state,
                        context_snapshot,
                        variables_snapshot,
                        interaction_count,
                        created_by,
                        is_recovery_point
                    ) VALUES (
                        NEW.id,
                        'Auto Checkpoint ' || NEW.interaction_count,
                        'Automatic checkpoint at ' || NEW.interaction_count || ' interactions',
                        'auto',
                        jsonb_build_object(
                            'state', NEW.state,
                            'context_size', NEW.context_size,
                            'success_rate', NEW.success_rate,
                            'error_count', NEW.error_count
                        ),
                        jsonb_build_object(
                            'context_window', NEW.context_window,
                            'context_summary', NEW.context_summary
                        ),
                        NEW.session_variables,
                        NEW.interaction_count,
                        'system',
                        (NEW.interaction_count % (checkpoint_interval * 4) = 0) -- Every 4th checkpoint is recovery point
                    );
                END IF;
            END IF;
            
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Create the auto checkpoint trigger
        """
        DROP TRIGGER IF EXISTS session_auto_checkpoint ON ai_sessions;
        CREATE TRIGGER session_auto_checkpoint
            AFTER UPDATE ON ai_sessions
            FOR EACH ROW
            EXECUTE FUNCTION auto_checkpoint_trigger();
        """
    ]
    
    logger.info("Creating session management functions and triggers...")
    
    for func_sql in functions:
        try:
            db.execute(text(func_sql))
            logger.debug(f"Created function/trigger: {func_sql[:50]}...")
        except Exception as e:
            logger.warning(f"Function creation warning: {str(e)[:100]}...")
    
    db.commit()
    logger.info("✅ Session functions and triggers created")


def create_sample_data(db):
    """Create sample session data for testing."""
    
    logger.info("Creating sample session data...")
    
    sample_sessions = [
        {
            "user_id": "test_user_1",
            "project_id": "knowledgehub_dev",
            "session_type": "interactive",
            "title": "KnowledgeHub Feature Development",
            "description": "Working on implementing session management features",
            "preferences": {"auto_save": True, "context_size": 100},
            "max_context_size": 100
        },
        {
            "user_id": "test_user_1",
            "project_id": "memory_system",
            "session_type": "debugging",
            "title": "Memory System Debugging Session",
            "description": "Debugging memory retrieval performance issues",
            "preferences": {"verbose_logging": True, "debug_mode": True},
            "max_context_size": 50
        },
        {
            "user_id": "test_user_2",
            "project_id": "api_optimization",
            "session_type": "workflow",
            "title": "API Performance Optimization",
            "description": "Systematic optimization of API endpoints",
            "preferences": {"benchmark_mode": True},
            "max_context_size": 75
        }
    ]
    
    session_ids = []
    
    for data in sample_sessions:
        try:
            # Insert sample session
            result = db.execute(text("""
                INSERT INTO ai_sessions (
                    user_id, project_id, session_type, title, description,
                    preferences, max_context_size, state
                ) VALUES (
                    :user_id, :project_id, :session_type, :title, :description,
                    :preferences, :max_context_size, 'active'
                ) RETURNING id
            """), {
                "user_id": data["user_id"],
                "project_id": data["project_id"],
                "session_type": data["session_type"],
                "title": data["title"],
                "description": data["description"],
                "preferences": data["preferences"],
                "max_context_size": data["max_context_size"]
            })
            
            session_id = result.fetchone()[0]
            session_ids.append(session_id)
            logger.info(f"Created sample session: {session_id}")
            
        except Exception as e:
            logger.warning(f"Sample session creation warning: {e}")
    
    # Create sample checkpoints for first session
    if session_ids:
        try:
            db.execute(text("""
                INSERT INTO session_checkpoints (
                    session_id, checkpoint_name, description, checkpoint_type,
                    session_state, context_snapshot, interaction_count,
                    created_by, is_recovery_point
                ) VALUES (
                    :session_id, 'Initial Setup', 'Session setup completed',
                    'manual', :session_state, :context_snapshot, 1,
                    'test_user_1', true
                )
            """), {
                "session_id": session_ids[0],
                "session_state": {"state": "active", "setup": "complete"},
                "context_snapshot": {"initialized": True, "features": ["memory", "session"]}
            })
            
            logger.info("Created sample checkpoint")
            
        except Exception as e:
            logger.warning(f"Sample checkpoint creation warning: {e}")
    
    db.commit()
    logger.info("✅ Sample session data created")


async def main():
    """Main function."""
    logger.info("🚀 Starting AI Session Management setup...")
    
    # Create session tables
    success = create_session_tables()
    
    if success:
        logger.info("🎉 AI Session Management setup complete!")
        
        # Optionally create sample data
        try:
            db = next(get_db())
            create_sample_data(db)
        except Exception as e:
            logger.warning(f"Sample data creation failed: {e}")
        
        return True
    else:
        logger.error("❌ AI Session Management setup failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)