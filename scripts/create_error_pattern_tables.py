#!/usr/bin/env python3
"""
Create AI Error Pattern Recognition Tables.

This script creates the enhanced error pattern tables for the AI error learning system
with pattern recognition, solution tracking, and predictive capabilities.
"""

import asyncio
import sys
import os
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy import text
from api.models import get_db
from api.models.error_pattern import (
    EnhancedErrorPattern, ErrorOccurrence, ErrorSolution, 
    ErrorFeedback, ErrorPrediction
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_error_pattern_tables():
    """Create all error pattern recognition tables."""
    try:
        logger.info("🧠 Creating AI Error Pattern Recognition tables...")
        
        db = next(get_db())
        
        # Enable required extensions
        db.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"))
        db.execute(text("CREATE EXTENSION IF NOT EXISTS \"pg_trgm\";"))  # For fuzzy text matching
        
        # Create error pattern tables
        from api.models.base import Base, engine
        
        # Create all tables defined in the models
        Base.metadata.create_all(bind=engine, tables=[
            EnhancedErrorPattern.__table__,
            ErrorOccurrence.__table__,
            ErrorSolution.__table__,
            ErrorFeedback.__table__,
            ErrorPrediction.__table__
        ])
        
        db.commit()
        
        # Verify tables were created
        result = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name IN (
                'enhanced_error_patterns', 'error_occurrences', 'error_solutions', 
                'error_feedback', 'error_predictions'
            )
            AND table_schema = 'public'
            ORDER BY table_name;
        """))
        
        tables = [row[0] for row in result]
        logger.info(f"✅ Created error pattern tables: {tables}")
        
        # Create additional indexes for performance
        create_additional_indexes(db)
        
        # Create custom functions and triggers
        create_error_functions(db)
        
        logger.info("🎉 AI Error Pattern Recognition tables created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create error pattern tables: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_additional_indexes(db):
    """Create additional performance indexes."""
    
    indexes = [
        # Text similarity indexes for fuzzy matching
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_patterns_message_trgm ON enhanced_error_patterns USING gin(error_message gin_trgm_ops);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_patterns_type_trgm ON enhanced_error_patterns USING gin(error_type gin_trgm_ops);",
        
        # Pattern matching indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_patterns_indicators_gin ON enhanced_error_patterns USING gin(key_indicators);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_patterns_context_gin ON enhanced_error_patterns USING gin(context);",
        
        # Performance optimization indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_patterns_confidence ON enhanced_error_patterns (confidence_score DESC) WHERE confidence_score > 0.7;",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_patterns_recent ON enhanced_error_patterns (last_seen DESC) WHERE occurrences > 5;",
        
        # Occurrence tracking indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_occurrences_unresolved ON error_occurrences (pattern_id, timestamp) WHERE resolved = false;",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_occurrences_resolution_time ON error_occurrences (resolution_time) WHERE resolved = true;",
        
        # Solution effectiveness indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_solutions_effective ON error_solutions (pattern_id, effectiveness_score DESC) WHERE status = 'verified';",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_solutions_recent ON error_solutions (last_used DESC) WHERE effectiveness_score > 0.5;",
        
        # Feedback analysis indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_feedback_helpful ON error_feedback (pattern_id) WHERE helpful = true;",
        
        # Prediction tracking indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_predictions_accurate ON error_predictions (prediction_confidence DESC) WHERE prediction_occurred = true;",
        
        # Composite indexes for common queries
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_patterns_search_composite ON enhanced_error_patterns (error_type, error_category, severity, occurrences DESC);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_occurrences_user_pattern ON error_occurrences (user_id, pattern_id, timestamp DESC);",
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


def create_error_functions(db):
    """Create custom PostgreSQL functions for error pattern operations."""
    
    functions = [
        # Function to calculate pattern hash
        """
        CREATE OR REPLACE FUNCTION calculate_error_pattern_hash(
            error_type TEXT,
            error_message TEXT,
            stack_trace TEXT
        ) RETURNS VARCHAR(64) AS $$
        DECLARE
            normalized_message TEXT;
            pattern_text TEXT;
        BEGIN
            -- Normalize error message by removing numbers and specific identifiers
            normalized_message := regexp_replace(
                regexp_replace(
                    lower(error_message),
                    '[0-9]+', 'N', 'g'
                ),
                '[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', 
                'UUID', 'gi'
            );
            
            -- Extract key stack trace elements
            pattern_text := COALESCE(error_type, '') || '::' || 
                          normalized_message || '::' ||
                          COALESCE(regexp_replace(stack_trace, ':[0-9]+', '', 'g'), '');
            
            RETURN encode(digest(pattern_text, 'sha256'), 'hex');
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Function to find similar error patterns
        """
        CREATE OR REPLACE FUNCTION find_similar_error_patterns(
            search_message TEXT,
            search_type TEXT DEFAULT NULL,
            similarity_threshold FLOAT DEFAULT 0.3,
            limit_count INTEGER DEFAULT 10
        ) RETURNS TABLE(
            pattern_id UUID,
            similarity_score FLOAT,
            error_type VARCHAR,
            error_message TEXT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                ep.id as pattern_id,
                GREATEST(
                    similarity(ep.error_message, search_message),
                    CASE 
                        WHEN search_type IS NOT NULL 
                        THEN similarity(ep.error_type, search_type)
                        ELSE 0
                    END
                ) as similarity_score,
                ep.error_type,
                ep.error_message
            FROM enhanced_error_patterns ep
            WHERE 
                similarity(ep.error_message, search_message) > similarity_threshold
                OR (search_type IS NOT NULL AND similarity(ep.error_type, search_type) > similarity_threshold)
            ORDER BY similarity_score DESC
            LIMIT limit_count;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Function to update pattern metrics
        """
        CREATE OR REPLACE FUNCTION update_error_pattern_metrics(
            pattern_id_param UUID,
            success_param BOOLEAN,
            resolution_time_param FLOAT DEFAULT NULL
        ) RETURNS VOID AS $$
        DECLARE
            current_pattern RECORD;
            alpha FLOAT := 0.1;  -- Learning rate
        BEGIN
            SELECT * INTO current_pattern FROM enhanced_error_patterns WHERE id = pattern_id_param;
            
            IF FOUND THEN
                -- Update occurrence count and last seen
                UPDATE enhanced_error_patterns 
                SET 
                    occurrences = occurrences + 1,
                    last_seen = NOW(),
                    -- Update success rate with exponential moving average
                    success_rate = alpha * (CASE WHEN success_param THEN 1.0 ELSE 0.0 END) + 
                                 (1 - alpha) * success_rate,
                    -- Update average resolution time if successful
                    avg_resolution_time = CASE 
                        WHEN success_param AND resolution_time_param IS NOT NULL 
                        THEN alpha * resolution_time_param + (1 - alpha) * avg_resolution_time
                        ELSE avg_resolution_time
                    END,
                    -- Update confidence based on consistency
                    confidence_score = CASE
                        WHEN occurrences > 10 AND success_rate > 0.8 THEN LEAST(0.95, confidence_score + 0.05)
                        WHEN occurrences > 10 AND success_rate < 0.3 THEN GREATEST(0.1, confidence_score - 0.05)
                        ELSE confidence_score
                    END
                WHERE id = pattern_id_param;
            END IF;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Function to get recommended solutions
        """
        CREATE OR REPLACE FUNCTION get_recommended_solutions(
            pattern_id_param UUID,
            limit_count INTEGER DEFAULT 5
        ) RETURNS TABLE(
            solution_id UUID,
            solution_text TEXT,
            effectiveness_score FLOAT,
            avg_resolution_time FLOAT,
            success_count INTEGER
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                s.id as solution_id,
                s.solution_text,
                s.effectiveness_score,
                s.avg_resolution_time,
                s.success_count
            FROM error_solutions s
            WHERE 
                s.pattern_id = pattern_id_param
                AND s.status IN ('verified', 'suggested')
            ORDER BY 
                s.effectiveness_score DESC,
                s.success_count DESC
            LIMIT limit_count;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Trigger to update search vector
        """
        CREATE OR REPLACE FUNCTION update_error_pattern_search_vector() RETURNS TRIGGER AS $$
        BEGIN
            NEW.search_vector := to_tsvector('english',
                COALESCE(NEW.error_type, '') || ' ' ||
                COALESCE(NEW.error_message, '') || ' ' ||
                COALESCE(NEW.primary_solution, '') || ' ' ||
                COALESCE(array_to_string(NEW.key_indicators, ' '), '')
            );
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Create the trigger
        """
        DROP TRIGGER IF EXISTS error_pattern_search_vector_update ON enhanced_error_patterns;
        CREATE TRIGGER error_pattern_search_vector_update
            BEFORE INSERT OR UPDATE ON enhanced_error_patterns
            FOR EACH ROW
            EXECUTE FUNCTION update_error_pattern_search_vector();
        """,
        
        # Function to predict error likelihood
        """
        CREATE OR REPLACE FUNCTION predict_error_likelihood(
            user_id_param VARCHAR,
            session_id_param UUID,
            context_factors JSON
        ) RETURNS TABLE(
            pattern_id UUID,
            likelihood_score FLOAT,
            risk_factors JSON
        ) AS $$
        BEGIN
            -- Simplified prediction based on recent error history and context
            RETURN QUERY
            WITH recent_errors AS (
                SELECT 
                    eo.pattern_id,
                    COUNT(*) as error_count,
                    MAX(eo.timestamp) as last_occurrence
                FROM error_occurrences eo
                WHERE 
                    eo.user_id = user_id_param
                    AND eo.timestamp > NOW() - INTERVAL '7 days'
                GROUP BY eo.pattern_id
            ),
            pattern_scores AS (
                SELECT 
                    re.pattern_id,
                    ep.confidence_score,
                    ep.occurrences,
                    re.error_count,
                    EXTRACT(EPOCH FROM (NOW() - re.last_occurrence)) / 3600 as hours_since_last,
                    -- Calculate likelihood based on frequency and recency
                    (re.error_count::FLOAT / 7) * -- Errors per day
                    (1.0 / (1.0 + hours_since_last / 24)) * -- Recency factor
                    ep.confidence_score as likelihood_score
                FROM recent_errors re
                JOIN enhanced_error_patterns ep ON ep.id = re.pattern_id
                WHERE ep.occurrences > 3
            )
            SELECT 
                ps.pattern_id,
                ps.likelihood_score,
                jsonb_build_object(
                    'error_count_7d', ps.error_count,
                    'hours_since_last', ps.hours_since_last,
                    'pattern_confidence', ps.confidence_score
                ) as risk_factors
            FROM pattern_scores ps
            WHERE ps.likelihood_score > 0.3
            ORDER BY ps.likelihood_score DESC
            LIMIT 10;
        END;
        $$ LANGUAGE plpgsql;
        """
    ]
    
    logger.info("Creating error pattern functions and triggers...")
    
    for func_sql in functions:
        try:
            db.execute(text(func_sql))
            logger.debug(f"Created function/trigger: {func_sql[:50]}...")
        except Exception as e:
            logger.warning(f"Function creation warning: {str(e)[:100]}...")
    
    db.commit()
    logger.info("✅ Error pattern functions and triggers created")


def create_sample_data(db):
    """Create sample error pattern data for testing."""
    
    logger.info("Creating sample error pattern data...")
    
    sample_patterns = [
        {
            "pattern_hash": "sample_syntax_001",
            "error_type": "SyntaxError",
            "error_category": "syntax",
            "error_message": "Unexpected token '}' at line N",
            "severity": "medium",
            "primary_solution": "Check for missing opening braces or extra closing braces",
            "solution_steps": ["Locate the line with the error", "Check matching braces", "Ensure proper syntax"],
            "key_indicators": ["unexpected token", "syntax error", "brace"],
            "occurrences": 50,
            "success_rate": 0.95
        },
        {
            "pattern_hash": "sample_null_001",
            "error_type": "TypeError",
            "error_category": "runtime",
            "error_message": "Cannot read property 'N' of null",
            "severity": "high",
            "primary_solution": "Add null checks before accessing object properties",
            "solution_steps": ["Identify the null object", "Add conditional check", "Handle null case appropriately"],
            "key_indicators": ["cannot read property", "null", "undefined"],
            "occurrences": 120,
            "success_rate": 0.88
        },
        {
            "pattern_hash": "sample_connection_001",
            "error_type": "ConnectionError",
            "error_category": "network",
            "error_message": "Failed to establish connection to database",
            "severity": "critical",
            "primary_solution": "Check database connection settings and network connectivity",
            "solution_steps": ["Verify database is running", "Check connection string", "Test network connectivity"],
            "key_indicators": ["connection", "database", "failed"],
            "occurrences": 30,
            "success_rate": 0.75
        }
    ]
    
    for pattern_data in sample_patterns:
        try:
            # Check if pattern already exists
            existing = db.execute(text("""
                SELECT id FROM enhanced_error_patterns WHERE pattern_hash = :hash
            """), {"hash": pattern_data["pattern_hash"]}).fetchone()
            
            if not existing:
                # Insert sample pattern
                result = db.execute(text("""
                    INSERT INTO enhanced_error_patterns (
                        pattern_hash, error_type, error_category, error_message,
                        severity, primary_solution, solution_steps, key_indicators,
                        occurrences, success_rate, confidence_score,
                        context, created_by
                    ) VALUES (
                        :pattern_hash, :error_type, :error_category, :error_message,
                        :severity, :primary_solution, :solution_steps, :key_indicators,
                        :occurrences, :success_rate, :confidence_score,
                        :context, :created_by
                    ) RETURNING id
                """), {
                    **pattern_data,
                    "confidence_score": pattern_data["success_rate"] * 0.9,
                    "context": {"sample": True},
                    "created_by": "system"
                })
                
                pattern_id = result.fetchone()[0]
                logger.info(f"Created sample error pattern: {pattern_id}")
                
                # Create a sample solution
                db.execute(text("""
                    INSERT INTO error_solutions (
                        pattern_id, solution_text, solution_steps,
                        success_count, effectiveness_score, status, created_by
                    ) VALUES (
                        :pattern_id, :solution_text, :solution_steps,
                        :success_count, :effectiveness_score, 'verified', 'system'
                    )
                """), {
                    "pattern_id": pattern_id,
                    "solution_text": pattern_data["primary_solution"],
                    "solution_steps": pattern_data["solution_steps"],
                    "success_count": int(pattern_data["occurrences"] * pattern_data["success_rate"]),
                    "effectiveness_score": pattern_data["success_rate"]
                })
            
        except Exception as e:
            logger.warning(f"Sample data creation warning: {e}")
    
    db.commit()
    logger.info("✅ Sample error pattern data created")


async def main():
    """Main function."""
    logger.info("🚀 Starting AI Error Pattern Recognition setup...")
    
    # Create error pattern tables
    success = create_error_pattern_tables()
    
    if success:
        logger.info("🎉 AI Error Pattern Recognition setup complete!")
        
        # Optionally create sample data
        try:
            db = next(get_db())
            create_sample_data(db)
        except Exception as e:
            logger.warning(f"Sample data creation failed: {e}")
        
        return True
    else:
        logger.error("❌ AI Error Pattern Recognition setup failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)