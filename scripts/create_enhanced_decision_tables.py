#!/usr/bin/env python3
"""
Create Enhanced Decision Recording Tables.

This script creates the enhanced decision tables for comprehensive decision tracking,
analysis, and pattern recognition.
"""

import asyncio
import sys
import os
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy import text
from api.models import get_db
from api.models.enhanced_decision import (
    EnhancedDecision, EnhancedAlternative, EnhancedDecisionOutcome,
    EnhancedDecisionFeedback, EnhancedDecisionRevision, DecisionPattern
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_enhanced_decision_tables():
    """Create all enhanced decision tables."""
    try:
        logger.info("🧠 Creating Enhanced Decision Recording tables...")
        
        db = next(get_db())
        
        # Enable required extensions
        db.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"))
        db.execute(text("CREATE EXTENSION IF NOT EXISTS \"pg_trgm\";"))  # For text similarity
        
        # Create decision tables
        from api.models.base import Base, engine
        
        # Check if old decision_patterns table exists
        old_patterns_check = db.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'decision_patterns' 
                AND table_schema = 'public'
            );
        """)).scalar()
        
        if old_patterns_check:
            logger.info("Found existing decision_patterns table, dropping it...")
            db.execute(text("DROP TABLE IF EXISTS decision_patterns CASCADE;"))
            db.commit()
        
        # Create all tables defined in the models
        Base.metadata.create_all(bind=engine, tables=[
            EnhancedDecision.__table__,
            EnhancedAlternative.__table__,
            EnhancedDecisionOutcome.__table__,
            EnhancedDecisionFeedback.__table__,
            EnhancedDecisionRevision.__table__,
            DecisionPattern.__table__
        ])
        
        db.commit()
        
        # Verify tables were created
        result = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name IN (
                'enhanced_decisions', 'enhanced_decision_alternatives', 
                'enhanced_decision_outcomes', 'enhanced_decision_feedback',
                'enhanced_decision_revisions', 'decision_patterns'
            )
            AND table_schema = 'public'
            ORDER BY table_name;
        """))
        
        tables = [row[0] for row in result]
        logger.info(f"✅ Created enhanced decision tables: {tables}")
        
        # Create additional indexes for performance
        create_additional_indexes(db)
        
        # Create custom functions and triggers
        create_decision_functions(db)
        
        logger.info("🎉 Enhanced Decision Recording tables created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create enhanced decision tables: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_additional_indexes(db):
    """Create additional performance indexes."""
    
    indexes = [
        # Decision similarity and pattern matching
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_decisions_similarity ON enhanced_decisions USING gin(title gin_trgm_ops);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_decisions_reasoning ON enhanced_decisions USING gin(reasoning gin_trgm_ops);",
        
        # Context and analysis
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_decisions_context_gin ON enhanced_decisions USING gin(context);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_decisions_impact_gin ON enhanced_decisions USING gin(impact_analysis);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_decisions_tags_gin ON enhanced_decisions USING gin(tags);",
        
        # Performance optimization
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_decisions_high_confidence ON enhanced_decisions (confidence_score DESC) WHERE confidence_score > 0.8;",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_decisions_recent ON enhanced_decisions (decided_at DESC) WHERE status IN ('decided', 'implemented', 'validated');",
        
        # Decision tree navigation
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_decisions_tree ON enhanced_decisions (parent_decision_id, created_at);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_decisions_path_gin ON enhanced_decisions USING gin(decision_path);",
        
        # Outcome tracking
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_outcomes_success ON enhanced_decision_outcomes (success_rating DESC) WHERE success_rating IS NOT NULL;",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_outcomes_metrics_gin ON enhanced_decision_outcomes USING gin(performance_metrics);",
        
        # Feedback analysis
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feedback_ratings ON enhanced_decision_feedback (rating DESC, effectiveness_rating DESC) WHERE rating IS NOT NULL;",
        
        # Pattern analysis
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_patterns_effectiveness ON decision_patterns (success_rate DESC, avg_confidence DESC) WHERE occurrence_count > 5;",
        
        # Composite indexes for common queries
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_decisions_analytics ON enhanced_decisions (decision_type, status, impact_level, confidence_score DESC);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_decisions_user_history ON enhanced_decisions (user_id, project_id, decided_at DESC);",
    ]
    
    logger.info("Creating additional performance indexes...")
    
    for index_sql in indexes:
        try:
            db.execute(text(index_sql))
            logger.debug(f"Created index: {index_sql[:50]}...")
        except Exception as e:
            logger.warning(f"Index creation warning: {str(e)[:100]}...")
    
    db.commit()
    logger.info("✅ Additional indexes created")


def create_decision_functions(db):
    """Create custom PostgreSQL functions for decision operations."""
    
    functions = [
        # Function to calculate decision pattern hash
        """
        CREATE OR REPLACE FUNCTION calculate_decision_pattern_hash(
            decision_type TEXT,
            context JSONB,
            constraints JSONB
        ) RETURNS VARCHAR(64) AS $$
        DECLARE
            pattern_text TEXT;
        BEGIN
            -- Create normalized pattern text from key decision characteristics
            pattern_text := COALESCE(decision_type, '') || '::' ||
                          COALESCE(jsonb_strip_nulls(context)::text, '{}') || '::' ||
                          COALESCE(jsonb_strip_nulls(constraints)::text, '{}');
            
            RETURN encode(digest(pattern_text, 'sha256'), 'hex');
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Function to find similar decisions
        """
        CREATE OR REPLACE FUNCTION find_similar_decisions(
            search_title TEXT,
            search_reasoning TEXT,
            decision_type_filter TEXT DEFAULT NULL,
            similarity_threshold FLOAT DEFAULT 0.3,
            limit_count INTEGER DEFAULT 10
        ) RETURNS TABLE(
            decision_id UUID,
            similarity_score FLOAT,
            title VARCHAR,
            chosen_option TEXT,
            confidence_score FLOAT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                d.id as decision_id,
                GREATEST(
                    similarity(d.title, search_title),
                    similarity(d.reasoning, search_reasoning) * 0.8
                ) as similarity_score,
                d.title,
                d.chosen_option,
                d.confidence_score
            FROM enhanced_decisions d
            WHERE 
                (decision_type_filter IS NULL OR d.decision_type = decision_type_filter)
                AND (
                    similarity(d.title, search_title) > similarity_threshold
                    OR similarity(d.reasoning, search_reasoning) > similarity_threshold
                )
            ORDER BY similarity_score DESC
            LIMIT limit_count;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Function to build decision tree
        """
        CREATE OR REPLACE FUNCTION get_decision_tree(
            root_decision_id UUID,
            max_depth INTEGER DEFAULT 10
        ) RETURNS TABLE(
            decision_id UUID,
            parent_id UUID,
            level INTEGER,
            title VARCHAR,
            decision_type VARCHAR,
            status VARCHAR,
            confidence_score FLOAT,
            path UUID[]
        ) AS $$
        WITH RECURSIVE decision_tree AS (
            -- Base case: root decision
            SELECT 
                d.id as decision_id,
                d.parent_decision_id as parent_id,
                0 as level,
                d.title,
                d.decision_type,
                d.status,
                d.confidence_score,
                ARRAY[d.id] as path
            FROM enhanced_decisions d
            WHERE d.id = root_decision_id
            
            UNION ALL
            
            -- Recursive case: child decisions
            SELECT 
                d.id as decision_id,
                d.parent_decision_id as parent_id,
                dt.level + 1 as level,
                d.title,
                d.decision_type,
                d.status,
                d.confidence_score,
                dt.path || d.id as path
            FROM enhanced_decisions d
            JOIN decision_tree dt ON d.parent_decision_id = dt.decision_id
            WHERE dt.level < max_depth
        )
        SELECT * FROM decision_tree ORDER BY level, created_at;
        $$ LANGUAGE sql;
        """,
        
        # Function to calculate decision success probability
        """
        CREATE OR REPLACE FUNCTION predict_decision_success(
            decision_type_param VARCHAR,
            confidence_param FLOAT,
            impact_level_param VARCHAR,
            context_param JSONB
        ) RETURNS FLOAT AS $$
        DECLARE
            base_probability FLOAT;
            pattern_boost FLOAT;
            context_factor FLOAT;
        BEGIN
            -- Base probability from confidence
            base_probability := confidence_param * 0.6;
            
            -- Pattern-based boost
            SELECT 
                COALESCE(AVG(p.success_rate) * 0.3, 0.0)
            INTO pattern_boost
            FROM decision_patterns p
            WHERE 
                p.pattern_type = decision_type_param
                AND p.occurrence_count > 5;
            
            -- Context complexity factor
            context_factor := CASE
                WHEN jsonb_array_length(context_param -> 'constraints') > 5 THEN -0.1
                WHEN impact_level_param = 'critical' THEN -0.05
                WHEN impact_level_param = 'minimal' THEN 0.05
                ELSE 0.0
            END;
            
            -- Calculate final probability
            RETURN GREATEST(0.1, LEAST(0.95, 
                base_probability + pattern_boost + context_factor
            ));
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Function to analyze decision impact
        """
        CREATE OR REPLACE FUNCTION analyze_decision_impact(
            decision_id_param UUID
        ) RETURNS TABLE(
            direct_impact_count INTEGER,
            indirect_impact_count INTEGER,
            total_affected_components INTEGER,
            downstream_decisions INTEGER,
            success_impact_score FLOAT
        ) AS $$
        DECLARE
            decision_record RECORD;
        BEGIN
            -- Get decision details
            SELECT * INTO decision_record 
            FROM enhanced_decisions 
            WHERE id = decision_id_param;
            
            -- Direct impact from affected components
            direct_impact_count := COALESCE(
                array_length(decision_record.affected_components, 1), 0
            );
            
            -- Count downstream decisions
            SELECT COUNT(*) INTO downstream_decisions
            FROM enhanced_decisions
            WHERE decision_id_param = ANY(decision_path);
            
            -- Calculate indirect impact (simplified)
            indirect_impact_count := downstream_decisions * 2;
            
            -- Total affected components
            total_affected_components := direct_impact_count + indirect_impact_count;
            
            -- Success impact score
            success_impact_score := CASE
                WHEN decision_record.impact_level = 'critical' THEN 0.9
                WHEN decision_record.impact_level = 'high' THEN 0.7
                WHEN decision_record.impact_level = 'medium' THEN 0.5
                WHEN decision_record.impact_level = 'low' THEN 0.3
                ELSE 0.1
            END * decision_record.confidence_score;
            
            RETURN QUERY SELECT 
                direct_impact_count,
                indirect_impact_count,
                total_affected_components,
                downstream_decisions,
                success_impact_score;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Trigger to maintain decision path
        """
        CREATE OR REPLACE FUNCTION update_decision_path() RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.parent_decision_id IS NOT NULL THEN
                -- Get parent's path and append current decision
                SELECT 
                    COALESCE(decision_path, ARRAY[]::UUID[]) || NEW.id
                INTO NEW.decision_path
                FROM enhanced_decisions
                WHERE id = NEW.parent_decision_id;
            ELSE
                -- Root decision
                NEW.decision_path := ARRAY[NEW.id];
            END IF;
            
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Create the trigger
        """
        DROP TRIGGER IF EXISTS decision_path_update ON enhanced_decisions;
        CREATE TRIGGER decision_path_update
            BEFORE INSERT OR UPDATE OF parent_decision_id ON enhanced_decisions
            FOR EACH ROW
            EXECUTE FUNCTION update_decision_path();
        """,
        
        # Function to mine decision patterns
        """
        CREATE OR REPLACE FUNCTION mine_decision_patterns(
            min_occurrences INTEGER DEFAULT 3,
            min_success_rate FLOAT DEFAULT 0.6
        ) RETURNS TABLE(
            pattern_type VARCHAR,
            pattern_characteristics JSONB,
            occurrence_count BIGINT,
            avg_success_rate FLOAT,
            common_context JSONB
        ) AS $$
        BEGIN
            RETURN QUERY
            WITH decision_groups AS (
                SELECT 
                    d.decision_type,
                    d.pattern_hash,
                    COUNT(*) as count,
                    AVG(CASE 
                        WHEN o.success_rating IS NOT NULL THEN o.success_rating
                        ELSE 0.5 
                    END) as avg_success,
                    jsonb_agg(DISTINCT d.context) as contexts
                FROM enhanced_decisions d
                LEFT JOIN enhanced_decision_outcomes o ON d.id = o.decision_id
                WHERE d.pattern_hash IS NOT NULL
                GROUP BY d.decision_type, d.pattern_hash
                HAVING COUNT(*) >= min_occurrences
            )
            SELECT 
                decision_type as pattern_type,
                jsonb_build_object(
                    'pattern_hash', pattern_hash,
                    'decision_count', count
                ) as pattern_characteristics,
                count as occurrence_count,
                avg_success as avg_success_rate,
                contexts -> 0 as common_context
            FROM decision_groups
            WHERE avg_success >= min_success_rate
            ORDER BY count DESC, avg_success DESC;
        END;
        $$ LANGUAGE plpgsql;
        """
    ]
    
    logger.info("Creating decision functions and triggers...")
    
    for func_sql in functions:
        try:
            db.execute(text(func_sql))
            logger.debug(f"Created function/trigger: {func_sql[:50]}...")
        except Exception as e:
            logger.warning(f"Function creation warning: {str(e)[:100]}...")
    
    db.commit()
    logger.info("✅ Decision functions and triggers created")


def create_sample_data(db):
    """Create sample decision data for testing."""
    
    logger.info("Creating sample decision data...")
    
    sample_decisions = [
        {
            "decision_type": "architectural",
            "category": "database",
            "title": "Choose primary database for KnowledgeHub",
            "chosen_option": "PostgreSQL with TimescaleDB extension",
            "reasoning": "PostgreSQL provides robust ACID compliance, excellent JSON support with JSONB, and TimescaleDB adds time-series capabilities needed for analytics",
            "confidence_score": 0.95,
            "impact_level": "critical",
            "context": {"project_phase": "initial", "team_size": 5},
            "constraints": {"budget": "moderate", "timeline": "6 months"},
            "alternatives": [
                {
                    "option": "MongoDB",
                    "pros": ["Flexible schema", "Good for documents"],
                    "cons": ["Less suitable for relational data", "No time-series support"],
                    "rejection_reason": "Need strong consistency and time-series capabilities"
                },
                {
                    "option": "MySQL",
                    "pros": ["Widely used", "Good performance"],
                    "cons": ["Limited JSON support", "No native time-series"],
                    "rejection_reason": "Inferior JSON handling compared to PostgreSQL"
                }
            ]
        },
        {
            "decision_type": "design",
            "category": "api",
            "title": "API architecture pattern",
            "chosen_option": "RESTful API with FastAPI",
            "reasoning": "FastAPI provides automatic OpenAPI documentation, async support, and type safety with Pydantic",
            "confidence_score": 0.88,
            "impact_level": "high",
            "context": {"api_consumers": ["web", "mobile", "cli"]},
            "constraints": {"performance": "high throughput required"},
            "alternatives": [
                {
                    "option": "GraphQL",
                    "pros": ["Flexible queries", "Single endpoint"],
                    "cons": ["Complex caching", "Learning curve"],
                    "rejection_reason": "Team more familiar with REST, caching simpler"
                }
            ]
        },
        {
            "decision_type": "implementation",
            "category": "caching",
            "title": "Caching strategy for vector embeddings",
            "chosen_option": "Redis with TTL-based eviction",
            "reasoning": "Redis provides fast in-memory access with automatic expiration, perfect for embedding cache",
            "confidence_score": 0.82,
            "impact_level": "medium",
            "context": {"embedding_size": 768, "daily_requests": 100000},
            "constraints": {"memory": "32GB available"},
            "alternatives": [
                {
                    "option": "In-process memory cache",
                    "pros": ["No network overhead", "Simplest"],
                    "cons": ["Not shared across instances", "Memory pressure"],
                    "rejection_reason": "Need shared cache for multiple API instances"
                }
            ]
        }
    ]
    
    for decision_data in sample_decisions:
        try:
            # Extract alternatives
            alternatives = decision_data.pop("alternatives", [])
            
            # Create decision
            result = db.execute(text("""
                INSERT INTO enhanced_decisions (
                    decision_type, category, title, chosen_option, reasoning,
                    confidence_score, impact_level, context, constraints,
                    status, user_id, project_id, pattern_hash
                ) VALUES (
                    :decision_type, :category, :title, :chosen_option, :reasoning,
                    :confidence_score, :impact_level, :context, :constraints,
                    'validated', 'system', 'knowledgehub',
                    calculate_decision_pattern_hash(:decision_type, :context, :constraints)
                ) RETURNING id
            """), {
                **decision_data,
                "context": db.execute(text("SELECT :val::jsonb"), {"val": str(decision_data["context"])}).scalar(),
                "constraints": db.execute(text("SELECT :val::jsonb"), {"val": str(decision_data["constraints"])}).scalar()
            })
            
            decision_id = result.fetchone()[0]
            logger.info(f"Created sample decision: {decision_id}")
            
            # Create alternatives
            for alt in alternatives:
                db.execute(text("""
                    INSERT INTO enhanced_decision_alternatives (
                        decision_id, option, pros, cons, rejection_reason
                    ) VALUES (
                        :decision_id, :option, :pros, :cons, :rejection_reason
                    )
                """), {
                    "decision_id": decision_id,
                    "option": alt["option"],
                    "pros": alt.get("pros", []),
                    "cons": alt.get("cons", []),
                    "rejection_reason": alt.get("rejection_reason")
                })
            
            # Create a successful outcome for some decisions
            if decision_data["confidence_score"] > 0.85:
                db.execute(text("""
                    INSERT INTO enhanced_decision_outcomes (
                        decision_id, status, success_rating, description,
                        performance_metrics
                    ) VALUES (
                        :decision_id, 'successful', :success_rating,
                        'Decision implemented successfully with positive results',
                        :metrics
                    )
                """), {
                    "decision_id": decision_id,
                    "success_rating": decision_data["confidence_score"],
                    "metrics": db.execute(text("SELECT :val::jsonb"), {"val": '{"response_time": "improved", "stability": "high"}'}).scalar()
                })
            
        except Exception as e:
            logger.warning(f"Sample data creation warning: {e}")
    
    db.commit()
    logger.info("✅ Sample decision data created")


async def main():
    """Main function."""
    logger.info("🚀 Starting Enhanced Decision Recording setup...")
    
    # Create enhanced decision tables
    success = create_enhanced_decision_tables()
    
    if success:
        logger.info("🎉 Enhanced Decision Recording setup complete!")
        
        # Optionally create sample data
        try:
            db = next(get_db())
            create_sample_data(db)
        except Exception as e:
            logger.warning(f"Sample data creation failed: {e}")
        
        return True
    else:
        logger.error("❌ Enhanced Decision Recording setup failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)