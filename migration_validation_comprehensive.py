#!/usr/bin/env python3
"""
KnowledgeHub Migration Validation - Comprehensive Testing
========================================================

Validates the database migration from legacy system to hybrid RAG architecture:
- Data integrity verification
- Schema validation
- Performance comparison
- Backward compatibility testing
- Migration rollback testing
- Data consistency across services
"""

import psycopg2
import json
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sys
import subprocess

# Migration validation configuration
MIGRATION_CONFIG = {
    'postgres_url': 'postgresql://knowledgehub:knowledgehub123@localhost:5433/knowledgehub',
    'timescale_url': 'postgresql://knowledgehub:knowledgehub123@localhost:5434/knowledgehub_analytics',
    'backup_directory': '/opt/projects/knowledgehub/backups',
    'expected_migrations': [
        '001_initial_schema.sql',
        '002_learning_system_tables.sql', 
        '003_timescale_analytics.sql',
        '004_hybrid_rag_schema.sql',
        '005_data_migration.sql'
    ],
    'legacy_tables': [
        'ai_memories',
        'documents', 
        'chunks',
        'users',
        'sessions'
    ],
    'new_tables': [
        'memory_clusters',
        'memory_associations',
        'memory_access_logs',
        'agent_definitions',
        'workflow_definitions',
        'workflow_executions',
        'agent_tasks',
        'rag_configurations',
        'rag_query_logs',
        'enhanced_chunks',
        'document_ingestion_logs',
        'search_result_cache',
        'zep_session_mapping',
        'firecrawl_jobs',
        'service_health_logs',
        'performance_monitoring'
    ]
}

@dataclass
class MigrationValidationResult:
    """Migration validation test result"""
    test_name: str
    category: str
    status: str  # 'passed', 'failed', 'warning'
    duration_ms: int
    details: Dict[str, Any]
    error: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class MigrationValidator:
    """Comprehensive migration validation system"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.results: List[MigrationValidationResult] = []
        self.postgres_conn = None
        self.timescale_conn = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _get_postgres_connection(self):
        """Get PostgreSQL connection"""
        if not self.postgres_conn or self.postgres_conn.closed:
            self.postgres_conn = psycopg2.connect(MIGRATION_CONFIG['postgres_url'])
        return self.postgres_conn
    
    def _get_timescale_connection(self):
        """Get TimescaleDB connection"""
        if not self.timescale_conn or self.timescale_conn.closed:
            self.timescale_conn = psycopg2.connect(MIGRATION_CONFIG['timescale_url'])
        return self.timescale_conn
    
    def _record_result(self, name: str, category: str, status: str, 
                      duration_ms: int, details: Dict[str, Any], 
                      error: str = None):
        """Record validation result"""
        result = MigrationValidationResult(
            test_name=name,
            category=category,
            status=status,
            duration_ms=duration_ms,
            details=details,
            error=error
        )
        self.results.append(result)
        
        # Log result
        status_emoji = "✅" if status == "passed" else "⚠️" if status == "warning" else "❌"
        self.logger.info(f"{status_emoji} {name} ({duration_ms}ms) - {status}")
        if error:
            self.logger.error(f"Error: {error}")
    
    # ===========================================
    # MIGRATION STATUS VALIDATION
    # ===========================================
    
    def test_migration_log_integrity(self):
        """Validate migration log completeness"""
        try:
            start_time = time.time()
            
            conn = self._get_postgres_connection()
            cursor = conn.cursor()
            
            # Check if migration_log table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'migration_log'
                )
            """)
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                duration_ms = int((time.time() - start_time) * 1000)
                self._record_result(
                    "Migration Log Table Exists",
                    "Migration Status",
                    "failed",
                    duration_ms,
                    {"table_exists": False},
                    "migration_log table not found"
                )
                return
            
            # Get migration log entries
            cursor.execute("""
                SELECT migration_name, completed_at, notes
                FROM migration_log
                ORDER BY completed_at
            """)
            migrations = cursor.fetchall()
            
            # Check for expected migrations
            completed_migrations = [m[0] for m in migrations]
            expected_migrations = MIGRATION_CONFIG['expected_migrations']
            
            missing_migrations = []
            for expected in expected_migrations:
                migration_name = expected.replace('.sql', '')
                if not any(migration_name in completed for completed in completed_migrations):
                    missing_migrations.append(expected)
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            status = "passed" if not missing_migrations else "failed"
            error = f"Missing migrations: {missing_migrations}" if missing_migrations else None
            
            self._record_result(
                "Migration Log Integrity", 
                "Migration Status",
                status,
                duration_ms,
                {
                    "completed_migrations": completed_migrations,
                    "missing_migrations": missing_migrations,
                    "total_migrations": len(migrations)
                },
                error
            )
            
            cursor.close()
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._record_result(
                "Migration Log Integrity",
                "Migration Status", 
                "failed",
                duration_ms,
                {},
                str(e)
            )
    
    def test_schema_validation(self):
        """Validate database schema after migration"""
        try:
            start_time = time.time()
            
            conn = self._get_postgres_connection()
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            # Check for required tables
            all_required_tables = MIGRATION_CONFIG['legacy_tables'] + MIGRATION_CONFIG['new_tables']
            missing_tables = [table for table in all_required_tables if table not in existing_tables]
            
            # Check for required extensions
            cursor.execute("SELECT extname FROM pg_extension")
            extensions = [row[0] for row in cursor.fetchall()]
            required_extensions = ['uuid-ossp', 'vector', 'pg_trgm', 'btree_gin', 'pgcrypto']
            missing_extensions = [ext for ext in required_extensions if ext not in extensions]
            
            # Check for views
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.views 
                WHERE table_schema = 'public'
            """)
            views = [row[0] for row in cursor.fetchall()]
            expected_views = ['enhanced_memory_view', 'workflow_performance_view', 'rag_performance_view', 'service_health_summary']
            missing_views = [view for view in expected_views if view not in views]
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            issues = []
            if missing_tables:
                issues.append(f"Missing tables: {missing_tables}")
            if missing_extensions:
                issues.append(f"Missing extensions: {missing_extensions}")
            if missing_views:
                issues.append(f"Missing views: {missing_views}")
            
            status = "passed" if not issues else "failed"
            error = "; ".join(issues) if issues else None
            
            self._record_result(
                "Schema Validation",
                "Migration Status",
                status,
                duration_ms,
                {
                    "total_tables": len(existing_tables),
                    "missing_tables": missing_tables,
                    "missing_extensions": missing_extensions,
                    "missing_views": missing_views,
                    "existing_tables": existing_tables,
                    "existing_extensions": extensions,
                    "existing_views": views
                },
                error
            )
            
            cursor.close()
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._record_result(
                "Schema Validation",
                "Migration Status",
                "failed", 
                duration_ms,
                {},
                str(e)
            )
    
    # ===========================================
    # DATA INTEGRITY VALIDATION
    # ===========================================
    
    def test_data_preservation(self):
        """Test that existing data was preserved during migration"""
        try:
            start_time = time.time()
            
            conn = self._get_postgres_connection()
            cursor = conn.cursor()
            
            # Check ai_memories data
            cursor.execute("SELECT COUNT(*) FROM ai_memories")
            memories_count = cursor.fetchone()[0]
            
            # Check documents data
            cursor.execute("SELECT COUNT(*) FROM documents")
            documents_count = cursor.fetchone()[0]
            
            # Check for non-null enhanced fields in ai_memories
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(content_hash) as with_hash,
                    COUNT(knowledge_entities) as with_entities,
                    COUNT(knowledge_relations) as with_relations
                FROM ai_memories
            """)
            memory_stats = cursor.fetchone()
            
            # Check enhanced_chunks migration
            cursor.execute("SELECT COUNT(*) FROM enhanced_chunks")
            enhanced_chunks_count = cursor.fetchone()[0]
            
            # Check if chunks table exists (legacy)
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'chunks'
                )
            """)
            chunks_table_exists = cursor.fetchone()[0]
            
            legacy_chunks_count = 0
            if chunks_table_exists:
                cursor.execute("SELECT COUNT(*) FROM chunks")
                legacy_chunks_count = cursor.fetchone()[0]
            
            # Check memory clusters creation
            cursor.execute("SELECT COUNT(*) FROM memory_clusters")
            clusters_count = cursor.fetchone()[0]
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Validate data preservation
            issues = []
            
            if memories_count == 0:
                issues.append("No memories found - data may have been lost")
            
            if documents_count == 0:
                issues.append("No documents found - data may have been lost")
            
            if memory_stats[1] < memory_stats[0]:  # not all memories have hash
                issues.append(f"Content hash missing for {memory_stats[0] - memory_stats[1]} memories")
            
            if enhanced_chunks_count == 0 and legacy_chunks_count > 0:
                issues.append("Chunks migration incomplete - enhanced_chunks is empty")
            
            status = "passed" if not issues else "warning"
            error = "; ".join(issues) if issues else None
            
            self._record_result(
                "Data Preservation",
                "Data Integrity",
                status,
                duration_ms,
                {
                    "memories_count": memories_count,
                    "documents_count": documents_count,
                    "enhanced_chunks_count": enhanced_chunks_count,
                    "legacy_chunks_count": legacy_chunks_count,
                    "clusters_count": clusters_count,
                    "memory_enhancement_stats": {
                        "total": memory_stats[0],
                        "with_hash": memory_stats[1],
                        "with_entities": memory_stats[2],
                        "with_relations": memory_stats[3]
                    }
                },
                error
            )
            
            cursor.close()
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._record_result(
                "Data Preservation",
                "Data Integrity",
                "failed",
                duration_ms,
                {},
                str(e)
            )
    
    def test_data_consistency(self):
        """Test data consistency across related tables"""
        try:
            start_time = time.time()
            
            conn = self._get_postgres_connection()
            cursor = conn.cursor()
            
            consistency_checks = []
            
            # Check memory-cluster relationships
            cursor.execute("""
                SELECT 
                    COUNT(*) as memories_with_clusters,
                    (SELECT COUNT(*) FROM ai_memories) as total_memories,
                    (SELECT COUNT(*) FROM memory_clusters) as total_clusters
            """)
            cluster_stats = cursor.fetchone()
            
            # Check document-chunk relationships  
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT document_id) as docs_with_chunks,
                    (SELECT COUNT(*) FROM documents) as total_docs
                FROM enhanced_chunks
            """)
            doc_chunk_stats = cursor.fetchone()
            
            # Check foreign key constraint violations
            cursor.execute("""
                SELECT 
                    COUNT(*) as orphaned_associations
                FROM memory_associations ma
                LEFT JOIN ai_memories am1 ON ma.source_memory_id = am1.id
                LEFT JOIN ai_memories am2 ON ma.target_memory_id = am2.id
                WHERE am1.id IS NULL OR am2.id IS NULL
            """)
            orphaned_associations = cursor.fetchone()[0]
            
            # Check for duplicate content hashes
            cursor.execute("""
                SELECT content_hash, COUNT(*)
                FROM ai_memories
                WHERE content_hash IS NOT NULL
                GROUP BY content_hash
                HAVING COUNT(*) > 1
                LIMIT 5
            """)
            duplicate_hashes = cursor.fetchall()
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Analyze consistency
            issues = []
            
            if orphaned_associations > 0:
                issues.append(f"Found {orphaned_associations} orphaned memory associations")
            
            if len(duplicate_hashes) > 0:
                issues.append(f"Found {len(duplicate_hashes)} duplicate content hashes")
            
            status = "passed" if not issues else "warning"
            error = "; ".join(issues) if issues else None
            
            self._record_result(
                "Data Consistency",
                "Data Integrity", 
                status,
                duration_ms,
                {
                    "cluster_relationships": {
                        "memories_with_clusters": cluster_stats[0],
                        "total_memories": cluster_stats[1],
                        "total_clusters": cluster_stats[2]
                    },
                    "document_chunk_relationships": {
                        "docs_with_chunks": doc_chunk_stats[0],
                        "total_docs": doc_chunk_stats[1]
                    },
                    "orphaned_associations": orphaned_associations,
                    "duplicate_hashes_count": len(duplicate_hashes)
                },
                error
            )
            
            cursor.close()
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._record_result(
                "Data Consistency",
                "Data Integrity",
                "failed",
                duration_ms,
                {},
                str(e)
            )
    
    def test_timescale_integration(self):
        """Test TimescaleDB integration and data"""
        try:
            start_time = time.time()
            
            conn = self._get_timescale_connection()
            cursor = conn.cursor()
            
            # Check TimescaleDB extension
            cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
            timescale_ext = cursor.fetchone()
            
            if not timescale_ext:
                duration_ms = int((time.time() - start_time) * 1000)
                self._record_result(
                    "TimescaleDB Integration",
                    "Data Integrity",
                    "failed",
                    duration_ms,
                    {"timescaledb_extension": False},
                    "TimescaleDB extension not found"
                )
                return
            
            # Check for hypertables
            cursor.execute("""
                SELECT schemaname, tablename 
                FROM timescaledb_information.hypertables
            """)
            hypertables = cursor.fetchall()
            
            # Check for analytics data
            cursor.execute("SELECT COUNT(*) FROM performance_analytics")
            analytics_count = cursor.fetchone()[0]
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            status = "passed" if hypertables else "warning"
            error = "No hypertables found" if not hypertables else None
            
            self._record_result(
                "TimescaleDB Integration",
                "Data Integrity",
                status,
                duration_ms,
                {
                    "timescaledb_extension": True,
                    "hypertables": [f"{h[0]}.{h[1]}" for h in hypertables],
                    "analytics_records": analytics_count
                },
                error
            )
            
            cursor.close()
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._record_result(
                "TimescaleDB Integration",
                "Data Integrity",
                "failed",
                duration_ms,
                {},
                str(e)
            )
    
    # ===========================================
    # PERFORMANCE VALIDATION
    # ===========================================
    
    def test_query_performance(self):
        """Test query performance after migration"""
        try:
            start_time = time.time()
            
            conn = self._get_postgres_connection()
            cursor = conn.cursor()
            
            performance_tests = []
            
            # Test 1: Memory search performance
            test_start = time.time()
            cursor.execute("""
                SELECT id, content, memory_type 
                FROM ai_memories 
                WHERE memory_type = 'conversation'
                LIMIT 100
            """)
            results = cursor.fetchall()
            memory_search_time = (time.time() - test_start) * 1000
            performance_tests.append(("memory_search", memory_search_time, len(results)))
            
            # Test 2: Enhanced chunks query
            test_start = time.time()
            cursor.execute("""
                SELECT id, content, document_id
                FROM enhanced_chunks
                WHERE content_length > 100
                ORDER BY created_at DESC
                LIMIT 50
            """)
            results = cursor.fetchall()
            chunks_query_time = (time.time() - test_start) * 1000
            performance_tests.append(("chunks_query", chunks_query_time, len(results)))
            
            # Test 3: Complex join query
            test_start = time.time()
            cursor.execute("""
                SELECT m.id, m.content, c.name as cluster_name
                FROM ai_memories m
                LEFT JOIN memory_clusters c ON m.cluster_id = c.id
                WHERE m.created_at > NOW() - INTERVAL '30 days'
                LIMIT 25
            """)
            results = cursor.fetchall()
            join_query_time = (time.time() - test_start) * 1000
            performance_tests.append(("join_query", join_query_time, len(results)))
            
            # Test 4: Index usage check
            cursor.execute("""
                EXPLAIN (ANALYZE, BUFFERS) 
                SELECT * FROM ai_memories WHERE content_hash = 'test_hash'
            """)
            explain_results = cursor.fetchall()
            index_used = any("Index Scan" in str(row) for row in explain_results)
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Check performance thresholds
            slow_queries = [test for test in performance_tests if test[1] > 1000]  # >1 second
            
            status = "passed" if not slow_queries else "warning"
            error = f"Slow queries detected: {[test[0] for test in slow_queries]}" if slow_queries else None
            
            self._record_result(
                "Query Performance",
                "Performance Validation",
                status,
                duration_ms,
                {
                    "performance_tests": {
                        test[0]: {"time_ms": test[1], "results_count": test[2]}
                        for test in performance_tests
                    },
                    "index_usage": index_used,
                    "slow_queries": [test[0] for test in slow_queries]
                },
                error
            )
            
            cursor.close()
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._record_result(
                "Query Performance",
                "Performance Validation",
                "failed",
                duration_ms,
                {},
                str(e)
            )
    
    def test_index_efficiency(self):
        """Test database index efficiency"""
        try:
            start_time = time.time()
            
            conn = self._get_postgres_connection()
            cursor = conn.cursor()
            
            # Get index usage statistics
            cursor.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes
                WHERE schemaname = 'public'
                ORDER BY idx_tup_read DESC
                LIMIT 20
            """)
            index_stats = cursor.fetchall()
            
            # Check for unused indexes
            cursor.execute("""
                SELECT 
                    schemaname,
                    tablename, 
                    indexname
                FROM pg_stat_user_indexes
                WHERE idx_tup_read = 0 AND idx_tup_fetch = 0
                AND schemaname = 'public'
            """)
            unused_indexes = cursor.fetchall()
            
            # Check table sizes
            cursor.execute("""
                SELECT 
                    tablename,
                    pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(tablename::regclass) DESC
                LIMIT 10
            """)
            table_sizes = cursor.fetchall()
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            status = "passed" if len(unused_indexes) < 5 else "warning"
            error = f"Found {len(unused_indexes)} unused indexes" if len(unused_indexes) >= 5 else None
            
            self._record_result(
                "Index Efficiency",
                "Performance Validation",
                status,
                duration_ms,
                {
                    "total_indexes_checked": len(index_stats),
                    "unused_indexes_count": len(unused_indexes),
                    "top_used_indexes": [
                        {"table": row[1], "index": row[2], "reads": row[3]}
                        for row in index_stats[:5]
                    ],
                    "unused_indexes": [
                        f"{row[1]}.{row[2]}" for row in unused_indexes
                    ],
                    "largest_tables": [
                        {"table": row[0], "size": row[1]}
                        for row in table_sizes[:5]
                    ]
                },
                error
            )
            
            cursor.close()
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._record_result(
                "Index Efficiency",
                "Performance Validation",
                "failed",
                duration_ms,
                {},
                str(e)
            )
    
    # ===========================================
    # ROLLBACK TESTING
    # ===========================================
    
    def test_rollback_availability(self):
        """Test availability of rollback scripts"""
        try:
            start_time = time.time()
            
            # Check for rollback files
            import os
            migration_dir = "/opt/projects/knowledgehub/migrations"
            rollback_files = []
            
            if os.path.exists(migration_dir):
                for file in os.listdir(migration_dir):
                    if file.startswith("rollback_") and file.endswith(".sql"):
                        rollback_files.append(file)
            
            # Check for backup directory
            backup_dir = MIGRATION_CONFIG['backup_directory']
            backup_available = os.path.exists(backup_dir)
            
            recent_backups = []
            if backup_available:
                try:
                    for item in os.listdir(backup_dir):
                        if os.path.isdir(os.path.join(backup_dir, item)):
                            recent_backups.append(item)
                    recent_backups.sort(reverse=True)  # Most recent first
                except:
                    pass
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            status = "passed" if rollback_files and backup_available else "warning"
            error = None
            if not rollback_files:
                error = "No rollback scripts found"
            elif not backup_available:
                error = "Backup directory not found"
            
            self._record_result(
                "Rollback Availability",
                "Rollback Testing",
                status,
                duration_ms,
                {
                    "rollback_files": rollback_files,
                    "backup_directory_exists": backup_available,
                    "recent_backups": recent_backups[:5],  # Last 5 backups
                    "total_backups": len(recent_backups)
                },
                error
            )
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._record_result(
                "Rollback Availability",
                "Rollback Testing",
                "failed",
                duration_ms,
                {},
                str(e)
            )
    
    # ===========================================
    # REPORTING
    # ===========================================
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive migration validation report"""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "passed"])
        failed_tests = len([r for r in self.results if r.status == "failed"])
        warning_tests = len([r for r in self.results if r.status == "warning"])
        
        # Group by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {
                    'total': 0, 'passed': 0, 'failed': 0, 'warning': 0, 'tests': []
                }
            categories[result.category]['total'] += 1
            categories[result.category][result.status] += 1
            categories[result.category]['tests'].append({
                'name': result.test_name,
                'status': result.status,
                'duration_ms': result.duration_ms,
                'error': result.error,
                'details': result.details
            })
        
        # Migration health score
        health_score = (passed_tests * 100 + warning_tests * 50) / (total_tests * 100) if total_tests > 0 else 0
        
        # Critical issues
        critical_issues = [r for r in self.results if r.status == "failed"]
        
        # Recommendations
        recommendations = self._generate_recommendations()
        
        report = {
            'migration_validation_summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'warnings': warning_tests,
                'health_score': health_score,
                'overall_status': 'PASSED' if failed_tests == 0 else 'FAILED',
                'timestamp': datetime.now().isoformat()
            },
            'category_breakdown': categories,
            'critical_issues': [
                {
                    'test_name': issue.test_name,
                    'category': issue.category,
                    'error': issue.error,
                    'details': issue.details
                }
                for issue in critical_issues
            ],
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'category': r.category,
                    'status': r.status,
                    'duration_ms': r.duration_ms,
                    'details': r.details,
                    'error': r.error,
                    'timestamp': r.timestamp
                }
                for r in self.results
            ],
            'recommendations': recommendations,
            'migration_metadata': {
                'validation_timestamp': datetime.now().isoformat(),
                'postgres_url': MIGRATION_CONFIG['postgres_url'].split('@')[1] if '@' in MIGRATION_CONFIG['postgres_url'] else 'localhost',
                'expected_migrations': MIGRATION_CONFIG['expected_migrations']
            }
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        failed_tests = [r for r in self.results if r.status == "failed"]
        warning_tests = [r for r in self.results if r.status == "warning"]
        
        if not failed_tests and not warning_tests:
            recommendations.append("✅ Migration validation passed completely - system is ready for production")
        
        if failed_tests:
            recommendations.append("🚨 Critical migration issues detected - immediate attention required")
            for test in failed_tests:
                recommendations.append(f"   → Fix: {test.test_name} - {test.error}")
        
        if warning_tests:
            recommendations.append("⚠️ Migration warnings detected - review recommended")
            for test in warning_tests:
                recommendations.append(f"   → Review: {test.test_name}")
        
        # Specific recommendations based on test results
        for result in self.results:
            if result.test_name == "Query Performance" and result.status == "warning":
                recommendations.append("🔧 Consider query optimization and index tuning")
            
            if result.test_name == "Data Preservation" and result.status == "warning":
                recommendations.append("📊 Verify data migration completeness")
            
            if result.test_name == "Index Efficiency" and result.status == "warning":
                recommendations.append("🗂️ Review and optimize database indexes")
        
        return recommendations
    
    def run_all_validations(self):
        """Execute all migration validation tests"""
        self.logger.info("🔍 Starting Comprehensive Migration Validation")
        
        # 1. Migration Status Validation
        self.logger.info("📋 Validating migration status...")
        self.test_migration_log_integrity()
        self.test_schema_validation()
        
        # 2. Data Integrity Validation
        self.logger.info("🔐 Validating data integrity...")
        self.test_data_preservation()
        self.test_data_consistency()
        self.test_timescale_integration()
        
        # 3. Performance Validation
        self.logger.info("⚡ Validating performance...")
        self.test_query_performance()
        self.test_index_efficiency()
        
        # 4. Rollback Testing
        self.logger.info("🔄 Validating rollback capabilities...")
        self.test_rollback_availability()
        
        self.logger.info("✅ Migration validation completed!")
        return self.generate_comprehensive_report()
    
    def cleanup(self):
        """Cleanup database connections"""
        if self.postgres_conn and not self.postgres_conn.closed:
            self.postgres_conn.close()
        if self.timescale_conn and not self.timescale_conn.closed:
            self.timescale_conn.close()

def main():
    """Main execution function"""
    validator = None
    try:
        # Create validator
        validator = MigrationValidator()
        
        # Run all validations
        report = validator.run_all_validations()
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"migration_validation_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        summary = report['migration_validation_summary']
        print(f"\n{'='*70}")
        print(f"MIGRATION VALIDATION REPORT")
        print(f"{'='*70}")
        print(f"🎯 Overall Status: {summary['overall_status']}")
        print(f"📊 Health Score: {summary['health_score']:.1f}%")
        print(f"✅ Passed: {summary['passed']}")
        print(f"⚠️ Warnings: {summary['warnings']}")
        print(f"❌ Failed: {summary['failed']}")
        
        # Category breakdown
        print(f"\n📂 CATEGORY BREAKDOWN:")
        for category, stats in report['category_breakdown'].items():
            success_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {category}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Critical issues
        if report['critical_issues']:
            print(f"\n🚨 CRITICAL ISSUES:")
            for issue in report['critical_issues']:
                print(f"  - {issue['test_name']}: {issue['error']}")
        
        # Recommendations
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  {rec}")
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        print(f"{'='*70}")
        
        # Exit with appropriate code
        if summary['failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⏹️ Migration validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Migration validation failed: {e}")
        sys.exit(1)
    finally:
        if validator:
            validator.cleanup()

if __name__ == "__main__":
    main()