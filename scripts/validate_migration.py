#!/usr/bin/env python3
"""
Database Migration Validation Script.

This script validates the hybrid RAG database migration by checking:
1. Schema integrity
2. Data preservation
3. Performance indexes
4. Service configurations
5. Data consistency
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any, List, Tuple
from datetime import datetime
import json

# Add the API directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from config.database_config import initialize_databases, db_manager, check_migration_status
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MigrationValidator:
    """Validates database migration completeness and correctness."""
    
    def __init__(self):
        self.validation_results = {
            "schema_validation": {},
            "data_validation": {},
            "performance_validation": {},
            "service_validation": {},
            "overall_status": "pending",
            "timestamp": datetime.utcnow().isoformat(),
            "errors": [],
            "warnings": []
        }
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete migration validation."""
        logger.info("Starting comprehensive migration validation...")
        
        try:
            # Initialize database connections
            initialize_databases()
            
            # Validate schema
            await self.validate_schema()
            
            # Validate data preservation
            await self.validate_data_preservation()
            
            # Validate performance indexes
            await self.validate_performance_indexes()
            
            # Validate service configurations
            await self.validate_service_configurations()
            
            # Validate data consistency
            await self.validate_data_consistency()
            
            # Determine overall status
            self.determine_overall_status()
            
            logger.info("Migration validation completed")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.validation_results["errors"].append({
                "type": "critical",
                "message": f"Validation process failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            })
            self.validation_results["overall_status"] = "failed"
        
        return self.validation_results
    
    async def validate_schema(self):
        """Validate database schema completeness."""
        logger.info("Validating database schema...")
        
        schema_validation = {
            "tables_created": {},
            "indexes_created": {},
            "constraints_created": {},
            "triggers_created": {},
            "views_created": {}
        }
        
        session = db_manager.get_postgres_session("primary")
        
        try:
            # Check required tables
            required_tables = [
                'memory_clusters',
                'memory_associations', 
                'memory_access_logs',
                'agent_definitions',
                'workflow_definitions',
                'workflow_executions',
                'agent_tasks',
                'rag_configurations',
                'rag_query_logs',
                'document_ingestion_logs',
                'enhanced_chunks',
                'search_result_cache',
                'zep_session_mapping',
                'firecrawl_jobs',
                'service_health_logs',
                'performance_monitoring',
                'service_configurations',
                'service_integration_logs',
                'service_dependencies'
            ]
            
            for table in required_tables:
                result = session.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = '{table}'
                    )
                """))
                
                exists = result.scalar()
                schema_validation["tables_created"][table] = exists
                
                if not exists:
                    self.validation_results["errors"].append({
                        "type": "schema",
                        "message": f"Required table '{table}' does not exist",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            # Check critical indexes
            critical_indexes = [
                'idx_ai_memories_content_hash',
                'idx_ai_memories_embeddings',
                'idx_memory_clusters_type',
                'idx_agent_definitions_role',
                'idx_workflow_executions_status',
                'idx_rag_query_logs_query_hash',
                'idx_enhanced_chunks_document_id',
                'idx_service_health_logs_service'
            ]
            
            for index in critical_indexes:
                result = session.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM pg_indexes 
                        WHERE indexname = '{index}'
                    )
                """))
                
                exists = result.scalar()
                schema_validation["indexes_created"][index] = exists
                
                if not exists:
                    self.validation_results["warnings"].append({
                        "type": "performance",
                        "message": f"Critical index '{index}' is missing",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            # Check foreign key constraints
            result = session.execute(text("""
                SELECT conname, conrelid::regclass as table_name
                FROM pg_constraint 
                WHERE contype = 'f'
                AND connamespace = 'public'::regnamespace
            """))
            
            constraints = result.fetchall()
            schema_validation["constraints_created"] = {
                row[0]: str(row[1]) for row in constraints
            }
            
            # Check views
            result = session.execute(text("""
                SELECT table_name 
                FROM information_schema.views 
                WHERE table_schema = 'public'
                AND table_name IN (
                    'enhanced_memory_view',
                    'workflow_performance_view', 
                    'rag_performance_view',
                    'service_health_summary'
                )
            """))
            
            views = [row[0] for row in result.fetchall()]
            for view in ['enhanced_memory_view', 'workflow_performance_view', 
                        'rag_performance_view', 'service_health_summary']:
                schema_validation["views_created"][view] = view in views
            
            self.validation_results["schema_validation"] = schema_validation
            logger.info("Schema validation completed")
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            self.validation_results["errors"].append({
                "type": "schema",
                "message": f"Schema validation error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            })
        finally:
            session.close()
    
    async def validate_data_preservation(self):
        """Validate that existing data was preserved and enhanced."""
        logger.info("Validating data preservation...")
        
        data_validation = {
            "memories_preserved": 0,
            "documents_preserved": 0,
            "chunks_migrated": 0,
            "clusters_created": 0,
            "default_configs_created": 0,
            "data_integrity_checks": {}
        }
        
        session = db_manager.get_postgres_session("primary")
        
        try:
            # Check memories preservation
            if self.table_exists(session, 'ai_memories'):
                result = session.execute(text("SELECT COUNT(*) FROM ai_memories"))
                data_validation["memories_preserved"] = result.scalar() or 0
                
                # Check enhanced fields
                result = session.execute(text("""
                    SELECT COUNT(*) FROM ai_memories 
                    WHERE content_hash IS NOT NULL AND content_hash != ''
                """))
                memories_with_hash = result.scalar() or 0
                
                data_validation["data_integrity_checks"]["memories_with_content_hash"] = {
                    "count": memories_with_hash,
                    "percentage": (memories_with_hash / max(data_validation["memories_preserved"], 1)) * 100
                }
            
            # Check documents preservation
            if self.table_exists(session, 'documents'):
                result = session.execute(text("SELECT COUNT(*) FROM documents"))
                data_validation["documents_preserved"] = result.scalar() or 0
            
            # Check chunks migration
            if self.table_exists(session, 'enhanced_chunks'):
                result = session.execute(text("SELECT COUNT(*) FROM enhanced_chunks"))
                data_validation["chunks_migrated"] = result.scalar() or 0
            
            # Check memory clusters creation
            result = session.execute(text("SELECT COUNT(*) FROM memory_clusters"))
            data_validation["clusters_created"] = result.scalar() or 0
            
            # Check default configurations
            result = session.execute(text("SELECT COUNT(*) FROM rag_configurations WHERE is_default = true"))
            data_validation["default_configs_created"] = result.scalar() or 0
            
            if data_validation["default_configs_created"] == 0:
                self.validation_results["warnings"].append({
                    "type": "configuration",
                    "message": "No default RAG configuration found",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Check agent definitions
            result = session.execute(text("SELECT COUNT(*) FROM agent_definitions"))
            default_agents = result.scalar() or 0
            data_validation["data_integrity_checks"]["default_agents"] = default_agents
            
            if default_agents < 4:  # researcher, analyst, synthesizer, validator
                self.validation_results["warnings"].append({
                    "type": "configuration",
                    "message": f"Only {default_agents} default agents found, expected at least 4",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            self.validation_results["data_validation"] = data_validation
            logger.info("Data preservation validation completed")
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            self.validation_results["errors"].append({
                "type": "data",
                "message": f"Data validation error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            })
        finally:
            session.close()
    
    async def validate_performance_indexes(self):
        """Validate that performance indexes are properly created."""
        logger.info("Validating performance indexes...")
        
        performance_validation = {
            "index_statistics": {},
            "query_performance": {},
            "table_statistics": {}
        }
        
        session = db_manager.get_postgres_session("primary")
        
        try:
            # Check index usage statistics
            result = session.execute(text("""
                SELECT 
                    schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
                FROM pg_stat_user_indexes 
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname
            """))
            
            indexes = result.fetchall()
            for row in indexes:
                index_key = f"{row[1]}.{row[2]}"  # table.index
                performance_validation["index_statistics"][index_key] = {
                    "scans": row[3] or 0,
                    "tuples_read": row[4] or 0,
                    "tuples_fetched": row[5] or 0
                }
            
            # Check table sizes and statistics
            result = session.execute(text("""
                SELECT 
                    tablename,
                    pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size,
                    n_tup_ins, n_tup_upd, n_tup_del, n_live_tup, n_dead_tup
                FROM pg_stat_user_tables 
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(tablename::regclass) DESC
            """))
            
            tables = result.fetchall()
            for row in tables:
                performance_validation["table_statistics"][row[0]] = {
                    "size": row[1],
                    "inserts": row[2] or 0,
                    "updates": row[3] or 0,
                    "deletes": row[4] or 0,
                    "live_tuples": row[5] or 0,
                    "dead_tuples": row[6] or 0
                }
            
            # Test query performance on key tables
            test_queries = [
                {
                    "name": "memory_lookup_by_hash",
                    "query": "SELECT COUNT(*) FROM ai_memories WHERE content_hash = 'test_hash'",
                    "expected_performance": "fast"
                },
                {
                    "name": "cluster_memory_count",
                    "query": "SELECT COUNT(*) FROM memory_clusters WHERE cluster_type = 'test'",
                    "expected_performance": "fast"
                },
                {
                    "name": "workflow_status_lookup", 
                    "query": "SELECT COUNT(*) FROM workflow_executions WHERE status = 'completed'",
                    "expected_performance": "fast"
                }
            ]
            
            for test in test_queries:
                start_time = datetime.utcnow()
                session.execute(text(test["query"]))
                end_time = datetime.utcnow()
                
                execution_time = (end_time - start_time).total_seconds() * 1000  # ms
                performance_validation["query_performance"][test["name"]] = {
                    "execution_time_ms": execution_time,
                    "expected": test["expected_performance"],
                    "status": "good" if execution_time < 100 else "slow"
                }
            
            self.validation_results["performance_validation"] = performance_validation
            logger.info("Performance validation completed")
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            self.validation_results["errors"].append({
                "type": "performance",
                "message": f"Performance validation error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            })
        finally:
            session.close()
    
    async def validate_service_configurations(self):
        """Validate service configurations and health."""
        logger.info("Validating service configurations...")
        
        service_validation = {
            "database_connections": {},
            "service_configs": {},
            "health_checks": {}
        }
        
        try:
            # Test database connections
            health = await db_manager.health_check()
            service_validation["database_connections"] = health
            
            # Check service configurations table
            session = db_manager.get_postgres_session("primary")
            
            try:
                if self.table_exists(session, 'service_configurations'):
                    result = session.execute(text("""
                        SELECT service_type, service_name, is_active, is_default
                        FROM service_configurations
                        ORDER BY service_type, service_name
                    """))
                    
                    configs = result.fetchall()
                    for row in configs:
                        service_key = f"{row[0]}.{row[1]}"
                        service_validation["service_configs"][service_key] = {
                            "type": row[0],
                            "name": row[1], 
                            "active": row[2],
                            "default": row[3]
                        }
                
                # Check required service types
                required_services = ['postgresql', 'redis', 'weaviate', 'neo4j', 'zep_memory']
                for service_type in required_services:
                    result = session.execute(
                        text("SELECT COUNT(*) FROM service_configurations WHERE service_type = :service_type"),
                        {"service_type": service_type}
                    )
                    count = result.scalar() or 0
                    
                    if count == 0:
                        self.validation_results["warnings"].append({
                            "type": "configuration",
                            "message": f"No configuration found for service type: {service_type}",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                
            finally:
                session.close()
            
            # Test Redis connection if configured
            try:
                redis_client = db_manager.get_redis_client("cache")
                redis_client.ping()
                service_validation["health_checks"]["redis"] = "healthy"
            except Exception as e:
                service_validation["health_checks"]["redis"] = f"error: {str(e)}"
                self.validation_results["warnings"].append({
                    "type": "service",
                    "message": f"Redis health check failed: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            self.validation_results["service_validation"] = service_validation
            logger.info("Service validation completed")
            
        except Exception as e:
            logger.error(f"Service validation failed: {e}")
            self.validation_results["errors"].append({
                "type": "service",
                "message": f"Service validation error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def validate_data_consistency(self):
        """Validate data consistency and referential integrity."""
        logger.info("Validating data consistency...")
        
        consistency_validation = {
            "referential_integrity": {},
            "data_quality": {},
            "constraint_violations": {}
        }
        
        session = db_manager.get_postgres_session("primary")
        
        try:
            # Check foreign key constraint violations
            if self.table_exists(session, 'ai_memories') and self.table_exists(session, 'memory_clusters'):
                result = session.execute(text("""
                    SELECT COUNT(*) FROM ai_memories 
                    WHERE cluster_id IS NOT NULL 
                    AND cluster_id NOT IN (SELECT id FROM memory_clusters)
                """))
                orphaned_memories = result.scalar() or 0
                consistency_validation["referential_integrity"]["orphaned_memories"] = orphaned_memories
                
                if orphaned_memories > 0:
                    self.validation_results["errors"].append({
                        "type": "data_integrity",
                        "message": f"Found {orphaned_memories} memories with invalid cluster references",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            # Check workflow execution integrity
            if self.table_exists(session, 'workflow_executions') and self.table_exists(session, 'workflow_definitions'):
                result = session.execute(text("""
                    SELECT COUNT(*) FROM workflow_executions 
                    WHERE workflow_id NOT IN (SELECT id FROM workflow_definitions)
                """))
                orphaned_executions = result.scalar() or 0
                consistency_validation["referential_integrity"]["orphaned_executions"] = orphaned_executions
                
                if orphaned_executions > 0:
                    self.validation_results["errors"].append({
                        "type": "data_integrity",
                        "message": f"Found {orphaned_executions} workflow executions with invalid workflow references",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            # Check data quality
            if self.table_exists(session, 'ai_memories'):
                # Check for empty content
                result = session.execute(text("""
                    SELECT COUNT(*) FROM ai_memories 
                    WHERE content IS NULL OR content = ''
                """))
                empty_content = result.scalar() or 0
                consistency_validation["data_quality"]["memories_with_empty_content"] = empty_content
                
                # Check for missing content hashes
                result = session.execute(text("""
                    SELECT COUNT(*) FROM ai_memories 
                    WHERE content_hash IS NULL OR content_hash = ''
                """))
                missing_hashes = result.scalar() or 0
                consistency_validation["data_quality"]["memories_missing_hash"] = missing_hashes
                
                if missing_hashes > 0:
                    self.validation_results["warnings"].append({
                        "type": "data_quality",
                        "message": f"Found {missing_hashes} memories without content hashes",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            # Check enhanced chunks consistency
            if self.table_exists(session, 'enhanced_chunks'):
                result = session.execute(text("""
                    SELECT COUNT(*) FROM enhanced_chunks 
                    WHERE content_length != LENGTH(content)
                """))
                inconsistent_lengths = result.scalar() or 0
                consistency_validation["data_quality"]["chunks_inconsistent_length"] = inconsistent_lengths
                
                if inconsistent_lengths > 0:
                    self.validation_results["warnings"].append({
                        "type": "data_quality",
                        "message": f"Found {inconsistent_lengths} chunks with inconsistent content lengths",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            self.validation_results["data_validation"]["consistency"] = consistency_validation
            logger.info("Data consistency validation completed")
            
        except Exception as e:
            logger.error(f"Consistency validation failed: {e}")
            self.validation_results["errors"].append({
                "type": "consistency",
                "message": f"Consistency validation error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            })
        finally:
            session.close()
    
    def table_exists(self, session, table_name: str) -> bool:
        """Check if a table exists."""
        result = session.execute(text(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{table_name}'
            )
        """))
        return result.scalar()
    
    def determine_overall_status(self):
        """Determine overall validation status."""
        error_count = len(self.validation_results["errors"])
        warning_count = len(self.validation_results["warnings"])
        
        if error_count == 0:
            if warning_count == 0:
                self.validation_results["overall_status"] = "success"
            elif warning_count <= 3:
                self.validation_results["overall_status"] = "success_with_warnings"
            else:
                self.validation_results["overall_status"] = "degraded"
        elif error_count <= 2:
            self.validation_results["overall_status"] = "partial_success"
        else:
            self.validation_results["overall_status"] = "failed"
    
    def generate_report(self) -> str:
        """Generate a human-readable validation report."""
        report = []
        report.append("=" * 60)
        report.append("HYBRID RAG DATABASE MIGRATION VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {self.validation_results['timestamp']}")
        report.append(f"Overall Status: {self.validation_results['overall_status'].upper()}")
        report.append("")
        
        # Schema validation
        if "schema_validation" in self.validation_results:
            schema = self.validation_results["schema_validation"]
            report.append("SCHEMA VALIDATION:")
            report.append(f"  Tables Created: {sum(1 for v in schema['tables_created'].values() if v)}/{len(schema['tables_created'])}")
            report.append(f"  Indexes Created: {sum(1 for v in schema['indexes_created'].values() if v)}/{len(schema['indexes_created'])}")
            report.append(f"  Views Created: {sum(1 for v in schema['views_created'].values() if v)}/{len(schema['views_created'])}")
            report.append("")
        
        # Data validation
        if "data_validation" in self.validation_results:
            data = self.validation_results["data_validation"]
            report.append("DATA VALIDATION:")
            report.append(f"  Memories Preserved: {data.get('memories_preserved', 0)}")
            report.append(f"  Documents Preserved: {data.get('documents_preserved', 0)}")
            report.append(f"  Chunks Migrated: {data.get('chunks_migrated', 0)}")
            report.append(f"  Clusters Created: {data.get('clusters_created', 0)}")
            report.append("")
        
        # Errors and warnings
        if self.validation_results["errors"]:
            report.append("ERRORS:")
            for error in self.validation_results["errors"]:
                report.append(f"  - [{error['type']}] {error['message']}")
            report.append("")
        
        if self.validation_results["warnings"]:
            report.append("WARNINGS:")
            for warning in self.validation_results["warnings"]:
                report.append(f"  - [{warning['type']}] {warning['message']}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if self.validation_results["overall_status"] == "success":
            report.append("  - Migration completed successfully! No actions required.")
        elif self.validation_results["overall_status"] == "success_with_warnings":
            report.append("  - Migration successful but review warnings for optimization.")
        else:
            report.append("  - Review and fix errors before proceeding to production.")
            report.append("  - Consider running migration rollback if critical errors exist.")
        
        report.append("=" * 60)
        
        return "\n".join(report)


async def main():
    """Run migration validation."""
    validator = MigrationValidator()
    
    print("Starting database migration validation...")
    print("-" * 50)
    
    try:
        # Run validation
        results = await validator.run_full_validation()
        
        # Generate and display report
        report = validator.generate_report()
        print(report)
        
        # Save detailed results to file
        results_file = f"migration_validation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Exit with appropriate code
        if results["overall_status"] in ["success", "success_with_warnings"]:
            sys.exit(0)
        else:
            sys.exit(1)
    
    except Exception as e:
        print(f"Validation failed with error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())