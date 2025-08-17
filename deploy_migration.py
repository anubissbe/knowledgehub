#!/usr/bin/env python3
"""
Hybrid RAG Database Migration Deployment Script.

This script orchestrates the complete migration process with safety checks,
validation, and rollback capabilities.
"""

import os
import sys
import json
import subprocess
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import argparse

# Add the API directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

from config.database_config import initialize_databases, db_manager, run_migrations
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'migration_deploy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MigrationDeployer:
    """Orchestrates the hybrid RAG database migration deployment."""
    
    def __init__(self, dry_run: bool = False, skip_backup: bool = False):
        self.dry_run = dry_run
        self.skip_backup = skip_backup
        self.start_time = datetime.now()
        self.deployment_log = {
            "deployment_id": f"migration_{self.start_time.strftime('%Y%m%d_%H%M%S')}",
            "start_time": self.start_time.isoformat(),
            "phases": {},
            "errors": [],
            "warnings": [],
            "rollback_performed": False
        }
    
    def run_deployment(self) -> bool:
        """Run the complete migration deployment process."""
        logger.info("=" * 60)
        logger.info("HYBRID RAG DATABASE MIGRATION DEPLOYMENT")
        logger.info("=" * 60)
        
        if self.dry_run:
            logger.info("🔍 DRY RUN MODE - No changes will be made")
        
        try:
            # Phase 1: Pre-deployment checks
            if not self._run_phase("pre_checks", self._pre_deployment_checks):
                return False
            
            # Phase 2: Backup procedures
            if not self.skip_backup:
                if not self._run_phase("backup", self._create_backups):
                    return False
            else:
                logger.warning("⚠️ Skipping backup procedures as requested")
            
            # Phase 3: Schema migration
            if not self._run_phase("schema_migration", self._deploy_schema_migration):
                return False
            
            # Phase 4: Data migration  
            if not self._run_phase("data_migration", self._deploy_data_migration):
                return False
            
            # Phase 5: Post-deployment validation
            if not self._run_phase("validation", self._post_deployment_validation):
                return False
            
            # Phase 6: Final verification
            if not self._run_phase("final_verification", self._final_verification):
                return False
            
            self._deployment_success()
            return True
            
        except Exception as e:
            logger.error(f"Critical deployment error: {e}")
            self.deployment_log["errors"].append({
                "type": "critical",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            self._deployment_failed()
            return False
    
    def _run_phase(self, phase_name: str, phase_function) -> bool:
        """Run a deployment phase with error handling."""
        logger.info(f"\n🚀 Phase: {phase_name.upper()}")
        logger.info("-" * 40)
        
        phase_start = datetime.now()
        
        try:
            success = phase_function()
            
            phase_duration = (datetime.now() - phase_start).total_seconds()
            self.deployment_log["phases"][phase_name] = {
                "status": "success" if success else "failed",
                "duration_seconds": phase_duration,
                "timestamp": phase_start.isoformat()
            }
            
            if success:
                logger.info(f"✅ Phase {phase_name} completed successfully ({phase_duration:.1f}s)")
                return True
            else:
                logger.error(f"❌ Phase {phase_name} failed")
                return False
                
        except Exception as e:
            phase_duration = (datetime.now() - phase_start).total_seconds()
            logger.error(f"💥 Phase {phase_name} crashed: {e}")
            
            self.deployment_log["phases"][phase_name] = {
                "status": "error",
                "duration_seconds": phase_duration,
                "error": str(e),
                "timestamp": phase_start.isoformat()
            }
            return False
    
    def _pre_deployment_checks(self) -> bool:
        """Run pre-deployment safety checks."""
        logger.info("Running pre-deployment checks...")
        
        checks = [
            ("Database Connection", self._check_database_connection),
            ("Disk Space", self._check_disk_space),
            ("User Permissions", self._check_permissions),
            ("Current Schema", self._check_current_schema),
            ("Data Integrity", self._check_data_integrity)
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                passed = check_func()
                status = "✅ PASS" if passed else "❌ FAIL"
                logger.info(f"  {check_name}: {status}")
                
                if not passed:
                    all_passed = False
                    self.deployment_log["errors"].append({
                        "type": "pre_check",
                        "check": check_name,
                        "message": f"Pre-deployment check failed: {check_name}",
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                logger.error(f"  {check_name}: ❌ ERROR - {e}")
                all_passed = False
        
        return all_passed
    
    def _check_database_connection(self) -> bool:
        """Check database connectivity."""
        try:
            initialize_databases()
            session = db_manager.get_postgres_session("primary")
            result = session.execute(text("SELECT version()"))
            version = result.scalar()
            session.close()
            logger.info(f"    Connected to: {version[:50]}...")
            return True
        except Exception as e:
            logger.error(f"    Connection failed: {e}")
            return False
    
    def _check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            stat = os.statvfs('.')
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            logger.info(f"    Free space: {free_gb:.1f} GB")
            return free_gb >= 2.0  # Require at least 2GB
        except Exception as e:
            logger.error(f"    Disk check failed: {e}")
            return False
    
    def _check_permissions(self) -> bool:
        """Check database user permissions."""
        try:
            session = db_manager.get_postgres_session("primary")
            # Test CREATE permission
            session.execute(text("CREATE TABLE IF NOT EXISTS migration_test_table (id INTEGER)"))
            session.execute(text("DROP TABLE IF EXISTS migration_test_table"))
            session.commit()
            session.close()
            return True
        except Exception as e:
            logger.error(f"    Permission check failed: {e}")
            return False
    
    def _check_current_schema(self) -> bool:
        """Check current schema state."""
        try:
            session = db_manager.get_postgres_session("primary")
            
            # Check for existing core tables
            result = session.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('ai_memories', 'documents', 'chunks')
            """))
            core_tables = result.scalar()
            
            session.close()
            logger.info(f"    Core tables found: {core_tables}/3")
            return core_tables >= 1  # At least one core table should exist
        except Exception as e:
            logger.error(f"    Schema check failed: {e}")
            return False
    
    def _check_data_integrity(self) -> bool:
        """Check existing data integrity."""
        try:
            session = db_manager.get_postgres_session("primary")
            
            # Count existing records
            if self._table_exists(session, 'ai_memories'):
                result = session.execute(text("SELECT COUNT(*) FROM ai_memories"))
                memory_count = result.scalar()
                logger.info(f"    Existing memories: {memory_count}")
            
            session.close()
            return True
        except Exception as e:
            logger.error(f"    Data integrity check failed: {e}")
            return False
    
    def _table_exists(self, session, table_name: str) -> bool:
        """Check if a table exists."""
        result = session.execute(text(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{table_name}'
            )
        """))
        return result.scalar()
    
    def _create_backups(self) -> bool:
        """Create database backups."""
        logger.info("Creating database backups...")
        
        if self.dry_run:
            logger.info("  📋 Would create database backup")
            return True
        
        try:
            backup_dir = "backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{backup_dir}/pre_migration_backup_{timestamp}.sql"
            
            # Create full database backup
            cmd = [
                "pg_dump",
                "-h", "localhost",
                "-p", "5433", 
                "-U", "knowledgehub",
                "-d", "knowledgehub",
                "-f", backup_file
            ]
            
            logger.info(f"    Creating backup: {backup_file}")
            result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PGPASSWORD": "knowledgehub123"})
            
            if result.returncode == 0:
                file_size = os.path.getsize(backup_file) / (1024*1024)  # MB
                logger.info(f"    ✅ Backup created: {file_size:.1f} MB")
                
                self.deployment_log["backup_file"] = backup_file
                self.deployment_log["backup_size_mb"] = file_size
                return True
            else:
                logger.error(f"    ❌ Backup failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"    💥 Backup error: {e}")
            return False
    
    def _deploy_schema_migration(self) -> bool:
        """Deploy the schema migration."""
        logger.info("Deploying schema migration...")
        
        if self.dry_run:
            logger.info("  📋 Would run: 004_hybrid_rag_schema.sql")
            return True
        
        try:
            migration_file = "migrations/004_hybrid_rag_schema.sql"
            if not os.path.exists(migration_file):
                logger.error(f"    Migration file not found: {migration_file}")
                return False
            
            logger.info(f"    Executing: {migration_file}")
            run_migrations([migration_file])
            
            # Verify schema creation
            session = db_manager.get_postgres_session("primary")
            result = session.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN (
                    'memory_clusters', 'agent_definitions', 'rag_configurations'
                )
            """))
            new_tables = result.scalar()
            session.close()
            
            logger.info(f"    ✅ Schema migration completed. New tables: {new_tables}")
            return new_tables >= 3
            
        except Exception as e:
            logger.error(f"    💥 Schema migration failed: {e}")
            return False
    
    def _deploy_data_migration(self) -> bool:
        """Deploy the data migration."""
        logger.info("Deploying data migration...")
        
        if self.dry_run:
            logger.info("  📋 Would run: 005_data_migration.sql")
            return True
        
        try:
            migration_file = "migrations/005_data_migration.sql"
            if not os.path.exists(migration_file):
                logger.error(f"    Migration file not found: {migration_file}")
                return False
            
            logger.info(f"    Executing: {migration_file}")
            run_migrations([migration_file])
            
            logger.info("    ✅ Data migration completed")
            return True
            
        except Exception as e:
            logger.error(f"    💥 Data migration failed: {e}")
            return False
    
    def _post_deployment_validation(self) -> bool:
        """Run post-deployment validation."""
        logger.info("Running post-deployment validation...")
        
        try:
            # Import and run the validation script
            sys.path.insert(0, 'scripts')
            from validate_migration import MigrationValidator
            
            validator = MigrationValidator()
            results = None
            
            if not self.dry_run:
                # Run actual validation
                import asyncio
                results = asyncio.run(validator.run_full_validation())
            else:
                logger.info("  📋 Would run migration validation")
                return True
            
            if results:
                status = results.get("overall_status", "unknown")
                error_count = len(results.get("errors", []))
                warning_count = len(results.get("warnings", []))
                
                logger.info(f"    Validation status: {status}")
                logger.info(f"    Errors: {error_count}, Warnings: {warning_count}")
                
                # Store validation results
                self.deployment_log["validation_results"] = results
                
                return status in ["success", "success_with_warnings"]
            else:
                logger.error("    No validation results returned")
                return False
                
        except Exception as e:
            logger.error(f"    💥 Validation failed: {e}")
            return False
    
    def _final_verification(self) -> bool:
        """Final verification of the migration."""
        logger.info("Running final verification...")
        
        if self.dry_run:
            logger.info("  📋 Would verify migration completion")
            return True
        
        try:
            session = db_manager.get_postgres_session("primary")
            
            # Check migration log
            result = session.execute(text("""
                SELECT COUNT(*) FROM migration_log 
                WHERE migration_name IN ('004_hybrid_rag_schema', '005_data_migration')
            """))
            migration_entries = result.scalar()
            
            # Check data preservation
            if self._table_exists(session, 'ai_memories'):
                result = session.execute(text("SELECT COUNT(*) FROM ai_memories"))
                memory_count = result.scalar()
                logger.info(f"    Memories preserved: {memory_count}")
            
            session.close()
            
            logger.info(f"    Migration log entries: {migration_entries}")
            return migration_entries == 2
            
        except Exception as e:
            logger.error(f"    💥 Final verification failed: {e}")
            return False
    
    def _deployment_success(self):
        """Handle successful deployment."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 MIGRATION DEPLOYMENT SUCCESSFUL!")
        logger.info("=" * 60)
        logger.info(f"Total Duration: {duration:.1f} seconds")
        
        self.deployment_log["status"] = "success"
        self.deployment_log["end_time"] = datetime.now().isoformat()
        self.deployment_log["total_duration_seconds"] = duration
        
        self._save_deployment_log()
        
        logger.info("\n📋 Next Steps:")
        logger.info("1. Restart KnowledgeHub API service")
        logger.info("2. Test application functionality")
        logger.info("3. Monitor performance for 24 hours")
        logger.info("4. Update documentation")
    
    def _deployment_failed(self):
        """Handle failed deployment."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        logger.error("\n" + "=" * 60)
        logger.error("💥 MIGRATION DEPLOYMENT FAILED!")
        logger.error("=" * 60)
        logger.error(f"Duration: {duration:.1f} seconds")
        
        self.deployment_log["status"] = "failed"
        self.deployment_log["end_time"] = datetime.now().isoformat()
        self.deployment_log["total_duration_seconds"] = duration
        
        self._save_deployment_log()
        
        logger.error("\n🚨 Recovery Options:")
        logger.error("1. Review deployment log for specific errors")
        logger.error("2. Run rollback script: psql -f migrations/rollback_004_005.sql")
        logger.error("3. Restore from backup if necessary")
        logger.error("4. Contact database team for assistance")
    
    def _save_deployment_log(self):
        """Save the deployment log to file."""
        try:
            log_file = f"deployment_log_{self.deployment_log['deployment_id']}.json"
            with open(log_file, 'w') as f:
                json.dump(self.deployment_log, f, indent=2)
            logger.info(f"📄 Deployment log saved: {log_file}")
        except Exception as e:
            logger.error(f"Failed to save deployment log: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deploy Hybrid RAG Database Migration")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (no actual changes)")
    parser.add_argument("--skip-backup", action="store_true", help="Skip backup procedures (not recommended)")
    parser.add_argument("--force", action="store_true", help="Force deployment even with warnings")
    
    args = parser.parse_args()
    
    if args.skip_backup and not args.force:
        print("⚠️ WARNING: Skipping backups is dangerous!")
        response = input("Are you sure you want to proceed without backups? (yes/no): ")
        if response.lower() != 'yes':
            print("Deployment cancelled.")
            return
    
    deployer = MigrationDeployer(dry_run=args.dry_run, skip_backup=args.skip_backup)
    success = deployer.run_deployment()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()