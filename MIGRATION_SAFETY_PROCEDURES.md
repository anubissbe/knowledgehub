# Migration Safety Procedures for Hybrid RAG System

## Overview

This document outlines the complete safety procedures for deploying the Hybrid RAG database migration. These procedures ensure data safety, system reliability, and the ability to recover from any issues.

## Pre-Migration Checklist

### 1. Environment Verification
- [ ] **Database Connection**: Verify connection to PostgreSQL (localhost:5433)
- [ ] **User Permissions**: Ensure `knowledgehub` user has CREATE/DROP/ALTER privileges
- [ ] **Disk Space**: Verify at least 2GB free space for migration and backups
- [ ] **Memory**: Ensure at least 1GB available RAM during migration
- [ ] **Dependencies**: Confirm all required Python packages are installed

### 2. Backup Procedures
- [ ] **Full Database Backup**: Create complete PostgreSQL dump
- [ ] **Critical Table Exports**: Export ai_memories, documents, chunks tables
- [ ] **Configuration Backup**: Backup current application configuration
- [ ] **Docker Volumes**: Backup postgres_data volume if using Docker

```bash
# Full database backup
pg_dump -h localhost -p 5433 -U knowledgehub -d knowledgehub > backup_pre_migration_$(date +%Y%m%d_%H%M%S).sql

# Critical tables backup
pg_dump -h localhost -p 5433 -U knowledgehub -d knowledgehub -t ai_memories -t documents -t chunks --data-only > critical_data_backup_$(date +%Y%m%d_%H%M%S).sql
```

### 3. System Health Check
- [ ] **Run validation script**: `python3 scripts/validate_migration.py`
- [ ] **Check existing data integrity**: Verify no corrupted records
- [ ] **Monitor resource usage**: Ensure system is not under heavy load
- [ ] **Test application functionality**: Verify current system works correctly

## Migration Execution Procedures

### Phase 1: Schema Migration (004_hybrid_rag_schema.sql)

**Duration**: ~2-5 minutes  
**Downtime**: No downtime required (additive changes)

```bash
# 1. Execute schema migration
python3 -c "
import sys
sys.path.insert(0, 'api')
from config.database_config import run_migrations
run_migrations(['migrations/004_hybrid_rag_schema.sql'])
"

# 2. Verify schema creation
python3 scripts/validate_migration.py --phase schema
```

**Expected Results**:
- ✅ 19 new tables created
- ✅ 30+ indexes created 
- ✅ 4 views created
- ✅ Foreign key constraints established

### Phase 2: Data Migration (005_data_migration.sql)

**Duration**: ~5-15 minutes (depends on data volume)  
**Downtime**: 30 seconds (during final data updates)

```bash
# 1. Put application in maintenance mode (if needed)
# 2. Execute data migration
python3 -c "
import sys
sys.path.insert(0, 'api')
from config.database_config import run_migrations
run_migrations(['migrations/005_data_migration.sql'])
"

# 3. Verify data migration
python3 scripts/validate_migration.py --phase data
```

**Expected Results**:
- ✅ All existing memories preserved and enhanced
- ✅ Documents migrated to ingestion logs
- ✅ Chunks converted to enhanced_chunks
- ✅ Default configurations created
- ✅ Service configurations established

### Phase 3: Final Validation

```bash
# Complete validation
python3 scripts/validate_migration.py

# Application restart
# Restart KnowledgeHub API to pick up new models
```

## Rollback Procedures

### Emergency Rollback (if critical issues occur)

**Maximum Time**: 5 minutes

```bash
# 1. IMMEDIATE: Stop application
docker-compose stop api

# 2. Execute rollback script
psql -h localhost -p 5433 -U knowledgehub -d knowledgehub -f migrations/rollback_004_005.sql

# 3. Restore from backup if needed
psql -h localhost -p 5433 -U knowledgehub -d knowledgehub < backup_pre_migration_YYYYMMDD_HHMMSS.sql

# 4. Restart application
docker-compose start api
```

### Selective Rollback (for specific issues)

```sql
-- Roll back only specific tables
SELECT safe_rollback_table('rag_configurations', true);
SELECT safe_rollback_table('agent_definitions', true);

-- Remove only specific columns
SELECT safe_remove_column('ai_memories', 'cluster_id');
```

## Monitoring and Alerts

### During Migration
- **CPU Usage**: Monitor for spikes >80%
- **Memory Usage**: Monitor for usage >90%
- **Disk I/O**: Watch for sustained high I/O
- **PostgreSQL Logs**: Monitor for errors or warnings
- **Lock Monitoring**: Check for long-running locks

### Post-Migration
- **Application Performance**: Response times should remain similar
- **Database Performance**: Query times should not increase significantly
- **Data Integrity**: Run periodic data consistency checks
- **User Experience**: Monitor for user-reported issues

## Safety Measures

### 1. Transaction Safety
- All migrations run in transactions with automatic rollback on errors
- Foreign key constraints prevent data corruption
- Comprehensive error handling with detailed logging

### 2. Data Preservation
- Zero data loss design - all existing data is preserved
- Enhanced fields are added, not replaced
- Backward compatibility maintained for existing queries

### 3. Performance Safety
- Indexes created with `CONCURRENTLY` where possible
- Statistics updated after migration
- Query performance monitored

### 4. Recovery Points
- **Before Migration**: Full database backup
- **After Schema**: Schema-only backup point
- **After Data**: Complete migration backup point
- **Rollback Log**: All changes logged in rollback_backup_log

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Migration Hangs/Fails
```bash
# Check for locks
SELECT * FROM pg_locks WHERE NOT granted;

# Check migration log
SELECT * FROM migration_log ORDER BY completed_at DESC;

# Manual intervention
-- Kill hanging queries if safe
-- Restart PostgreSQL if necessary
```

#### 2. Foreign Key Violations
```sql
-- Check for orphaned records
SELECT COUNT(*) FROM ai_memories 
WHERE cluster_id IS NOT NULL 
AND cluster_id NOT IN (SELECT id FROM memory_clusters);

-- Fix orphaned records
UPDATE ai_memories SET cluster_id = NULL 
WHERE cluster_id NOT IN (SELECT id FROM memory_clusters);
```

#### 3. Performance Degradation
```sql
-- Update statistics
ANALYZE;

-- Check missing indexes
SELECT * FROM pg_stat_user_tables WHERE schemaname = 'public';

-- Recreate critical indexes if needed
CREATE INDEX CONCURRENTLY idx_ai_memories_content_hash ON ai_memories(content_hash);
```

### 4. Application Compatibility Issues
```python
# Test model imports
try:
    from models.hybrid_rag import RAGConfiguration
    from models.agent_workflow import AgentDefinition
    print("✅ Model imports successful")
except ImportError as e:
    print(f"❌ Model import failed: {e}")

# Test database connections
from config.database_config import db_manager
session = db_manager.get_postgres_session("primary")
result = session.execute("SELECT COUNT(*) FROM ai_memories")
print(f"✅ Database connection successful: {result.scalar()} memories")
```

## Post-Migration Procedures

### 1. Application Testing
- [ ] **Basic Functionality**: Test core memory operations
- [ ] **New Features**: Test hybrid RAG queries
- [ ] **Performance**: Verify response times
- [ ] **Integration**: Test with external services

### 2. Performance Optimization
- [ ] **Query Analysis**: Analyze slow query log
- [ ] **Index Optimization**: Add indexes for frequent queries
- [ ] **Cache Warming**: Pre-populate Redis cache
- [ ] **Statistics Update**: Ensure accurate query planning

### 3. Documentation Updates
- [ ] **API Documentation**: Update with new endpoints
- [ ] **Schema Documentation**: Document new table structures
- [ ] **User Guide**: Update user-facing documentation
- [ ] **Operations Guide**: Update deployment procedures

## Contact Information

**Migration Owner**: DATABASE MIGRATION AGENT  
**Emergency Contact**: System Administrator  
**Escalation Path**: DevOps Team → Database Team → Architecture Team

## Migration Log Template

```
Migration Date: YYYY-MM-DD HH:MM:SS
Migration Version: 004_hybrid_rag_schema + 005_data_migration
Environment: Production/Staging/Development
Executed By: [Name/System]

Pre-Migration Status:
- Database Size: XXX MB
- Record Counts: ai_memories(XXX), documents(XXX), chunks(XXX)
- Performance Baseline: Avg response time XXXms

Migration Results:
- Schema Migration: SUCCESS/FAILED
- Data Migration: SUCCESS/FAILED
- Validation: SUCCESS/FAILED
- Total Duration: XXX minutes

Post-Migration Status:
- New Table Count: XXX
- Data Preserved: 100%/XX%
- Performance Impact: +/-XX%
- Issues Encountered: None/[List]

Rollback Status: Not Required/Applied Successfully
Next Steps: [Action items]
```

---

**Remember**: When in doubt, stop the migration and consult the team. Data safety is the top priority.