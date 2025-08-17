# Hybrid RAG Migration - Quick Reference

## 🚀 Deployment Commands

### Full Migration Deployment
```bash
# Production deployment
python3 deploy_migration.py

# Dry run (recommended first)
python3 deploy_migration.py --dry-run

# Skip backup (not recommended)
python3 deploy_migration.py --skip-backup --force
```

### Manual Migration Steps
```bash
# 1. Schema migration
python3 -c "
import sys; sys.path.insert(0, 'api')
from config.database_config import run_migrations
run_migrations(['migrations/004_hybrid_rag_schema.sql'])
"

# 2. Data migration  
python3 -c "
import sys; sys.path.insert(0, 'api')
from config.database_config import run_migrations
run_migrations(['migrations/005_data_migration.sql'])
"

# 3. Validation
python3 scripts/validate_migration.py
```

### Rollback Commands
```bash
# Emergency rollback
psql -h localhost -p 5433 -U knowledgehub -d knowledgehub -f migrations/rollback_004_005.sql

# Full restore from backup
psql -h localhost -p 5433 -U knowledgehub -d knowledgehub < backup_pre_migration_YYYYMMDD_HHMMSS.sql
```

## 📋 Key Files Created

### Migration Scripts
- `migrations/004_hybrid_rag_schema.sql` - Schema migration (19 tables, 30+ indexes, 4 views)
- `migrations/005_data_migration.sql` - Data migration with preservation
- `migrations/rollback_004_005.sql` - Complete rollback script

### Model Files
- `api/models/agent_workflow.py` - Multi-agent workflow models
- `api/models/hybrid_rag.py` - RAG system models  
- `api/models/service_integration.py` - Service integration models

### Configuration
- `api/config/__init__.py` - Package initialization
- `api/config/database_config.py` - Enhanced database configuration

### Validation & Deployment
- `scripts/validate_migration.py` - Comprehensive validation script
- `deploy_migration.py` - Automated deployment orchestrator
- `MIGRATION_SAFETY_PROCEDURES.md` - Complete safety procedures

## 🎯 What's New

### Database Schema
- **Agent Workflows**: Multi-agent orchestration with LangGraph
- **Hybrid RAG**: Dense + sparse + graph retrieval
- **Enhanced Memory**: Clustering, associations, decay factors
- **Service Integration**: Health monitoring, performance tracking
- **Advanced Analytics**: Time-series data, performance metrics

### Key Features
- **Zero Data Loss**: All existing data preserved and enhanced
- **Performance Optimized**: 30+ strategic indexes
- **Service Health**: Real-time monitoring and alerting
- **Cache Management**: Intelligent result caching
- **Multi-tenant Ready**: Scalable service configuration

## ⚡ Quick Validation

```bash
# Check database connection
python3 -c "
import sys; sys.path.insert(0, 'api')
from config.database_config import initialize_databases, db_manager
initialize_databases()
session = db_manager.get_postgres_session('primary')
print('✅ Database connected')
"

# Check migration status
python3 scripts/validate_migration.py | grep "Overall Status"

# Count new tables
psql -h localhost -p 5433 -U knowledgehub -d knowledgehub -c "
SELECT COUNT(*) as new_tables FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name ~ '^(memory_clusters|agent_|workflow_|rag_|enhanced_|service_)';
"
```

## 🚨 Emergency Contacts

- **Migration Issues**: Check deployment logs
- **Data Loss**: Use rollback procedures immediately  
- **Performance**: Monitor query performance post-migration
- **Application Errors**: Restart services and check model imports

## 📊 Expected Results

### Before Migration
- **Tables**: ~3-5 core tables (ai_memories, documents, chunks)
- **Features**: Basic memory storage

### After Migration  
- **Tables**: 19 new tables + enhanced existing
- **Features**: Hybrid RAG, multi-agent workflows, service health monitoring
- **Performance**: Similar or improved with new indexes
- **Data**: 100% preserved + enhanced metadata

---

**Status**: ✅ MIGRATION SYSTEM COMPLETE AND READY FOR DEPLOYMENT