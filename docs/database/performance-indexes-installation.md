# Performance Indexes Installation Guide

## Quick Installation

### Prerequisites

- PostgreSQL database with KnowledgeHub memory schema
- Database access credentials
- Python 3.11+ with asyncpg (for automated script)

### Option 1: Direct SQL Installation (Recommended)

```bash
# Navigate to KnowledgeHub directory
cd /opt/projects/knowledgehub

# Backup current indexes (optional but recommended)
psql -h localhost -U khuser -d knowledgehub -f src/api/database/backup_indexes.sql

# Apply performance indexes
psql -h localhost -U khuser -d knowledgehub -f src/api/database/performance_indexes.sql
```

### Option 2: Automated Installation with Monitoring

```bash
# Install Python dependencies (if needed)
pip install asyncpg

# Run automated installation script
python3 src/api/database/apply_performance_indexes.py
```

### Verification

Check that indexes were created successfully:

```sql
-- Count new indexes
SELECT COUNT(*) as total_indexes 
FROM pg_indexes 
WHERE tablename IN ('memories', 'memory_sessions');

-- List new performance indexes
SELECT indexname 
FROM pg_indexes 
WHERE tablename IN ('memories', 'memory_sessions') 
AND indexname LIKE '%_optimized%' 
ORDER BY indexname;
```

Expected result: 26 total indexes with several containing '_optimized' suffix.

## Docker Installation

If running in Docker environment:

```bash
# Copy SQL file to container
docker cp src/api/database/performance_indexes.sql knowledgehub-postgres:/tmp/

# Execute in container
docker exec knowledgehub-postgres psql -U khuser -d knowledgehub -f /tmp/performance_indexes.sql

# Verify installation
docker exec knowledgehub-postgres psql -U khuser -d knowledgehub -c "
SELECT COUNT(*) as total_indexes 
FROM pg_indexes 
WHERE tablename IN ('memories', 'memory_sessions');"
```

## Performance Testing

Quick performance test after installation:

```sql
-- Test session-based query performance
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM memories 
WHERE session_id = (SELECT id FROM memory_sessions ORDER BY created_at DESC LIMIT 1) 
ORDER BY importance DESC LIMIT 10;

-- Test importance-based query performance  
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM memories 
WHERE importance >= 0.8 
ORDER BY created_at DESC LIMIT 20;
```

Look for:
- Index usage in query plans (Index Scan vs Seq Scan)
- Execution time < 50ms for typical queries
- High buffer hit ratios

## Troubleshooting

### Common Issues

**Issue**: "Index already exists" errors
**Solution**: This is normal and expected. The script uses `IF NOT EXISTS` clauses.

**Issue**: Slow installation (>5 minutes)
**Solution**: Normal for large datasets. Run during low-traffic periods.

**Issue**: High memory usage during installation
**Solution**: PostgreSQL temporarily uses more memory for index building. Monitor system resources.

### Rollback if Needed

```sql
-- Emergency rollback (removes all new indexes)
DROP INDEX IF EXISTS idx_memories_session_type_importance_optimized;
DROP INDEX IF EXISTS idx_memories_created_importance_desc;
DROP INDEX IF EXISTS idx_memories_importance_range;
-- (See backup_indexes.sql for complete rollback script)
```

## Monitoring

After installation, monitor performance:

```sql
-- Check index usage after 24 hours
SELECT indexname, idx_scan, idx_tup_read 
FROM pg_stat_user_indexes 
WHERE tablename IN ('memories', 'memory_sessions') 
AND idx_scan > 0 
ORDER BY idx_scan DESC;

-- Update statistics for optimal performance
ANALYZE memories;
ANALYZE memory_sessions;
```

## Success Indicators

✅ Installation successful if:
- All 26+ indexes created without errors
- Query performance improved by 70-95%
- No increase in error rates
- Index usage statistics show active usage

For detailed information, see the [Complete Performance Indexes Guide](performance-indexes-guide.md).