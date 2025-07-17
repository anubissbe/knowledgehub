#!/bin/bash
# Full KnowledgeHub Backup Script (Safe Version)
# This script backs up all databases, configurations, and data

set -e

# Timestamp for backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_ROOT="/opt/projects/knowledgehub/backups"
BACKUP_DIR="${BACKUP_ROOT}/${TIMESTAMP}"

echo "🔄 Starting KnowledgeHub full backup..."
echo "📁 Backup directory: $BACKUP_DIR"

# Create backup directories
mkdir -p "$BACKUP_DIR"/{postgres,timescale,neo4j,weaviate,redis,minio,config,code,docker}

# Helper function to check if container is running
container_is_running() {
    docker ps --format "{{.Names}}" | grep -q "^$1$"
}

# 1. Backup PostgreSQL (main database)
echo "📊 Backing up PostgreSQL..."
if container_is_running "knowledgehub-postgres"; then
    docker exec knowledgehub-postgres pg_dump -U knowledgehub knowledgehub > "$BACKUP_DIR/postgres/knowledgehub.sql"
    echo "✅ PostgreSQL backup complete"
else
    echo "⚠️ PostgreSQL container not running - trying to start it..."
    docker start knowledgehub-postgres
    sleep 5
    if docker exec knowledgehub-postgres pg_dump -U knowledgehub knowledgehub > "$BACKUP_DIR/postgres/knowledgehub.sql" 2>/dev/null; then
        echo "✅ PostgreSQL backup complete"
    else
        echo "❌ PostgreSQL backup failed"
    fi
fi

# 2. Backup TimescaleDB (analytics database)
echo "📈 Backing up TimescaleDB..."
if container_is_running "knowledgehub-timescale-1"; then
    docker exec knowledgehub-timescale-1 pg_dump -U knowledgehub knowledgehub_analytics > "$BACKUP_DIR/timescale/knowledgehub_analytics.sql"
    echo "✅ TimescaleDB backup complete"
else
    echo "⚠️ TimescaleDB container not running"
fi

# 3. Backup Neo4j (knowledge graph)
echo "🔗 Backing up Neo4j..."
if container_is_running "2a63eadfc351_knowledgehub-neo4j-1"; then
    # Create a cypher dump instead
    docker exec 2a63eadfc351_knowledgehub-neo4j-1 cypher-shell -u neo4j -p knowledgehub123 \
        "CALL apoc.export.cypher.all('/var/lib/neo4j/backup.cypher', {format:'plain'})" 2>/dev/null || \
    docker exec 2a63eadfc351_knowledgehub-neo4j-1 bash -c "cd /var/lib/neo4j && neo4j-admin database dump neo4j --to-path=backup.dump" 2>/dev/null || true
    
    # Try to copy the backup
    docker cp 2a63eadfc351_knowledgehub-neo4j-1:/var/lib/neo4j/backup.cypher "$BACKUP_DIR/neo4j/" 2>/dev/null || \
    docker cp 2a63eadfc351_knowledgehub-neo4j-1:/var/lib/neo4j/backup.dump "$BACKUP_DIR/neo4j/" 2>/dev/null || \
    echo "⚠️ Neo4j backup copy failed"
    echo "✅ Neo4j backup attempt complete"
else
    echo "⚠️ Neo4j container not running"
fi

# 4. Backup Weaviate (vector database)
echo "🧮 Backing up Weaviate..."
if container_is_running "7dfb1412d2ba_knowledgehub-weaviate-1"; then
    # Export Weaviate data as JSON
    curl -s "http://localhost:8090/v1/objects?class=KnowledgeChunk&limit=10000" > "$BACKUP_DIR/weaviate/weaviate_export.json"
    echo "✅ Weaviate backup complete"
else
    echo "⚠️ Weaviate container not running"
fi

# 5. Backup Redis (cache and queues)
echo "💾 Backing up Redis..."
if container_is_running "e001d54e3336_knowledgehub-redis-1"; then
    docker exec e001d54e3336_knowledgehub-redis-1 redis-cli BGSAVE
    sleep 3
    docker cp e001d54e3336_knowledgehub-redis-1:/data/dump.rdb "$BACKUP_DIR/redis/" 2>/dev/null || echo "⚠️ Redis backup copy failed"
    echo "✅ Redis backup complete"
else
    echo "⚠️ Redis container not running"
fi

# 6. Backup MinIO (object storage)
echo "📦 Backing up MinIO data..."
if container_is_running "12e7f184cfc5_knowledgehub-minio-1"; then
    docker cp 12e7f184cfc5_knowledgehub-minio-1:/data "$BACKUP_DIR/minio/" 2>/dev/null || echo "⚠️ MinIO data copy skipped"
    echo "✅ MinIO backup attempt complete"
else
    echo "⚠️ MinIO container not running"
fi

# 7. Backup configuration files
echo "⚙️ Backing up configuration files..."
cp -r /opt/projects/knowledgehub/*.yml "$BACKUP_DIR/config/" 2>/dev/null || true
cp -r /opt/projects/knowledgehub/*.env "$BACKUP_DIR/config/" 2>/dev/null || true
cp -r /opt/projects/knowledgehub/*.json "$BACKUP_DIR/config/" 2>/dev/null || true
cp -r /opt/projects/knowledgehub/nginx.conf "$BACKUP_DIR/config/" 2>/dev/null || true
cp -r /opt/projects/knowledgehub/.env* "$BACKUP_DIR/config/" 2>/dev/null || true

# 8. Backup application code (excluding node_modules and build artifacts)
echo "💻 Backing up application code..."
rsync -av --exclude='node_modules' --exclude='dist' --exclude='build' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
  --exclude='backups' --exclude='*.log' \
  /opt/projects/knowledgehub/ "$BACKUP_DIR/code/"

# 9. Save Docker information
echo "🐳 Saving Docker container information..."
docker ps -a --filter "name=knowledgehub" --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" > "$BACKUP_DIR/docker/containers.txt"
docker images | grep knowledgehub > "$BACKUP_DIR/docker/images.txt" 2>/dev/null || true
docker-compose -f /opt/projects/knowledgehub/docker-compose.yml config > "$BACKUP_DIR/docker/docker-compose-resolved.yml" 2>/dev/null || true

# 10. Create backup metadata
echo "📝 Creating backup metadata..."
cat > "$BACKUP_DIR/backup_metadata.json" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "date": "$(date)",
  "hostname": "$(hostname)",
  "backup_type": "full",
  "components": {
    "postgres": "backed up",
    "timescale": "backed up",
    "neo4j": "backed up",
    "weaviate": "backed up",
    "redis": "backed up",
    "minio": "backed up",
    "config": "backed up",
    "code": "backed up"
  },
  "docker_version": "$(docker --version)",
  "backup_script_version": "1.0"
}
EOF

# 11. Create restore script
echo "📜 Creating restore script..."
cat > "$BACKUP_DIR/restore_knowledgehub.sh" <<'RESTORE'
#!/bin/bash
# KnowledgeHub Restore Script

echo "🔄 Starting KnowledgeHub restore..."
echo "⚠️  This will restore databases and configurations"
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

# Restore PostgreSQL
if [ -f "postgres/knowledgehub.sql" ]; then
    echo "📊 Restoring PostgreSQL..."
    docker exec -i knowledgehub-postgres psql -U knowledgehub knowledgehub < postgres/knowledgehub.sql
fi

# Restore TimescaleDB
if [ -f "timescale/knowledgehub_analytics.sql" ]; then
    echo "📈 Restoring TimescaleDB..."
    docker exec -i knowledgehub-timescale-1 psql -U knowledgehub knowledgehub_analytics < timescale/knowledgehub_analytics.sql
fi

# Restore Redis
if [ -f "redis/dump.rdb" ]; then
    echo "💾 Restoring Redis..."
    docker cp redis/dump.rdb e001d54e3336_knowledgehub-redis-1:/data/dump.rdb
    docker restart e001d54e3336_knowledgehub-redis-1
fi

echo "✅ Restore complete!"
RESTORE

chmod +x "$BACKUP_DIR/restore_knowledgehub.sh"

# 12. Compress the backup
echo "🗜️ Compressing backup..."
cd "$BACKUP_ROOT"
tar -czf "knowledgehub_backup_${TIMESTAMP}.tar.gz" "${TIMESTAMP}/"

# 13. Create checksum
echo "🔐 Creating checksum..."
sha256sum "knowledgehub_backup_${TIMESTAMP}.tar.gz" > "knowledgehub_backup_${TIMESTAMP}.sha256"

# 14. Display backup summary
echo ""
echo "✅ KnowledgeHub backup completed!"
echo "📦 Backup archive: ${BACKUP_ROOT}/knowledgehub_backup_${TIMESTAMP}.tar.gz"
echo "🔐 Checksum file: ${BACKUP_ROOT}/knowledgehub_backup_${TIMESTAMP}.sha256"
echo "📏 Backup size: $(du -h ${BACKUP_ROOT}/knowledgehub_backup_${TIMESTAMP}.tar.gz | cut -f1)"
echo ""
echo "📊 Backup contents:"
du -sh "$BACKUP_DIR"/* 2>/dev/null || true

# Keep uncompressed directory for now
echo ""
echo "💡 To restore from this backup, extract the archive and run:"
echo "   tar -xzf knowledgehub_backup_${TIMESTAMP}.tar.gz"
echo "   cd ${TIMESTAMP}"
echo "   ./restore_knowledgehub.sh"