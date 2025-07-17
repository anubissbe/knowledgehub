#!/bin/bash
# Full KnowledgeHub Backup Script
# This script backs up all databases, configurations, and data

set -e

# Timestamp for backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_ROOT="/opt/projects/knowledgehub/backups"
BACKUP_DIR="${BACKUP_ROOT}/${TIMESTAMP}"

echo "🔄 Starting KnowledgeHub full backup..."
echo "📁 Backup directory: $BACKUP_DIR"

# Create backup directories
mkdir -p "$BACKUP_DIR"/{postgres,timescale,neo4j,weaviate,redis,minio,config,code}

# 1. Backup PostgreSQL (main database)
echo "📊 Backing up PostgreSQL..."
# Try different possible container names
if docker exec knowledgehub-postgres pg_dump -U knowledgehub knowledgehub > "$BACKUP_DIR/postgres/knowledgehub.sql" 2>/dev/null; then
    echo "✅ PostgreSQL backup complete"
elif docker exec knowledgehub-postgres-1 pg_dump -U knowledgehub knowledgehub > "$BACKUP_DIR/postgres/knowledgehub.sql" 2>/dev/null; then
    echo "✅ PostgreSQL backup complete"
else
    echo "⚠️ PostgreSQL backup skipped - container not running"
fi

# 2. Backup TimescaleDB (analytics database)
echo "📈 Backing up TimescaleDB..."
docker exec knowledgehub-timescale-1 pg_dump -U knowledgehub knowledgehub_analytics > "$BACKUP_DIR/timescale/knowledgehub_analytics.sql"
echo "✅ TimescaleDB backup complete"

# 3. Backup Neo4j (knowledge graph)
echo "🔗 Backing up Neo4j..."
docker exec knowledgehub-neo4j-1 neo4j-admin database dump neo4j --to-path=/backups/neo4j_backup.dump
docker cp knowledgehub-neo4j-1:/backups/neo4j_backup.dump "$BACKUP_DIR/neo4j/"
echo "✅ Neo4j backup complete"

# 4. Backup Weaviate (vector database)
echo "🧮 Backing up Weaviate..."
# Weaviate backup via API
curl -X POST "http://localhost:8090/v1/backups/filesystem" \
  -H "Content-Type: application/json" \
  -d "{\"id\": \"backup-${TIMESTAMP}\", \"include\": [\"KnowledgeChunk\"]}" || echo "⚠️ Weaviate backup API not available, copying data directory"

# Copy Weaviate data directory as fallback
docker cp knowledgehub-weaviate-1:/var/lib/weaviate "$BACKUP_DIR/weaviate/" 2>/dev/null || echo "⚠️ Weaviate data copy skipped"

# 5. Backup Redis (cache and queues)
echo "💾 Backing up Redis..."
docker exec knowledgehub-redis-1 redis-cli --rdb /data/dump.rdb BGSAVE
sleep 2
docker cp knowledgehub-redis-1:/data/dump.rdb "$BACKUP_DIR/redis/"
echo "✅ Redis backup complete"

# 6. Backup MinIO (object storage)
echo "📦 Backing up MinIO data..."
# Create MinIO backup directory
mkdir -p "$BACKUP_DIR/minio/data"
# Copy MinIO data
docker cp knowledgehub-minio-1:/data "$BACKUP_DIR/minio/" 2>/dev/null || echo "⚠️ MinIO data copy skipped"

# 7. Backup configuration files
echo "⚙️ Backing up configuration files..."
cp -r /opt/projects/knowledgehub/*.yml "$BACKUP_DIR/config/" 2>/dev/null || true
cp -r /opt/projects/knowledgehub/*.env "$BACKUP_DIR/config/" 2>/dev/null || true
cp -r /opt/projects/knowledgehub/*.json "$BACKUP_DIR/config/" 2>/dev/null || true
cp -r /opt/projects/knowledgehub/nginx.conf "$BACKUP_DIR/config/" 2>/dev/null || true

# 8. Backup application code (excluding node_modules and build artifacts)
echo "💻 Backing up application code..."
rsync -av --exclude='node_modules' --exclude='dist' --exclude='build' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
  --exclude='backups' \
  /opt/projects/knowledgehub/ "$BACKUP_DIR/code/"

# 9. Create Docker container info
echo "🐳 Saving Docker container information..."
docker ps -a --filter "name=knowledgehub" --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" > "$BACKUP_DIR/docker_containers.txt"
docker-compose -f /opt/projects/knowledgehub/docker-compose.yml config > "$BACKUP_DIR/docker-compose-resolved.yml"

# 10. Create backup metadata
echo "📝 Creating backup metadata..."
cat > "$BACKUP_DIR/backup_metadata.json" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "date": "$(date)",
  "hostname": "$(hostname)",
  "backup_type": "full",
  "components": {
    "postgres": "$(docker exec knowledgehub-postgres-1 psql -U knowledgehub -d knowledgehub -c 'SELECT COUNT(*) FROM documents;' -t | xargs)",
    "timescale": "$(docker exec knowledgehub-timescale-1 psql -U knowledgehub -d knowledgehub_analytics -c 'SELECT COUNT(*) FROM document_metrics;' -t | xargs)",
    "neo4j": "$(curl -s -u neo4j:knowledgehub123 -X POST http://localhost:7474/db/neo4j/tx/commit -H 'Content-Type: application/json' -d '{"statements":[{"statement":"MATCH (n) RETURN count(n)"}]}' | jq -r '.results[0].data[0].row[0]')",
    "weaviate": "$(curl -s http://localhost:8090/v1/objects?class=KnowledgeChunk&limit=0 | jq -r '.totalResults // 0')"
  },
  "docker_version": "$(docker --version)",
  "docker_compose_version": "$(docker-compose --version)"
}
EOF

# 11. Compress the backup
echo "🗜️ Compressing backup..."
cd "$BACKUP_ROOT"
tar -czf "knowledgehub_backup_${TIMESTAMP}.tar.gz" "${TIMESTAMP}/"

# 12. Create checksum
echo "🔐 Creating checksum..."
sha256sum "knowledgehub_backup_${TIMESTAMP}.tar.gz" > "knowledgehub_backup_${TIMESTAMP}.sha256"

# 13. Display backup summary
echo ""
echo "✅ KnowledgeHub backup completed successfully!"
echo "📦 Backup archive: ${BACKUP_ROOT}/knowledgehub_backup_${TIMESTAMP}.tar.gz"
echo "🔐 Checksum file: ${BACKUP_ROOT}/knowledgehub_backup_${TIMESTAMP}.sha256"
echo "📏 Backup size: $(du -h ${BACKUP_ROOT}/knowledgehub_backup_${TIMESTAMP}.tar.gz | cut -f1)"
echo ""
echo "📊 Backup contents:"
du -sh "$BACKUP_DIR"/*

# 14. Cleanup uncompressed backup directory (optional)
# rm -rf "$BACKUP_DIR"