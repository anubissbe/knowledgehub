# KnowledgeHub Automated Backup Setup

## Overview

This guide explains how to set up automated backups for KnowledgeHub, including all data stores:
- PostgreSQL database (metadata, relationships)
- Redis (cache, session data)
- Weaviate (vector embeddings)
- MinIO (object storage)

## Components

### 1. Backup Script (`scripts/backup.sh`)
- Main backup script that handles all data stores
- Creates compressed backups with timestamps
- Automatic cleanup of old backups (30 days retention)
- Creates backup manifest with metadata

### 2. Cron Wrapper (`scripts/backup-cron.sh`)
- Wrapper script for automated execution
- Handles logging and notifications
- Maintains backup logs

### 3. Restore Script (`scripts/restore.sh`)
- Restores data from backups
- Supports full or partial restoration
- Interactive component selection

## Setup Instructions

### Option 1: Using Cron (Recommended for Synology)

1. **Make scripts executable**:
```bash
chmod +x /opt/projects/knowledgehub/scripts/backup.sh
chmod +x /opt/projects/knowledgehub/scripts/backup-cron.sh
chmod +x /opt/projects/knowledgehub/scripts/restore.sh
```

2. **Add to crontab**:
```bash
# Edit crontab
crontab -e

# Add daily backup at 2:00 AM
0 2 * * * /opt/projects/knowledgehub/scripts/backup-cron.sh
```

3. **Verify cron is running**:
```bash
# Check cron service
sudo systemctl status crond || sudo systemctl status cron

# List current crontab
crontab -l
```

### Option 2: Using systemd Timer (Linux Systems)

1. **Copy systemd files**:
```bash
sudo cp /opt/projects/knowledgehub/scripts/knowledgehub-backup.service /etc/systemd/system/
sudo cp /opt/projects/knowledgehub/scripts/knowledgehub-backup.timer /etc/systemd/system/
```

2. **Enable and start timer**:
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable timer to start on boot
sudo systemctl enable knowledgehub-backup.timer

# Start timer
sudo systemctl start knowledgehub-backup.timer

# Check status
sudo systemctl status knowledgehub-backup.timer
```

3. **Test backup service**:
```bash
# Run backup manually
sudo systemctl start knowledgehub-backup.service

# Check logs
sudo journalctl -u knowledgehub-backup.service -f
```

## Manual Backup

Run backup manually at any time:
```bash
/opt/projects/knowledgehub/scripts/backup.sh
```

## Backup Location and Structure

Backups are stored in: `/opt/projects/knowledgehub/backups/`

Structure:
```
backups/
├── 20250106_140000/          # Backup directory (timestamp)
├── postgresql_20250106_140000.sql.gz
├── redis_20250106_140000.rdb.gz
├── weaviate_20250106_140000.tar.gz
├── minio_20250106_140000.tar.gz
├── config_20250106_140000.tar.gz
└── manifest_20250106_140000.json
```

## Restoration

### Full Restore
```bash
# List available backups
ls -la /opt/projects/knowledgehub/backups/

# Restore from specific backup
/opt/projects/knowledgehub/scripts/restore.sh 20250106_140000
```

### Partial Restore
The restore script offers options to restore individual components:
1. All components
2. PostgreSQL only
3. Redis only
4. Weaviate only
5. MinIO only

## Monitoring

### Check Backup Logs
```bash
# View recent backup logs
tail -f /opt/projects/knowledgehub/logs/backup.log

# Check systemd timer logs (if using systemd)
sudo journalctl -u knowledgehub-backup.timer -f
```

### Verify Backup Success
```bash
# Check latest backup
ls -lah /opt/projects/knowledgehub/backups/ | tail -5

# View backup manifest
cat /opt/projects/knowledgehub/backups/manifest_*.json | jq '.' | tail -20
```

## Configuration

### Modify Backup Settings

Edit `/opt/projects/knowledgehub/scripts/backup.sh`:

```bash
# Backup location
BACKUP_DIR="/opt/projects/knowledgehub/backups"

# Retention period (days)
RETENTION_DAYS=30

# Database credentials
POSTGRES_USER="${POSTGRES_USER:-khuser}"
POSTGRES_DB="${POSTGRES_DB:-knowledgehub}"
```

### Change Backup Schedule

**For cron**: Edit crontab
```bash
crontab -e
# Change timing: MIN HOUR * * *
# Example: 30 3 * * * (3:30 AM daily)
```

**For systemd**: Edit timer
```bash
sudo systemctl edit knowledgehub-backup.timer
# Modify OnCalendar value
```

## Best Practices

1. **Regular Testing**: Periodically test restore procedures
2. **Off-site Backups**: Copy backups to remote location
3. **Monitor Disk Space**: Ensure adequate space for backups
4. **Verify Backups**: Check backup integrity regularly
5. **Document Changes**: Update this guide when modifying backup procedures

## Troubleshooting

### Backup Fails

1. Check container status:
```bash
docker ps | grep knowledgehub
```

2. Check disk space:
```bash
df -h /opt/projects/knowledgehub/backups
```

3. Check permissions:
```bash
ls -la /opt/projects/knowledgehub/scripts/
```

4. Review logs:
```bash
tail -100 /opt/projects/knowledgehub/logs/backup.log
```

### Restore Fails

1. Verify backup files exist:
```bash
ls -la /opt/projects/knowledgehub/backups/<backup_id>/
```

2. Check container status:
```bash
docker ps -a | grep knowledgehub
```

3. Check Docker logs:
```bash
docker logs knowledgehub-postgres --tail=50
```

## Security Considerations

1. **Backup Encryption**: Consider encrypting backups for sensitive data
2. **Access Control**: Restrict backup directory permissions
3. **Secure Transfer**: Use encrypted channels for off-site backup
4. **Retention Policy**: Balance retention needs with security requirements

## Disaster Recovery Plan

1. **Regular Backups**: Daily automated backups
2. **Off-site Storage**: Copy to remote location weekly
3. **Test Restores**: Monthly restore testing
4. **Documentation**: Keep this guide updated
5. **Contact Info**: Maintain emergency contact list