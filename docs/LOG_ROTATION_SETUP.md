# KnowledgeHub Log Rotation Implementation

**Date**: 2025-07-06  
**Status**: ✅ COMPLETED

## Overview

Implemented comprehensive log rotation for all KnowledgeHub services to prevent disk space issues and maintain manageable log files.

## Implementation

### 1. Docker Container Log Rotation

Added log rotation configuration to docker-compose.yml for key services:

```yaml
logging:
  driver: json-file
  options:
    max-size: "50m"      # Maximum size per log file
    max-file: "5"        # Keep 5 rotated files
    compress: "true"     # Compress rotated logs
```

Applied to:
- API Gateway (knowledgehub-api)
- Scraper Service (knowledgehub-scraper)
- RAG Processor (knowledgehub-rag)

### 2. Application Log Rotation

Created `/opt/projects/knowledgehub/config/logrotate.conf` with:

- **API logs**: Daily rotation, 14 days retention, 100MB max
- **Scraper logs**: Daily rotation, 7 days retention, 50MB max
- **RAG logs**: Daily rotation, 7 days retention, 50MB max
- **Scheduler logs**: Weekly rotation, 4 weeks retention, 20MB max
- **Backup logs**: Monthly rotation, 12 months retention, 10MB max
- **Cleanup logs**: Monthly rotation, 6 months retention, 10MB max

### 3. Docker Daemon Configuration

Created `/opt/projects/knowledgehub/config/docker-logrotate.json`:

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "5",
    "compress": "true",
    "labels": "com.knowledgehub.service",
    "env": "APP_ENV,SERVICE_NAME"
  }
}
```

### 4. Setup and Monitoring Scripts

- **setup-log-rotation.sh**: Installs logrotate configuration
- **check-log-rotation.sh**: Monitors rotation status

## Installation

### Manual Setup

1. **Create log directories**:
```bash
mkdir -p /opt/projects/knowledgehub/logs/{api,scraper,rag,scheduler}
```

2. **Install logrotate configuration** (requires sudo):
```bash
sudo cp /opt/projects/knowledgehub/config/logrotate.conf /etc/logrotate.d/knowledgehub
sudo chmod 644 /etc/logrotate.d/knowledgehub
```

3. **Test configuration**:
```bash
sudo logrotate -d /etc/logrotate.d/knowledgehub
```

4. **Apply Docker logging** (restart containers):
```bash
cd /opt/projects/knowledgehub
docker compose up -d --force-recreate api scraper rag-processor
```

### Automated Setup

Run the setup script:
```bash
/opt/projects/knowledgehub/scripts/setup-log-rotation.sh
```

## Monitoring

### Check Log Sizes
```bash
# Application logs
find /opt/projects/knowledgehub/logs -name "*.log" -type f -exec ls -lh {} \;

# Docker container logs
docker ps --format "{{.Names}}" | grep knowledgehub | xargs -I {} docker inspect {} | jq -r '.[0].Name + ": " + .HostConfig.LogConfig.Type'
```

### Check Rotation Status
```bash
# Last rotation
sudo grep knowledgehub /var/lib/logrotate/status

# Force rotation test
sudo logrotate -f /etc/logrotate.d/knowledgehub
```

### Use Monitoring Script
```bash
/opt/projects/knowledgehub/scripts/check-log-rotation.sh
```

## Benefits

1. **Prevents Disk Full**: Automatic rotation prevents runaway log growth
2. **Compressed Storage**: Old logs are compressed to save space
3. **Configurable Retention**: Different retention policies per service
4. **Performance**: Smaller active log files improve performance
5. **Debugging**: Keeps sufficient history for troubleshooting

## Log File Locations

```
/opt/projects/knowledgehub/logs/
├── api/            # API Gateway logs
├── scraper/        # Scraper service logs
├── rag/            # RAG processor logs
├── scheduler/      # Scheduler logs
├── backup.log      # Backup operation logs
└── cleanup.log     # Cleanup operation logs
```

## Configuration Details

### Rotation Schedule
- **Daily**: API, Scraper, RAG (high-volume services)
- **Weekly**: Scheduler (moderate volume)
- **Monthly**: Backup, Cleanup (low volume)

### Retention Policy
- **API**: 14 days (for debugging recent issues)
- **Processing Services**: 7 days
- **Operations**: 12 months (backup), 6 months (cleanup)

### Size Limits
- **API**: 100MB (high traffic)
- **Services**: 50MB (moderate traffic)
- **Operations**: 10MB (low volume)

## Troubleshooting

### Logs Not Rotating
1. Check logrotate service: `systemctl status logrotate`
2. Run manually: `sudo logrotate -f /etc/logrotate.d/knowledgehub`
3. Check permissions: `ls -la /opt/projects/knowledgehub/logs/`

### Docker Logs Growing
1. Check current config: `docker inspect <container> | jq .HostConfig.LogConfig`
2. Recreate container: `docker compose up -d --force-recreate <service>`
3. Verify new config: `docker inspect <container> | grep -A5 LogConfig`

### Finding Large Logs
```bash
# System-wide
find /var/log -type f -size +100M

# Docker specific
find /var/lib/docker/containers -name "*.log" -size +100M
```

## Next Steps

1. **Monitor**: Check log sizes weekly
2. **Adjust**: Fine-tune rotation schedules based on actual usage
3. **Centralize**: Consider log aggregation (already have Loki stack)
4. **Alert**: Set up monitoring for log rotation failures