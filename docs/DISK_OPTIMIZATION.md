# KnowledgeHub Disk Usage Optimization

**Date**: 2025-07-06  
**Status**: ✅ COMPLETED

## Overview

Implemented comprehensive disk usage optimization for the KnowledgeHub project, reducing disk usage from 74% to 45% (freed ~59GB).

## Problem

The root partition was at 74% capacity (152GB used out of 219GB), with significant space consumed by:
- Docker build cache: 52.47GB
- Unused Docker images: 80.88GB
- Git repository: 2.5GB
- Old backups and temporary files

## Solution Implemented

### 1. Disk Optimization Script
Created `/opt/projects/knowledgehub/scripts/optimize-disk-usage.sh` that:

- **Cleans old backups** (>30 days)
- **Removes npm cache** and node_modules
- **Deletes Python cache** (__pycache__, *.pyc, *.pyo)
- **Removes temporary files** (*.tmp, *.log >7 days, *.swp)
- **Optimizes Git repository** (gc --aggressive, repack)
- **Docker cleanup**:
  - Removes stopped containers
  - Removes dangling images
  - Cleans build cache
  - Optionally removes unused images (interactive)
  - Prunes unused volumes

### 2. Automated Weekly Cleanup
- Created `/opt/projects/knowledgehub/scripts/cleanup-cron.sh`
- Scheduled via cron: Sundays at 3:00 AM
- Runs non-interactive cleanup automatically
- Logs to `/opt/projects/knowledgehub/logs/cleanup.log`

### 3. Results

**Initial State**:
- Disk usage: 74% (152GB used)
- Docker images: 101.9GB total
- Build cache: 52.47GB

**After Optimization**:
- Disk usage: 45% (93GB used)
- **Total space freed: ~59GB**
- Project size reduced from 4GB to 1.3GB

## Usage

### Manual Cleanup
```bash
# Interactive mode (asks about Docker images)
/opt/projects/knowledgehub/scripts/optimize-disk-usage.sh

# Non-interactive mode
echo "n" | /opt/projects/knowledgehub/scripts/optimize-disk-usage.sh
```

### Check Disk Usage
```bash
# Overall disk usage
df -h /

# Docker disk usage
docker system df

# KnowledgeHub specific
du -h /opt/projects/knowledgehub --max-depth=2 | sort -rh | head -20
```

### View Cleanup Logs
```bash
tail -f /opt/projects/knowledgehub/logs/cleanup.log
```

## Cron Schedule

```bash
# Daily backup at 2 AM
0 2 * * * /opt/projects/knowledgehub/scripts/backup-cron.sh

# Weekly cleanup at 3 AM on Sundays
0 3 * * 0 /opt/projects/knowledgehub/scripts/cleanup-cron.sh
```

## Best Practices

1. **Regular Monitoring**: Check disk usage weekly
2. **Docker Images**: Periodically review and remove unused images
3. **Backup Retention**: Current policy is 30 days
4. **Log Rotation**: Implement for all services (next task)
5. **External Storage**: Consider moving old backups off-system

## Safety Features

- Preserves running containers and their images
- Keeps recent backups (30 days)
- Interactive prompt for aggressive cleanup
- Detailed logging of all operations
- Creates disk usage reports

## Recommendations

1. **Log Rotation**: Implement log rotation for all KnowledgeHub services
2. **Backup Storage**: Move older backups to network storage
3. **Docker Registry**: Use a registry for base images to avoid re-downloads
4. **Monitoring**: Set up alerts for disk usage >80%

## Troubleshooting

### If cleanup fails:
1. Check permissions: `ls -la /opt/projects/knowledgehub/scripts/`
2. Run manually with verbose: `bash -x /opt/projects/knowledgehub/scripts/optimize-disk-usage.sh`
3. Check logs: `tail -100 /opt/projects/knowledgehub/logs/cleanup.log`

### If space fills up quickly:
1. Check Docker logs: `du -h /var/lib/docker/containers/*/`
2. Check for large files: `find /opt -type f -size +1G`
3. Review running containers: `docker ps --size`

## Files Created/Modified

```
/opt/projects/knowledgehub/
├── scripts/
│   ├── optimize-disk-usage.sh    # Main optimization script
│   └── cleanup-cron.sh           # Cron wrapper script
├── logs/
│   └── cleanup.log               # Cleanup operation logs
└── disk_usage_report_*.txt       # Detailed usage reports
```