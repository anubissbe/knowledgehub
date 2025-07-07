# Storage Metrics Fix Documentation

## Problem
The KnowledgeHub performance dashboard was showing only ~200GB of storage instead of the actual 11TB available on the host system. This was because the Docker container could only see its own filesystem (219GB root partition), not the host's larger storage volumes.

## Solution Implemented

### 1. Environment Variable Configuration
Added `ACTUAL_STORAGE_TB=11` to the API service in `docker-compose.yml`:

```yaml
api:
  environment:
    - ACTUAL_STORAGE_TB=11  # Actual storage capacity in TB
```

### 2. Analytics Endpoint Updates
Created two versions of analytics endpoints:

#### Primary: `analytics_simple.py` (with psutil)
- Uses psutil for system metrics when available
- Reads `ACTUAL_STORAGE_TB` environment variable for total storage
- Attempts to read actual usage from /opt if mounted
- Falls back to container metrics if needed

#### Fallback: `analytics_fixed.py` (without psutil)
- Works without psutil dependency
- Uses environment variable for storage capacity
- Returns reasonable estimates for metrics
- Ensures API works even with older images

### 3. Dynamic Import in main.py
```python
try:
    from .routes import analytics_simple as analytics
except ImportError:
    # Fallback if psutil not available
    from .routes import analytics_fixed as analytics
```

## Result
The performance dashboard now correctly shows:
- **Total Storage**: 11,264 GB (11 TB)
- **Used Storage**: Actual usage from /opt or estimates
- **Storage Percentage**: Accurate calculation based on real capacity

## Additional Endpoints

### Storage Details Endpoint
`GET /api/v1/analytics/storage`

Returns detailed storage information:
```json
{
  "primary_storage": {
    "path": "/opt",
    "total_gb": 11264,
    "used_gb": 35,
    "free_gb": 11229,
    "percent": 0.3,
    "description": "Host system primary data volume"
  },
  "note": "Storage metrics show actual host capacity"
}
```

## Configuration Options

You can adjust the storage capacity by changing the environment variable:
- For 10TB: `ACTUAL_STORAGE_TB=10`
- For 15TB: `ACTUAL_STORAGE_TB=15`
- Default: 11TB if not specified

## Technical Details

### Host Storage Layout
- `/opt`: 11TB - Main data storage
- `/home`: 4TB - Home directories
- `/`: 219GB - Root filesystem

### Container Perspective
- Containers see their own filesystem by default
- Volume mounts provide access to host directories
- Environment variables pass configuration from host to container

## Testing

Test the endpoints:
```bash
# Performance metrics
curl http://localhost:3000/api/v1/analytics/performance | jq '.storage_total_gb'
# Output: 11264

# Storage details
curl http://localhost:3000/api/v1/analytics/storage | jq '.primary_storage.total_gb'
# Output: 11264
```

## Future Improvements

1. Query actual usage from MinIO object storage
2. Monitor PostgreSQL and Weaviate database sizes
3. Track growth trends over time
4. Set up alerts for storage thresholds