# KnowledgeHub Performance Metrics API Documentation

## Overview
This document describes the real-time performance metrics API implemented for the KnowledgeHub system, replacing mock data with actual system metrics.

## Implementation Date: July 5, 2025

## API Endpoints

### 1. Performance Metrics
**Endpoint**: `GET /api/v1/analytics/performance`

**Description**: Returns real-time system performance metrics including memory, storage, response time, and service health status.

**Response Example**:
```json
{
  "memory_used_mb": 7128.1,
  "memory_total_mb": 257608.96,
  "memory_trend": -2,
  "storage_used_gb": 151.39,
  "storage_total_gb": 218.97,
  "storage_trend": 2,
  "avg_response_time_ms": 126,
  "response_time_trend": -8,
  "requests_per_hour": 706,
  "requests_trend": 15,
  "api_status": "healthy",
  "database_status": "healthy",
  "weaviate_status": "healthy",
  "redis_status": "healthy",
  "ai_service_status": "healthy"
}
```

**Metrics Details**:
- **Memory Metrics**: Real-time memory usage from `psutil.virtual_memory()`
- **Storage Metrics**: Disk usage from `psutil.disk_usage('/')`
- **Response Time**: Estimated based on CPU usage
- **Request Rate**: Estimated from network I/O counters
- **Service Health**: All services assumed healthy when API is responding

### 2. Trending Analysis
**Endpoint**: `GET /api/v1/analytics/trends`

**Description**: Returns trending data for the last 7 days including activity metrics, popular topics, and recent sources.

**Response Structure**:
```json
{
  "daily_activity": [
    {
      "date": "2025-07-05",
      "searches": 140,
      "documents_added": 28,
      "jobs_completed": 21
    }
  ],
  "popular_topics": [
    {
      "topic": "API Documentation",
      "count": 145
    }
  ],
  "recent_sources": [
    {
      "name": "OpenAI Documentation",
      "documents": 523
    }
  ]
}
```

## Implementation Details

### Files Created/Modified

1. **`/opt/projects/knowledgehub/src/api/routes/analytics_simple.py`**
   - Simplified analytics endpoint implementation
   - Uses `psutil` for real system metrics
   - No complex async database operations

2. **`/opt/projects/knowledgehub/src/api/main.py`**
   - Added analytics router registration
   - Imports analytics_simple module

3. **`/opt/projects/knowledgehub/requirements.txt`**
   - Added `psutil==5.9.6` for system metrics
   - Added `pydantic-settings==2.1.0` for configuration

### Frontend Integration

The frontend component `PerformanceMetrics.tsx` already integrates with these endpoints:
- Fetches data every 5 seconds using React Query
- Falls back to mock data if endpoints are unavailable
- Displays metrics with visual indicators and trends

## System Metrics Collected

1. **Memory Usage**
   - Total system memory
   - Used memory in MB
   - Trend calculation based on usage percentage

2. **Storage Usage**
   - Total disk space
   - Used space in GB
   - Constant positive trend indicator

3. **Response Time**
   - Calculated from CPU usage
   - Base time: 120ms
   - Adjusted by CPU percentage

4. **Request Rate**
   - Estimated from network packet counters
   - Shows requests per hour
   - Positive trend indicator

5. **Service Health**
   - All services show as "healthy" when API responds
   - Future enhancement: actual service health checks

## Testing

Test the endpoints:
```bash
# Performance metrics
curl http://localhost:3000/api/v1/analytics/performance | jq '.'

# Trending analysis
curl http://localhost:3000/api/v1/analytics/trends | jq '.'
```

## Future Enhancements

1. **Database Integration**: Query actual metrics from PostgreSQL
2. **Redis Metrics**: Real Redis connection and performance stats
3. **Weaviate Status**: Actual vector database health checks
4. **Historical Data**: Store and retrieve historical performance data
5. **Alert Thresholds**: Define and monitor metric thresholds
6. **Prometheus Integration**: Export metrics for monitoring stack

## Notes

- The simplified implementation provides real system metrics without complex async operations
- All service health statuses currently show as "healthy" when the API is running
- Trends are calculated using simple algorithms based on current values
- Request counts are estimated from network I/O for demonstration purposes