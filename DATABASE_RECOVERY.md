# KnowledgeHub Database Connection Recovery & Retry Logic

## Overview

The KnowledgeHub Database Recovery System provides comprehensive resilience for all database operations through automatic connection recovery, intelligent retry mechanisms, circuit breaker patterns, and performance optimization. Built on top of SQLAlchemy, asyncpg, and psycopg2, it ensures maximum database availability and reliability.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │────│ Database Recovery│────│   Connection    │
│    Requests     │    │     Manager      │    │     Pools       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │ Retry Logic &   │              │
         │              │ Circuit Breaker │              │
         │              └─────────────────┘              │
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────│ Health Monitoring│──────────────┘
                        │ & Metrics       │
                        └─────────────────┘
```

## Key Components

### 1. Database Recovery Manager

**Location**: `api/services/database_recovery.py`

Provides comprehensive database resilience:

#### Connection Types
- **SQLAlchemy Async Engine**: For ORM-based async operations
- **SQLAlchemy Sync Engine**: For ORM-based sync operations
- **Asyncpg Pool**: For high-performance async PostgreSQL operations
- **Psycopg2 Pool**: For sync PostgreSQL operations with threading support

#### Core Features
- **Automatic Connection Recovery**: Detects failures and recovers connections
- **Connection Pooling**: Efficient connection management with monitoring
- **Circuit Breaker Pattern**: Prevents cascading failures
- **Retry Strategies**: Multiple backoff strategies for different scenarios
- **Performance Tracking**: Query metrics and connection statistics
- **Health Monitoring**: Continuous health checks with failure tracking

### 2. Retry Strategies

#### Exponential Backoff (Default)
```python
RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=60.0,
    backoff_multiplier=2.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF
)
```

#### Linear Backoff
```python
RetryConfig(
    strategy=RetryStrategy.LINEAR_BACKOFF,
    initial_delay=2.0
)
```

#### Fixed Delay
```python
RetryConfig(
    strategy=RetryStrategy.FIXED_DELAY,
    initial_delay=5.0
)
```

#### Custom Retry Configuration
```python
@with_db_retry(retry_config=RetryConfig(
    max_retries=5,
    initial_delay=2.0,
    max_delay=120.0,
    timeout=60.0
))
async def critical_operation():
    # Your database operation
    pass
```

### 3. Circuit Breaker Pattern

Prevents database overload during failures:

- **Failure Threshold**: 5 consecutive failures (configurable)
- **Open Duration**: 60 seconds before retry
- **Automatic Reset**: After successful operation
- **Graceful Degradation**: Read-only mode available

### 4. Connection Pool Management

#### Pool Configuration
- **Min Connections**: 5 per pool
- **Max Connections**: 20 per pool (configurable)
- **Connection Timeout**: 10 seconds
- **Statement Timeout**: 30 seconds
- **Pool Recycle**: 3600 seconds (1 hour)

#### Pool Monitoring
- Active/idle connection tracking
- Connection wait time metrics
- Pool utilization percentage
- Automatic cleanup on errors

## API Endpoints

### Database Recovery Management

**Base URL**: `/api/database-recovery`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Database connection health status |
| `/status` | GET | Comprehensive connection status |
| `/connections` | GET | Detailed connection pool information |
| `/recover` | POST | Trigger database connection recovery |
| `/test-connection` | POST | Test specific connection types |
| `/retry-config` | PUT | Update retry configuration |
| `/circuit-breaker/reset` | POST | Reset circuit breaker state |
| `/metrics` | GET | Database performance metrics |
| `/alerts` | GET | Active database alerts |

### Usage Examples

```bash
# Check database health
curl http://localhost:3000/api/database-recovery/health

# Get connection details
curl http://localhost:3000/api/database-recovery/connections

# Test all connections
curl -X POST http://localhost:3000/api/database-recovery/test-connection \
  -H "Content-Type: application/json" \
  -d '{"connection_type": "all", "test_query": "SELECT 1"}'

# Trigger recovery
curl -X POST http://localhost:3000/api/database-recovery/recover

# Update retry configuration
curl -X PUT http://localhost:3000/api/database-recovery/retry-config \
  -H "Content-Type: application/json" \
  -d '{
    "max_retries": 5,
    "initial_delay": 2.0,
    "max_delay": 120.0,
    "strategy": "exponential_backoff"
  }'

# Get performance metrics
curl "http://localhost:3000/api/database-recovery/metrics?time_window_minutes=10"
```

## Middleware Integration

### Database Recovery Middleware

Automatically handles database failures at the request level:

```python
app.add_middleware(
    DatabaseRecoveryMiddleware,
    enable_circuit_breaker=True,
    read_only_fallback=True
)
```

**Features**:
- Automatic circuit breaker checks
- Health monitoring per request
- Graceful degradation for read-only operations
- Background recovery triggers

### Connection Pool Middleware

Tracks connection usage and ensures proper cleanup:

```python
app.add_middleware(DatabaseConnectionPoolMiddleware)
```

**Features**:
- Connection usage tracking
- Automatic cleanup on errors
- Pool utilization warnings
- Request-level metrics

## Resilient Database Service

### High-Level Interface

**Location**: `api/services/resilient_database.py`

Provides simplified database operations with built-in resilience:

```python
from api.services.resilient_database import resilient_db

# Create memory with automatic retry
memory = await resilient_db.create_memory(memory_data)

# Search with optimization
results = await resilient_db.search_memories(
    user_id="user123",
    query="machine learning",
    limit=10
)

# Batch operations
memories = await resilient_db.batch_create_memories(memories_data)

# Complex transactions
async with resilient_db.transaction() as session:
    # Multiple operations in single transaction
    await create_document(session, doc_data)
    await create_chunks(session, chunks_data)
```

### Key Features
- **Automatic Retry**: All operations have built-in retry logic
- **Query Optimization**: Intelligent query planning and caching
- **Batch Processing**: Efficient bulk operations
- **Transaction Management**: ACID compliance with rollback
- **Performance Monitoring**: Operation timing and metrics

## Performance Optimization

### Query Optimization
- **Index Usage**: Automatic index hints for common queries
- **Query Caching**: 5-minute TTL for repeated queries
- **Batch Processing**: Operations grouped for efficiency
- **Connection Reuse**: Minimized connection overhead

### Best Practices
1. **Use Batch Operations**: For multiple inserts/updates
2. **Leverage Transactions**: Group related operations
3. **Monitor Pool Usage**: Keep utilization below 80%
4. **Configure Appropriate Timeouts**: Based on operation complexity

## Health Monitoring

### Health Check Response
```json
{
  "state": "connected",
  "last_check": "2025-01-20T10:30:00Z",
  "consecutive_failures": 0,
  "connections": {
    "total": 20,
    "active": 5,
    "idle": 15,
    "wait_time_ms": 2.5
  },
  "performance": {
    "success_rate": 99.5,
    "avg_execution_time": 12.3
  },
  "circuit_breaker": {
    "open": false,
    "failures": 0,
    "threshold": 5
  }
}
```

### Alert Types
- **Connection Failures**: Consecutive connection failures detected
- **Circuit Breaker Open**: Database circuit breaker activated
- **Pool Saturation**: Connection pool > 80% utilized
- **Performance Degradation**: Success rate < 95%

## Error Handling

### Retryable Errors
- `OperationalError`: Database operation failures
- `DisconnectionError`: Connection lost
- `TimeoutError`: Query timeout
- `PostgresConnectionError`: PostgreSQL connection issues
- `psycopg2.OperationalError`: Psycopg2 operation failures

### Non-Retryable Errors
- `IntegrityError`: Constraint violations
- `DataError`: Invalid data types
- `ProgrammingError`: SQL syntax errors
- `NotSupportedError`: Unsupported operations

## Configuration

### Environment Variables
```bash
# Database connection
DB_HOST=localhost
DB_PORT=5433
DB_USER=knowledgehub
DB_PASS=knowledgehub
DB_NAME=knowledgehub

# Connection pool
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600

# Recovery settings
RECOVERY_MAX_RETRIES=3
RECOVERY_BACKOFF_MULTIPLIER=2.0
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=60
```

### Connection String Options
```python
# Async connection with SSL
postgresql+asyncpg://user:pass@host:port/db?ssl=require

# Sync connection with application name
postgresql://user:pass@host:port/db?application_name=knowledgehub

# Connection with custom timeout
postgresql://user:pass@host:port/db?connect_timeout=10
```

## Troubleshooting

### Common Issues

1. **Circuit Breaker Open**:
   ```bash
   # Reset circuit breaker
   curl -X POST http://localhost:3000/api/database-recovery/circuit-breaker/reset
   
   # Check current state
   curl http://localhost:3000/api/database-recovery/status
   ```

2. **Connection Pool Exhausted**:
   ```bash
   # Check pool utilization
   curl http://localhost:3000/api/database-recovery/connections
   
   # Increase pool size
   export DB_POOL_SIZE=30
   export DB_MAX_OVERFLOW=20
   ```

3. **Slow Queries**:
   ```bash
   # Check query performance
   curl http://localhost:3000/api/database-recovery/metrics?time_window_minutes=5
   
   # Enable query logging
   export DB_ECHO=true
   ```

4. **Recovery Failures**:
   ```bash
   # Check recovery status
   curl http://localhost:3000/api/database-recovery/health
   
   # Manually trigger recovery
   curl -X POST http://localhost:3000/api/database-recovery/recover
   ```

### Debug Commands

```bash
# Test specific connection type
curl -X POST http://localhost:3000/api/database-recovery/test-connection \
  -H "Content-Type: application/json" \
  -d '{"connection_type": "asyncpg", "test_query": "SELECT version()"}'

# Get active alerts
curl http://localhost:3000/api/database-recovery/alerts

# View connection metrics
curl http://localhost:3000/api/database-recovery/metrics | jq '.connection_metrics'

# Check retry configuration
curl http://localhost:3000/api/database-recovery/status | jq '.retry_config'
```

## Performance Metrics

### Key Metrics
- **Success Rate**: Percentage of successful database operations
- **Average Query Time**: Mean execution time for queries
- **Pool Utilization**: Active connections / total connections
- **Circuit Breaker Trips**: Number of circuit breaker activations
- **Recovery Time**: Mean time to recover failed connections

### Monitoring Integration

#### Prometheus Metrics
```
# Connection pool metrics
db_connections_active{pool="async_engine"}
db_connections_idle{pool="asyncpg"}
db_connections_wait_time_ms{pool="sync_engine"}

# Query performance metrics
db_query_duration_seconds{operation="select", status="success"}
db_query_total{operation="insert", status="error"}

# Circuit breaker metrics
db_circuit_breaker_state{state="open"}
db_circuit_breaker_failures_total
```

#### Grafana Dashboard
Pre-configured panels for:
- Connection pool status
- Query performance trends
- Error rates and recovery
- Circuit breaker state
- Alert notifications

## Best Practices

### 1. Connection Management
- Use connection pooling for all operations
- Set appropriate pool sizes based on load
- Monitor pool utilization regularly
- Implement connection timeouts

### 2. Error Handling
- Use the retry decorator for critical operations
- Configure appropriate retry strategies
- Log all database errors with context
- Monitor error patterns for optimization

### 3. Performance
- Use batch operations for bulk data
- Leverage query optimization features
- Monitor query execution times
- Implement appropriate indexes

### 4. Resilience
- Configure circuit breaker thresholds
- Set up health check monitoring
- Implement graceful degradation
- Plan for recovery scenarios

This comprehensive database recovery system ensures KnowledgeHub maintains reliable database connectivity with automatic recovery, intelligent retry logic, and performance optimization for production environments.