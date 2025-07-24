# KnowledgeHub Distributed Tracing System

## Overview

The KnowledgeHub Distributed Tracing System provides comprehensive observability for request flows, performance analysis, and debugging capabilities across all services. Built on OpenTelemetry standards, it offers real-time insights into system behavior and performance bottlenecks.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │────│ OpenTelemetry   │────│     Jaeger      │
│   (Instrumented)│    │   Collector     │    │      UI         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │     Tempo       │              │
         │              │   (Storage)     │              │
         │              └─────────────────┘              │
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────│ Grafana Tracing │──────────────┘
                        │  (Visualization)│
                        └─────────────────┘
```

## Key Components

### 1. OpenTelemetry Instrumentation

**Location**: `api/services/opentelemetry_tracing.py`

Provides comprehensive automatic and manual instrumentation:

#### Automatic Instrumentation
- **FastAPI**: HTTP request/response tracing
- **SQLAlchemy**: Database query tracing
- **Redis**: Cache operation tracing
- **HTTP Clients**: External service call tracing
- **AsyncPG**: PostgreSQL async operation tracing

#### Custom Instrumentation
- **Memory Operations**: Semantic search performance (target: <50ms)
- **AI Operations**: Model inference and embedding generation
- **Business Logic**: Custom span creation with detailed attributes

#### Key Features
- **Distributed Context Propagation**: Trace correlation across services
- **Performance Analysis**: Automated performance grading and bottleneck detection
- **Error Tracking**: Exception capture and error correlation
- **Custom Attributes**: Rich metadata for debugging and analysis

### 2. Tracing Middleware

**Location**: `api/middleware/tracing_middleware.py`

Automatic tracing for all HTTP requests with:
- **Request Metadata**: Method, URL, headers, query parameters
- **Performance Classification**: Fast (<200ms), Medium (<1s), Slow (>1s)
- **Error Detection**: HTTP status code analysis and exception tracking
- **Trace Headers**: X-Trace-ID and X-Span-ID for correlation

### 3. Traced Services

#### Memory Service Tracing
**Location**: `api/services/traced_memory_service.py`

Comprehensive tracing for memory operations:
- **Search Operations**: Target <50ms with detailed performance analysis
- **Embedding Generation**: AI model performance tracking
- **Vector Search**: Database query optimization insights
- **Result Enrichment**: Data processing performance

#### Database Operation Tracing
Automatic tracking of:
- **Query Performance**: Execution time and optimization opportunities
- **Connection Pool**: Usage patterns and bottlenecks
- **Transaction Analysis**: Multi-operation performance

#### AI Operation Tracing
Detailed tracking of:
- **Model Inference**: Processing time and throughput
- **Embedding Generation**: Input/output token analysis
- **Performance Classification**: Fast/Normal/Slow categorization

### 4. Tracing Infrastructure

#### Jaeger
- **UI Access**: http://localhost:16686
- **Service Maps**: Visual service dependency mapping
- **Trace Search**: Advanced filtering and analysis
- **Performance Analysis**: Statistical views and trends

#### OpenTelemetry Collector
- **Multi-Protocol Support**: OTLP, Jaeger, Zipkin
- **Data Processing**: Filtering, sampling, and enrichment
- **Multiple Exporters**: Send to Jaeger, Tempo, and custom backends

#### Tempo
- **High-Scale Storage**: Efficient trace storage and querying
- **TraceQL**: Advanced query language for trace analysis
- **Grafana Integration**: Seamless visualization integration

#### Grafana Tracing Dashboards
- **Service Overview**: Request rates, latency, and error rates
- **Performance Analysis**: Operation-specific performance metrics
- **Bottleneck Identification**: Automated slow operation detection

## Performance Targets & Monitoring

### Target Performance Metrics

| Operation Type | Target Latency | Alert Threshold | Critical Threshold |
|---------------|----------------|-----------------|-------------------|
| Memory Search | <50ms (95th) | >50ms | >100ms |
| API Requests | <200ms (95th) | >1s | >2s |
| Database Queries | <100ms (95th) | >200ms | >500ms |
| AI Operations | <2s (95th) | >5s | >10s |
| Session Restoration | <1s | >2s | >5s |

### Automatic Performance Analysis

The system automatically:
1. **Grades Operations**: A-F performance grades based on latency
2. **Identifies Bottlenecks**: Operations exceeding performance targets
3. **Calculates Impact Scores**: Prioritizes optimization efforts
4. **Provides Recommendations**: Specific optimization suggestions

## API Endpoints

### Tracing Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tracing/status` | GET | Tracing system status and configuration |
| `/api/tracing/performance/summary` | GET | Comprehensive performance analysis |
| `/api/tracing/performance/bottlenecks` | GET | Identify performance bottlenecks |
| `/api/tracing/operations/{name}/analysis` | GET | Detailed operation performance analysis |
| `/api/tracing/traces/search` | GET | Search traces with filtering |
| `/api/tracing/health` | GET | Tracing system health check |

### Usage Examples

```bash
# Get overall tracing status
curl http://localhost:3000/api/tracing/status

# Identify performance bottlenecks
curl "http://localhost:3000/api/tracing/performance/bottlenecks?threshold_ms=100"

# Analyze memory operation performance
curl http://localhost:3000/api/tracing/operations/memory.search/analysis

# Search for slow traces
curl "http://localhost:3000/api/tracing/traces/search?min_duration_ms=1000"
```

## Development Integration

### Adding Custom Tracing

```python
from api.services.opentelemetry_tracing import otel_tracing

# Custom span creation
with otel_tracing.start_span("custom_operation", attributes={"key": "value"}) as span:
    # Your operation code
    result = perform_operation()
    span.set_attribute("result_count", len(result))
    return result

# Decorator-based tracing
@otel_tracing.trace_memory_operation("search", user_id, "semantic")
async def search_memories(query: str, user_id: str):
    # Automatically traced memory operation
    return await perform_search(query)
```

### Convenience Decorators

```python
from api.middleware.tracing_middleware import (
    trace_db_query, trace_ai_embedding, trace_memory_search, 
    trace_http_call, trace_redis_op
)

@trace_db_query("get_user", "users")
async def get_user(user_id: str):
    # Automatically traced database operation
    pass

@trace_ai_embedding("sentence-transformers")
async def generate_embedding(text: str):
    # Automatically traced AI operation
    pass

@trace_http_call("external_api", "https://api.example.com", "GET")
async def call_external_service():
    # Automatically traced external call
    pass
```

## Deployment

### Quick Start

1. **Start Tracing Stack**:
   ```bash
   cd /opt/projects/knowledgehub
   ./scripts/start-tracing.sh
   ```

2. **Access Interfaces**:
   - Jaeger UI: http://localhost:16686
   - Grafana Tracing: http://localhost:3031 (admin/admin123)
   - OpenTelemetry Collector: http://localhost:13133

3. **View Traces**:
   The application automatically sends traces when running. Access Jaeger UI and select "knowledgehub-api" service to view traces.

### Docker Services

| Service | Purpose | Port | Health Check |
|---------|---------|------|--------------|
| Jaeger | Trace visualization | 16686 | http://localhost:16686 |
| OTEL Collector | Trace processing | 13133 | http://localhost:13133 |
| Tempo | Trace storage | 3200 | http://localhost:3200/ready |
| Zipkin | Alternative UI | 9411 | http://localhost:9411/health |
| Grafana | Dashboards | 3031 | http://localhost:3031/api/health |

### Configuration Files

| File | Purpose |
|------|---------|
| `otel-collector/otel-collector.yml` | Collector configuration |
| `tempo/tempo.yml` | Tempo storage configuration |
| `grafana-tracing/provisioning/` | Dashboard and datasource setup |
| `docker-compose.tracing.yml` | Service orchestration |

## Advanced Features

### Performance Analysis

#### Automatic Bottleneck Detection
The system automatically identifies operations that:
- Exceed performance targets (95th percentile)
- Have high frequency of slow operations
- Impact overall system performance

#### Performance Grading
Operations are automatically graded A-F based on:
- **Memory Operations**: A(<50ms), B(<100ms), C(<200ms), D(<500ms), F(>500ms)
- **Database Operations**: A(<100ms), B(<250ms), C(<500ms), D(<1s), F(>1s)
- **AI Operations**: A(<2s), B(<5s), C(<10s), D(<20s), F(>20s)
- **API Operations**: A(<200ms), B(<500ms), C(<1s), D(<2s), F(>2s)

#### Impact Scoring
Bottlenecks are prioritized using:
```
Impact Score = Duration Impact × Frequency Impact × Reliability Impact
```

### Trace Analysis

#### Service Dependency Mapping
Automatic generation of service dependency graphs showing:
- Call relationships between services
- Performance characteristics of each connection
- Error rates and retry patterns

#### Critical Path Analysis
Identification of:
- Longest operations in request traces
- Performance bottlenecks in critical user flows
- Optimization opportunities with highest impact

### Error Correlation

#### Exception Tracking
Automatic capture of:
- Exception types and messages
- Stack traces and error context
- Correlation with performance degradation

#### Error Pattern Analysis
Detection of:
- Recurring error patterns
- Error clustering by operation or time
- Performance impact of error conditions

## Troubleshooting

### Common Issues

1. **Traces Not Appearing**:
   ```bash
   # Check tracing status
   curl http://localhost:3000/api/tracing/status
   
   # Verify Jaeger connectivity
   curl http://localhost:16686/api/services
   
   # Check collector health
   curl http://localhost:13133/
   ```

2. **Performance Analysis Missing**:
   - Ensure application is generating sufficient traffic
   - Check performance summary endpoint
   - Verify trace sampling configuration

3. **High Memory Usage**:
   - Review trace retention settings
   - Adjust sampling rates for high-volume environments
   - Monitor collector memory limits

### Debugging Commands

```bash
# View tracing service logs
docker-compose -f docker-compose.tracing.yml logs -f jaeger
docker-compose -f docker-compose.tracing.yml logs -f otel-collector

# Check service health
./scripts/start-tracing.sh  # Includes health checks

# Test trace generation
curl -X POST http://localhost:3000/api/memories/ \
  -H "Content-Type: application/json" \
  -d '{"content": "test memory", "user_id": "test"}'
```

## Security Considerations

1. **Sensitive Data**: Automatic filtering of authorization headers and sensitive attributes
2. **Access Control**: Restrict access to tracing interfaces in production
3. **Data Retention**: Configure appropriate trace retention policies
4. **Network Security**: Secure collector endpoints and inter-service communication
5. **Sampling**: Implement appropriate sampling to manage data volume and costs

## Performance Impact

### Overhead Analysis
- **CPU Overhead**: <2% average CPU increase
- **Memory Overhead**: ~50MB for trace buffers
- **Network Overhead**: <1% additional network traffic
- **Latency Impact**: <1ms per traced operation

### Optimization Settings
```yaml
# Production sampling configuration
sampling_ratio: 0.1  # 10% sampling for high-volume environments
batch_size: 1024     # Optimize network efficiency
export_timeout: 30s  # Balance responsiveness and reliability
```

## Monitoring Integration

### Prometheus Metrics
The tracing system exports metrics to Prometheus:
- `otel_collector_spans_received_total`
- `otel_collector_spans_exported_total`
- `jaeger_traces_received_total`
- `tempo_ingester_traces_created_total`

### Grafana Integration
Pre-configured dashboards provide:
- **Service Performance Overview**: Request rates, latencies, error rates
- **Trace Analysis**: Detailed trace breakdowns and dependencies
- **Performance Trends**: Historical performance analysis
- **Bottleneck Detection**: Automated identification of slow operations

## Best Practices

### Span Design
1. **Meaningful Names**: Use descriptive span names (e.g., "memory.semantic_search")
2. **Appropriate Granularity**: Balance detail with overhead
3. **Rich Attributes**: Add contextual information for debugging
4. **Error Handling**: Always record exceptions and error states

### Performance Optimization
1. **Sampling Strategy**: Implement intelligent sampling for high-volume services
2. **Batch Processing**: Use batch span processors for efficiency
3. **Async Export**: Non-blocking trace export to minimize latency impact
4. **Resource Limits**: Configure memory and CPU limits for stability

### Production Deployment
1. **Monitoring**: Monitor tracing system health and performance
2. **Alerting**: Set up alerts for trace processing failures
3. **Backup**: Implement trace data backup strategies
4. **Scaling**: Plan for horizontal scaling of tracing infrastructure

This comprehensive distributed tracing system ensures full visibility into KnowledgeHub's performance characteristics, enabling proactive optimization and rapid issue resolution in production environments.