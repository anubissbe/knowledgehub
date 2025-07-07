# KnowledgeHub Optional Services Deployment

## Overview
This document details the deployment of optional services that were built but not initially running in the KnowledgeHub infrastructure.

## Date: July 5, 2025

## Services Deployed

### 1. Monitoring Stack
**File**: `docker-compose.monitoring.yml`
**Services**:
- Prometheus (port 9090) - Metrics collection
- Grafana (port 3030) - Visualization dashboards
- AlertManager (port 9093) - Alert management
- Node Exporter (port 9100) - System metrics
- cAdvisor (port 8081) - Container metrics

**Deployment Command**:
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

**Configuration**:
- Prometheus scrapes all KnowledgeHub services
- Grafana pre-configured with datasources
- AlertManager rules for system monitoring
- 30-day metrics retention

### 2. AI Service
**File**: `docker-compose.ai.yml`
**Port**: 8002 (changed from 8001 due to conflict)
**Features**:
- Threat analysis and risk scoring
- Content similarity search
- Embedding generation (GPU-accelerated)
- FastAPI-based REST API

**Deployment Command**:
```bash
docker-compose -f docker-compose.ai.yml up -d
```

**Endpoints**:
- `/api/ai/analyze-threats` - Security threat detection
- `/api/ai/content-similarity` - Semantic search
- `/api/ai/risk-scoring` - Component risk assessment
- `/health` - Service health check

### 3. MCP Server Integration
**Added to**: Main `docker-compose.yml`
**Port**: 3008 (changed from 3002 due to conflict)
**Features**:
- WebSocket-based Model Context Protocol server
- Knowledge base search and storage
- Resource management
- Tool execution

**Configuration Changes**:
```yaml
mcp-server:
  build:
    context: .
    dockerfile: docker/mcp.Dockerfile
  container_name: knowledgehub-mcp
  ports:
    - "0.0.0.0:3008:3002"
  healthcheck:
    test: ["CMD", "test", "-f", "/tmp/mcp_healthy"]
```

## Advanced Analytics Dashboard

### Components Created
**Location**: `/opt/projects/knowledgehub/src/web-ui/src/components/analytics/`

1. **AdvancedAnalytics.tsx**
   - Main analytics dashboard component
   - Integrates all analytics widgets

2. **ChunkAnalysis.tsx**
   - Chunk size distribution histogram
   - Token count analysis
   - Quality metrics visualization

3. **SourcePerformance.tsx**
   - Source crawl performance metrics
   - Success/failure rates
   - Documents per source

4. **UserActivity.tsx**
   - User search activity tracking
   - Popular search terms
   - Usage patterns

5. **SystemMetrics.tsx**
   - Queue status monitoring
   - Processing rates
   - System health indicators

## Documentation Updates

### Files Updated:
1. **README.md**
   - Updated service count from 11 to 16
   - Added monitoring stack details
   - Added AI service information
   - Corrected port mappings

2. **docs/architecture.md**
   - Fixed port discrepancies
   - Updated service descriptions
   - Added optional services section

3. **CLAUDE.md**
   - Added monitoring commands
   - Updated service list
   - Added health check instructions

## Verification Steps

1. **Check Monitoring Stack**:
   ```bash
   curl http://localhost:9090/-/healthy  # Prometheus
   curl http://localhost:3030/api/health # Grafana
   ```

2. **Check AI Service**:
   ```bash
   curl http://localhost:8002/health
   ```

3. **Check MCP Server**:
   ```bash
   docker logs knowledgehub-mcp
   ```

4. **View Analytics Dashboard**:
   Navigate to: http://localhost:5173/analytics

## Total Service Count

**Before**: 8 core services running
**After**: 16 services running (including optional services)

- Core Services: 8
- Monitoring Stack: 5
- AI Service: 1
- MCP Server: 1
- Web UI: 1

**Total**: 16 containerized services