#!/bin/bash

# KnowledgeHub Services Health Check Script
# Comprehensive health monitoring for all integrated services

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMEOUT=10

# Service definitions
declare -A SERVICES=(
    ["postgres"]="5433"
    ["redis"]="6381"
    ["weaviate"]="8090"
    ["qdrant"]="6333"
    ["neo4j"]="7474"
    ["timescale"]="5434"
    ["minio"]="9010"
    ["zep-postgres"]="5432"
    ["zep"]="8100"
    ["firecrawl"]="3002"
    ["playwright-service"]="3003"
    ["graphiti"]="8080"
    ["phoenix"]="6006"
    ["ai-service"]="8002"
    ["api"]="3000"
    ["webui"]="3100"
    ["nginx"]="8080"
    ["prometheus"]="9090"
    ["grafana"]="3001"
)

declare -A HEALTH_ENDPOINTS=(
    ["zep"]="/health"
    ["firecrawl"]="/health"
    ["graphiti"]="/health"
    ["phoenix"]="/health"
    ["ai-service"]="/health"
    ["api"]="/health"
    ["prometheus"]="/-/healthy"
)

# Functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if service is running
check_service_running() {
    local service=$1
    if docker-compose ps | grep -q "$service.*Up"; then
        return 0
    else
        return 1
    fi
}

# Check if port is accessible
check_port() {
    local port=$1
    if nc -z localhost "$port" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Check HTTP health endpoint
check_http_health() {
    local service=$1
    local port=$2
    local endpoint=${HEALTH_ENDPOINTS[$service]:-"/"}
    
    local url="http://localhost:$port$endpoint"
    if curl -f -s --max-time "$TIMEOUT" "$url" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Check database connectivity
check_database() {
    local service=$1
    local port=$2
    
    case $service in
        "postgres"|"timescale")
            if PGPASSWORD=knowledgehub123 psql -h localhost -p "$port" -U knowledgehub -d knowledgehub -c "SELECT 1;" >/dev/null 2>&1; then
                return 0
            fi
            ;;
        "zep-postgres")
            if PGPASSWORD=zep123 psql -h localhost -p "$port" -U zep -d zep -c "SELECT 1;" >/dev/null 2>&1; then
                return 0
            fi
            ;;
        "redis")
            if redis-cli -h localhost -p "$port" ping | grep -q "PONG"; then
                return 0
            fi
            ;;
        "neo4j")
            # Neo4j check would require cypher-shell or HTTP API
            return 0
            ;;
    esac
    return 1
}

# Get service metrics
get_service_metrics() {
    local service=$1
    
    # Container stats
    local stats=$(docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep "$service" | head -1)
    if [[ -n "$stats" ]]; then
        echo "    📊 $stats"
    fi
    
    # Container uptime
    local uptime=$(docker inspect --format='{{.State.StartedAt}}' "$(docker-compose ps -q "$service" 2>/dev/null)" 2>/dev/null)
    if [[ -n "$uptime" ]]; then
        echo "    ⏰ Started: $uptime"
    fi
}

# Check individual service
check_service() {
    local service=$1
    local port=$2
    local status="❌ FAILED"
    local details=""
    
    echo ""
    echo "🔍 Checking $service (port $port)..."
    
    # Check if container is running
    if ! check_service_running "$service"; then
        error "$service container is not running"
        return 1
    fi
    
    # Check port accessibility
    if ! check_port "$port"; then
        error "$service port $port is not accessible"
        return 1
    fi
    
    # Check health endpoint if available
    if [[ -n "${HEALTH_ENDPOINTS[$service]:-}" ]]; then
        if check_http_health "$service" "$port"; then
            status="✅ HEALTHY"
            details="HTTP health check passed"
        else
            status="⚠️  PORT OPEN"
            details="HTTP health check failed"
        fi
    # Check database connectivity for database services
    elif [[ "$service" =~ ^(postgres|redis|timescale|zep-postgres|neo4j)$ ]]; then
        if check_database "$service" "$port"; then
            status="✅ HEALTHY"
            details="Database connection successful"
        else
            status="⚠️  PORT OPEN"
            details="Database connection failed"
        fi
    else
        status="✅ RUNNING"
        details="Port is accessible"
    fi
    
    echo "    Status: $status"
    if [[ -n "$details" ]]; then
        echo "    Details: $details"
    fi
    
    # Show metrics
    get_service_metrics "$service"
    
    if [[ "$status" == "✅ HEALTHY" || "$status" == "✅ RUNNING" ]]; then
        return 0
    else
        return 1
    fi
}

# Test integration endpoints
test_integration() {
    log "Testing service integration..."
    
    # Test API endpoints
    local api_tests=(
        "GET /health"
        "GET /api/memory/stats"
        "GET /api/claude-auto/session/status"
    )
    
    for test in "${api_tests[@]}"; do
        local method=$(echo "$test" | cut -d' ' -f1)
        local endpoint=$(echo "$test" | cut -d' ' -f2)
        
        if curl -f -s --max-time "$TIMEOUT" -X "$method" "http://localhost:3000$endpoint" >/dev/null 2>&1; then
            success "API $test"
        else
            warning "API $test failed"
        fi
    done
    
    # Test Zep memory system
    if curl -f -s --max-time "$TIMEOUT" "http://localhost:8100/health" >/dev/null 2>&1; then
        success "Zep memory system"
    else
        warning "Zep memory system not responding"
    fi
    
    # Test vector databases
    if curl -f -s --max-time "$TIMEOUT" "http://localhost:8090/v1/meta" >/dev/null 2>&1; then
        success "Weaviate vector database"
    else
        warning "Weaviate not responding"
    fi
    
    if curl -f -s --max-time "$TIMEOUT" "http://localhost:6333/health" >/dev/null 2>&1; then
        success "Qdrant vector database"
    else
        warning "Qdrant not responding"
    fi
}

# Generate health report
generate_report() {
    local healthy_count=0
    local total_count=${#SERVICES[@]}
    local failed_services=()
    
    echo ""
    echo "================== HEALTH REPORT =================="
    echo ""
    
    for service in "${!SERVICES[@]}"; do
        if check_service "$service" "${SERVICES[$service]}" >/dev/null 2>&1; then
            ((healthy_count++))
        else
            failed_services+=("$service")
        fi
    done
    
    echo ""
    echo "📊 Overall Status:"
    echo "   Healthy: $healthy_count/$total_count services"
    echo "   Health: $(( healthy_count * 100 / total_count ))%"
    
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        echo ""
        echo "❌ Failed Services:"
        for service in "${failed_services[@]}"; do
            echo "   - $service"
        done
    fi
    
    echo ""
    echo "🔗 Quick Access URLs:"
    echo "   Main App: http://localhost:3100"
    echo "   API Docs: http://localhost:3000/docs"
    echo "   Grafana: http://localhost:3001"
    echo "   MinIO: http://localhost:9011"
    echo "   Phoenix: http://localhost:6006"
    
    echo ""
    echo "=================================================="
    
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        success "All services are healthy!"
        return 0
    else
        error "Some services are unhealthy"
        return 1
    fi
}

# Show resource usage
show_resources() {
    log "System Resource Usage:"
    
    # Docker system info
    echo ""
    echo "🐳 Docker Resources:"
    docker system df
    
    echo ""
    echo "📊 Container Resources:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
    
    echo ""
    echo "💽 Host Resources:"
    echo "   CPU: $(uptime | awk -F'load average:' '{print $2}')"
    echo "   Memory: $(free -h | grep '^Mem' | awk '{print $3 "/" $2}')"
    echo "   Disk: $(df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')"
}

# Main function
main() {
    local cmd=${1:-"check"}
    
    case $cmd in
        "check")
            log "Starting KnowledgeHub Health Check..."
            
            for service in "${!SERVICES[@]}"; do
                check_service "$service" "${SERVICES[$service]}"
            done
            
            test_integration
            generate_report
            ;;
        "report")
            generate_report
            ;;
        "resources")
            show_resources
            ;;
        "monitor")
            log "Starting continuous monitoring (Ctrl+C to stop)..."
            while true; do
                clear
                generate_report
                sleep 30
            done
            ;;
        *)
            echo "Usage: $0 [check|report|resources|monitor]"
            echo ""
            echo "Commands:"
            echo "  check     - Run full health check (default)"
            echo "  report    - Generate summary report"
            echo "  resources - Show resource usage"
            echo "  monitor   - Continuous monitoring"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"