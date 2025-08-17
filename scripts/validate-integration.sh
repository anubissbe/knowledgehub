#!/bin/bash

# KnowledgeHub Integration Validation Script
# Validates all services are properly integrated and functional

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMEOUT=30
LOG_FILE="$PROJECT_DIR/validation.log"

# Test counters
TESTS_TOTAL=0
TESTS_PASSED=0
TESTS_FAILED=0

# Functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}✅ $1${NC}" | tee -a "$LOG_FILE"
    ((TESTS_PASSED++))
}

failure() {
    echo -e "${RED}❌ $1${NC}" | tee -a "$LOG_FILE"
    ((TESTS_FAILED++))
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    ((TESTS_TOTAL++))
    echo -n "Testing: $test_name... "
    
    if eval "$test_command" >/dev/null 2>&1; then
        success "$test_name"
        return 0
    else
        failure "$test_name"
        return 1
    fi
}

# Test service connectivity
test_service_connectivity() {
    log "Testing service connectivity..."
    
    # Core services
    run_test "PostgreSQL Connection" "nc -z localhost 5433"
    run_test "Redis Connection" "nc -z localhost 6381"
    run_test "TimescaleDB Connection" "nc -z localhost 5434"
    run_test "MinIO Connection" "nc -z localhost 9010"
    
    # Vector databases
    run_test "Weaviate Connection" "nc -z localhost 8090"
    run_test "Qdrant Connection" "nc -z localhost 6333"
    
    # Graph database
    run_test "Neo4j Connection" "nc -z localhost 7474"
    
    # AI and memory services
    run_test "Zep Memory Connection" "nc -z localhost 8100"
    run_test "AI Service Connection" "nc -z localhost 8002"
    run_test "Firecrawl Connection" "nc -z localhost 3002"
    run_test "Graphiti Connection" "nc -z localhost 8080"
    run_test "Phoenix Connection" "nc -z localhost 6006"
    
    # Application services
    run_test "Main API Connection" "nc -z localhost 3000"
    run_test "Web UI Connection" "nc -z localhost 3100"
    run_test "Nginx Proxy Connection" "nc -z localhost 8080"
    
    # Monitoring
    run_test "Prometheus Connection" "nc -z localhost 9090"
    run_test "Grafana Connection" "nc -z localhost 3001"
}

# Test HTTP health endpoints
test_health_endpoints() {
    log "Testing HTTP health endpoints..."
    
    run_test "Main API Health" "curl -f -s --max-time $TIMEOUT http://localhost:3000/health"
    run_test "AI Service Health" "curl -f -s --max-time $TIMEOUT http://localhost:8002/health"
    run_test "Zep Health" "curl -f -s --max-time $TIMEOUT http://localhost:8100/health"
    run_test "Phoenix Health" "curl -f -s --max-time $TIMEOUT http://localhost:6006/health"
    run_test "Qdrant Health" "curl -f -s --max-time $TIMEOUT http://localhost:6333/health"
    run_test "Nginx Health" "curl -f -s --max-time $TIMEOUT http://localhost:8080/nginx-health"
    run_test "Prometheus Health" "curl -f -s --max-time $TIMEOUT http://localhost:9090/-/healthy"
}

# Test database connectivity
test_database_connectivity() {
    log "Testing database connectivity..."
    
    # PostgreSQL
    if command -v psql &> /dev/null; then
        run_test "PostgreSQL Query" "PGPASSWORD=knowledgehub123 psql -h localhost -p 5433 -U knowledgehub -d knowledgehub -c 'SELECT 1;'"
        run_test "TimescaleDB Query" "PGPASSWORD=knowledgehub123 psql -h localhost -p 5434 -U knowledgehub -d knowledgehub_analytics -c 'SELECT 1;'"
    else
        warning "psql not available, skipping PostgreSQL tests"
    fi
    
    # Redis
    if command -v redis-cli &> /dev/null; then
        run_test "Redis Ping" "redis-cli -h localhost -p 6381 ping | grep -q PONG"
    else
        warning "redis-cli not available, skipping Redis tests"
    fi
}

# Test API endpoints
test_api_endpoints() {
    log "Testing API endpoints..."
    
    # Core API endpoints
    run_test "API Docs" "curl -f -s --max-time $TIMEOUT http://localhost:3000/docs"
    run_test "API OpenAPI" "curl -f -s --max-time $TIMEOUT http://localhost:3000/openapi.json"
    
    # Memory system endpoints
    run_test "Memory Stats" "curl -f -s --max-time $TIMEOUT http://localhost:3000/api/memory/stats"
    run_test "Session Status" "curl -f -s --max-time $TIMEOUT http://localhost:3000/api/claude-auto/session/status"
    
    # AI features endpoints
    run_test "AI Features" "curl -f -s --max-time $TIMEOUT http://localhost:3000/api/ai-features/"
    run_test "Proactive Assistant" "curl -f -s --max-time $TIMEOUT http://localhost:3000/api/proactive/"
    
    # Search endpoints
    run_test "Search Endpoint" "curl -f -s --max-time $TIMEOUT http://localhost:3000/api/search/"
    
    # Sources endpoints
    run_test "Sources List" "curl -f -s --max-time $TIMEOUT http://localhost:3000/api/sources/"
}

# Test vector database functionality
test_vector_databases() {
    log "Testing vector database functionality..."
    
    # Weaviate
    run_test "Weaviate Meta" "curl -f -s --max-time $TIMEOUT http://localhost:8090/v1/meta"
    run_test "Weaviate Schema" "curl -f -s --max-time $TIMEOUT http://localhost:8090/v1/schema"
    
    # Qdrant
    run_test "Qdrant Collections" "curl -f -s --max-time $TIMEOUT http://localhost:6333/collections"
    run_test "Qdrant Cluster" "curl -f -s --max-time $TIMEOUT http://localhost:6333/cluster"
}

# Test AI service integrations
test_ai_services() {
    log "Testing AI service integrations..."
    
    # Zep Memory System
    run_test "Zep Users List" "curl -f -s --max-time $TIMEOUT http://localhost:8100/api/v1/users"
    run_test "Zep Sessions List" "curl -f -s --max-time $TIMEOUT http://localhost:8100/api/v1/sessions"
    
    # Firecrawl (may not have health endpoint)
    run_test "Firecrawl Status" "curl -f -s --max-time $TIMEOUT http://localhost:3002/ || true"
    
    # Graphiti
    run_test "Graphiti Status" "curl -f -s --max-time $TIMEOUT http://localhost:8080/"
}

# Test web interface
test_web_interface() {
    log "Testing web interface..."
    
    # Frontend accessibility
    run_test "Web UI Index" "curl -f -s --max-time $TIMEOUT http://localhost:3100/"
    run_test "Web UI via Nginx" "curl -f -s --max-time $TIMEOUT http://localhost:8080/"
    
    # Static resources
    run_test "Favicon" "curl -f -s --max-time $TIMEOUT http://localhost:3100/favicon.ico"
}

# Test monitoring and observability
test_monitoring() {
    log "Testing monitoring and observability..."
    
    # Prometheus
    run_test "Prometheus Targets" "curl -f -s --max-time $TIMEOUT http://localhost:9090/api/v1/targets"
    run_test "Prometheus Config" "curl -f -s --max-time $TIMEOUT http://localhost:9090/api/v1/status/config"
    
    # Grafana
    run_test "Grafana Health" "curl -f -s --max-time $TIMEOUT http://localhost:3001/api/health"
    
    # Phoenix Observability
    run_test "Phoenix UI" "curl -f -s --max-time $TIMEOUT http://localhost:6006/"
}

# Test service dependencies
test_service_dependencies() {
    log "Testing service dependencies..."
    
    # Test that services can communicate with their dependencies
    
    # API -> Database
    run_test "API -> PostgreSQL" "curl -f -s --max-time $TIMEOUT http://localhost:3000/api/memory/stats | grep -q 'total_memories'"
    
    # API -> Redis
    run_test "API -> Redis" "curl -f -s --max-time $TIMEOUT http://localhost:3000/health | grep -q 'redis'"
    
    # API -> Vector DB
    run_test "API -> Weaviate" "curl -f -s --max-time $TIMEOUT -X POST http://localhost:3000/api/search/ -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"limit\":1}'"
}

# Test data flow
test_data_flow() {
    log "Testing data flow..."
    
    # Create a test memory entry
    local test_id="integration-test-$(date +%s)"
    local test_data="{\"content\":\"Integration test memory\",\"project\":\"test\",\"context\":\"$test_id\"}"
    
    # Test memory creation
    if curl -f -s --max-time $TIMEOUT -X POST http://localhost:3000/api/memory/ \
        -H "Content-Type: application/json" \
        -d "$test_data" > /dev/null; then
        success "Memory Creation"
        ((TESTS_PASSED++))
        
        # Test memory retrieval
        if curl -f -s --max-time $TIMEOUT "http://localhost:3000/api/memory/?context=$test_id" | grep -q "$test_id"; then
            success "Memory Retrieval"
            ((TESTS_PASSED++))
        else
            failure "Memory Retrieval"
            ((TESTS_FAILED++))
        fi
    else
        failure "Memory Creation"
        ((TESTS_FAILED++))
    fi
    
    ((TESTS_TOTAL += 2))
}

# Test load balancing and performance
test_performance() {
    log "Testing performance..."
    
    # Test concurrent requests
    local concurrent_requests=5
    local temp_file="/tmp/concurrent_test_$$"
    
    for i in $(seq 1 $concurrent_requests); do
        (curl -f -s --max-time $TIMEOUT http://localhost:3000/health > "$temp_file.$i") &
    done
    
    wait
    
    local success_count=0
    for i in $(seq 1 $concurrent_requests); do
        if [[ -f "$temp_file.$i" ]] && grep -q "status" "$temp_file.$i"; then
            ((success_count++))
        fi
        rm -f "$temp_file.$i"
    done
    
    if [[ $success_count -eq $concurrent_requests ]]; then
        success "Concurrent Request Handling ($success_count/$concurrent_requests)"
        ((TESTS_PASSED++))
    else
        failure "Concurrent Request Handling ($success_count/$concurrent_requests)"
        ((TESTS_FAILED++))
    fi
    
    ((TESTS_TOTAL++))
}

# Generate integration report
generate_report() {
    log "Generating integration validation report..."
    
    local report_file="$PROJECT_DIR/integration-validation-$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
    "validation_timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "environment": {
        "docker_version": "$(docker --version | cut -d' ' -f3 | tr -d ',')",
        "docker_compose_version": "$(docker-compose --version | cut -d' ' -f3 | tr -d ',')",
        "hostname": "$(hostname)",
        "os": "$(uname -s)",
        "architecture": "$(uname -m)"
    },
    "test_results": {
        "total_tests": $TESTS_TOTAL,
        "passed": $TESTS_PASSED,
        "failed": $TESTS_FAILED,
        "success_rate": "$(echo "scale=2; $TESTS_PASSED * 100 / $TESTS_TOTAL" | bc)%"
    },
    "services_status": {
        "core_services": "$(docker-compose ps | grep -E "(postgres|redis|timescale|minio)" | grep -c "Up")/4",
        "vector_databases": "$(docker-compose ps | grep -E "(weaviate|qdrant)" | grep -c "Up")/2",
        "ai_services": "$(docker-compose ps | grep -E "(zep|ai-service|firecrawl|graphiti|phoenix)" | grep -c "Up")/5",
        "application": "$(docker-compose ps | grep -E "(api|webui|nginx)" | grep -c "Up")/3",
        "monitoring": "$(docker-compose ps | grep -E "(prometheus|grafana)" | grep -c "Up")/2"
    },
    "resource_usage": {
        "disk_usage": "$(df -h $PROJECT_DIR | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')",
        "memory_usage": "$(free -h | grep '^Mem' | awk '{print $3 "/" $2}')",
        "docker_images": "$(docker images | wc -l)",
        "docker_containers": "$(docker ps | wc -l)"
    }
}
EOF
    
    success "Integration report saved to $report_file"
}

# Show summary
show_summary() {
    echo ""
    echo "=================================================="
    echo "           INTEGRATION VALIDATION SUMMARY"
    echo "=================================================="
    echo ""
    echo "📊 Test Results:"
    echo "   Total Tests: $TESTS_TOTAL"
    echo "   Passed: $TESTS_PASSED"
    echo "   Failed: $TESTS_FAILED"
    echo "   Success Rate: $(echo "scale=2; $TESTS_PASSED * 100 / $TESTS_TOTAL" | bc)%"
    echo ""
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}🎉 ALL TESTS PASSED! Integration is successful.${NC}"
        echo ""
        echo "🔗 Service URLs:"
        echo "   Main Application: http://localhost:3100"
        echo "   API Documentation: http://localhost:3000/docs"
        echo "   Grafana Dashboard: http://localhost:3001"
        echo "   Phoenix Observability: http://localhost:6006"
        echo "   Prometheus: http://localhost:9090"
        echo ""
        echo "✅ The KnowledgeHub system is fully operational!"
    else
        echo -e "${RED}❌ Some tests failed. Review the logs for details.${NC}"
        echo ""
        echo "🔍 Troubleshooting:"
        echo "   1. Check service logs: docker-compose logs <service-name>"
        echo "   2. Verify service status: docker-compose ps"
        echo "   3. Run health check: ./scripts/health-check.sh"
        echo "   4. Check system resources: docker system df"
    fi
    
    echo ""
    echo "📝 Logs available at: $LOG_FILE"
    echo "=================================================="
}

# Main function
main() {
    log "Starting KnowledgeHub Integration Validation"
    
    # Initialize log file
    echo "# KnowledgeHub Integration Validation - $(date)" > "$LOG_FILE"
    
    # Run test suites
    test_service_connectivity
    test_health_endpoints
    test_database_connectivity
    test_api_endpoints
    test_vector_databases
    test_ai_services
    test_web_interface
    test_monitoring
    test_service_dependencies
    test_data_flow
    test_performance
    
    # Generate report and show summary
    generate_report
    show_summary
    
    # Exit with appropriate code
    if [[ $TESTS_FAILED -eq 0 ]]; then
        exit 0
    else
        exit 1
    fi
}

# Run main function
main "$@"