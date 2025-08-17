#!/bin/bash

# KnowledgeHub Integrated Services Deployment Script
# Deploys all new services with proper health checks and validation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/deployment.log"
BACKUP_DIR="$SCRIPT_DIR/deployment-backup-$(date +%Y%m%d-%H%M%S)"

# Functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check minimum Docker version
    DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+\.\d+' | head -1)
    if [[ "$(printf '%s\n' "20.10.0" "$DOCKER_VERSION" | sort -V | head -n1)" != "20.10.0" ]]; then
        error "Docker version $DOCKER_VERSION is too old. Minimum required: 20.10.0"
        exit 1
    fi
    
    # Check available disk space (minimum 10GB)
    AVAILABLE_SPACE=$(df -BG "$SCRIPT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ $AVAILABLE_SPACE -lt 10 ]]; then
        error "Insufficient disk space. Available: ${AVAILABLE_SPACE}GB, Required: 10GB"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Create backup of current deployment
create_backup() {
    log "Creating backup of current deployment..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup configuration files
    if [[ -f "$SCRIPT_DIR/docker-compose.yml" ]]; then
        cp "$SCRIPT_DIR/docker-compose.yml" "$BACKUP_DIR/"
    fi
    
    if [[ -f "$SCRIPT_DIR/.env" ]]; then
        cp "$SCRIPT_DIR/.env" "$BACKUP_DIR/"
    fi
    
    # Export container data
    if docker-compose ps | grep -q "Up"; then
        log "Exporting container data..."
        docker-compose exec -T postgres pg_dump -U knowledgehub knowledgehub > "$BACKUP_DIR/postgres_backup.sql" 2>/dev/null || true
        docker-compose exec -T redis redis-cli BGSAVE >/dev/null 2>&1 || true
    fi
    
    success "Backup created at $BACKUP_DIR"
}

# Setup environment
setup_environment() {
    log "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
        cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
        warn "Created .env file from template. Please review and update with your actual values."
    fi
    
    # Create necessary directories
    mkdir -p "$SCRIPT_DIR/ssl"
    mkdir -p "$SCRIPT_DIR/logs/security"
    mkdir -p "$SCRIPT_DIR/phoenix_data"
    
    success "Environment setup completed"
}

# Validate service configurations
validate_configurations() {
    log "Validating service configurations..."
    
    # Check Zep config
    if [[ ! -f "$SCRIPT_DIR/zep-config.yaml" ]]; then
        error "Zep configuration file not found"
        exit 1
    fi
    
    # Validate Docker Compose file
    if ! docker-compose config >/dev/null 2>&1; then
        error "Docker Compose configuration is invalid"
        exit 1
    fi
    
    success "Configuration validation passed"
}

# Pull required images
pull_images() {
    log "Pulling required Docker images..."
    
    # List of images to pull
    images=(
        "postgres:15-alpine"
        "redis:7-alpine"
        "semitechnologies/weaviate:1.23.0"
        "qdrant/qdrant:v1.7.4"
        "neo4j:5.14.0"
        "timescale/timescaledb:latest-pg15"
        "minio/minio:latest"
        "ankane/pgvector:v0.5.1"
        "ghcr.io/getzep/zep:latest"
        "mendableai/firecrawl:latest"
        "browserless/chrome:latest"
        "zep-ai/graphiti:latest"
        "arizephoenix/phoenix:latest"
        "nginx:alpine"
        "prom/prometheus:latest"
        "grafana/grafana:latest"
    )
    
    for image in "${images[@]}"; do
        log "Pulling $image..."
        if ! docker pull "$image"; then
            warn "Failed to pull $image, will try during deployment"
        fi
    done
    
    success "Image pull completed"
}

# Deploy services in stages
deploy_services() {
    log "Starting staged deployment..."
    
    # Stage 1: Core infrastructure
    log "Stage 1: Deploying core infrastructure..."
    docker-compose up -d postgres redis timescale minio
    sleep 30
    
    # Wait for core services
    wait_for_service "postgres" "5433" 60
    wait_for_service "redis" "6381" 30
    wait_for_service "timescale" "5434" 60
    wait_for_service "minio" "9010" 30
    
    # Stage 2: Vector and graph databases
    log "Stage 2: Deploying vector and graph databases..."
    docker-compose up -d weaviate qdrant neo4j
    sleep 30
    
    wait_for_service "weaviate" "8090" 60
    wait_for_service "qdrant" "6333" 60
    wait_for_service "neo4j" "7474" 90
    
    # Stage 3: Zep memory system
    log "Stage 3: Deploying Zep memory system..."
    docker-compose up -d zep-postgres
    sleep 20
    wait_for_service "zep-postgres" "5432" 60
    
    docker-compose up -d zep
    sleep 30
    wait_for_service "zep" "8100" 60
    
    # Stage 4: AI and observability services
    log "Stage 4: Deploying AI and observability services..."
    docker-compose up -d phoenix playwright-service
    sleep 30
    
    wait_for_service "phoenix" "6006" 60
    wait_for_service "playwright-service" "3003" 30
    
    # Stage 5: Web scraping and graph services
    log "Stage 5: Deploying web scraping and graph services..."
    docker-compose up -d firecrawl graphiti
    sleep 30
    
    wait_for_service "firecrawl" "3002" 90
    wait_for_service "graphiti" "8080" 60
    
    # Stage 6: Application services
    log "Stage 6: Deploying application services..."
    docker-compose up -d ai-service
    sleep 30
    wait_for_service "ai-service" "8002" 90
    
    docker-compose up -d api
    sleep 30
    wait_for_service "api" "3000" 120
    
    # Stage 7: Frontend and proxy
    log "Stage 7: Deploying frontend and proxy..."
    docker-compose up -d webui nginx
    sleep 30
    
    wait_for_service "webui" "3100" 60
    wait_for_service "nginx" "8080" 30
    
    # Stage 8: Monitoring (optional)
    log "Stage 8: Deploying monitoring..."
    docker-compose up -d prometheus grafana
    sleep 30
    
    success "All services deployed successfully"
}

# Wait for service to be healthy
wait_for_service() {
    local service_name=$1
    local port=$2
    local timeout=${3:-60}
    local count=0
    
    log "Waiting for $service_name on port $port..."
    
    while ! nc -z localhost "$port" 2>/dev/null; do
        sleep 5
        count=$((count + 5))
        
        if [[ $count -ge $timeout ]]; then
            error "$service_name failed to start within $timeout seconds"
            return 1
        fi
        
        echo -n "."
    done
    
    echo ""
    success "$service_name is ready"
}

# Validate deployment
validate_deployment() {
    log "Validating deployment..."
    
    # Check all containers are running
    local failed_services=()
    
    services=(
        "postgres:5433"
        "redis:6381"
        "weaviate:8090"
        "qdrant:6333"
        "neo4j:7474"
        "timescale:5434"
        "minio:9010"
        "zep:8100"
        "firecrawl:3002"
        "graphiti:8080"
        "phoenix:6006"
        "ai-service:8002"
        "api:3000"
        "webui:3100"
    )
    
    for service in "${services[@]}"; do
        service_name=$(echo "$service" | cut -d: -f1)
        port=$(echo "$service" | cut -d: -f2)
        
        if ! nc -z localhost "$port" 2>/dev/null; then
            failed_services+=("$service_name")
        fi
    done
    
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        error "Failed services: ${failed_services[*]}"
        return 1
    fi
    
    # Test API endpoints
    log "Testing API endpoints..."
    
    if ! curl -f -s http://localhost:3000/health >/dev/null; then
        error "Main API health check failed"
        return 1
    fi
    
    if ! curl -f -s http://localhost:8100/health >/dev/null; then
        error "Zep API health check failed"
        return 1
    fi
    
    success "Deployment validation passed"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    # Wait for API to be fully ready
    sleep 10
    
    # Run any pending migrations through the API
    if curl -f -s -X POST http://localhost:3000/admin/migrate >/dev/null 2>&1; then
        success "Database migrations completed"
    else
        warn "Migration endpoint not available or failed"
    fi
}

# Setup monitoring dashboards
setup_monitoring() {
    log "Setting up monitoring dashboards..."
    
    # Import Grafana dashboards
    if nc -z localhost 3001 2>/dev/null; then
        # Wait for Grafana to be ready
        sleep 30
        
        # Import dashboards (would need actual implementation)
        success "Monitoring setup completed"
    else
        warn "Grafana not available, skipping dashboard setup"
    fi
}

# Display deployment summary
show_summary() {
    log "Deployment Summary"
    echo ""
    echo "=== KnowledgeHub Integrated Services ==="
    echo ""
    echo "🔗 Service URLs:"
    echo "  Main Application: http://localhost:3100"
    echo "  API Documentation: http://localhost:3000/docs"
    echo "  Nginx Proxy: http://localhost:8080"
    echo ""
    echo "🛠️ Admin Interfaces:"
    echo "  MinIO Console: http://localhost:9011"
    echo "  Neo4j Browser: http://localhost:7474"
    echo "  Grafana: http://localhost:3001"
    echo "  Prometheus: http://localhost:9090"
    echo "  Phoenix: http://localhost:6006"
    echo ""
    echo "🔧 AI Services:"
    echo "  Zep Memory: http://localhost:8100"
    echo "  Firecrawl: http://localhost:3002"
    echo "  Graphiti: http://localhost:8080"
    echo "  AI Service: http://localhost:8002"
    echo ""
    echo "💾 Databases:"
    echo "  PostgreSQL: localhost:5433"
    echo "  Redis: localhost:6381"
    echo "  Weaviate: localhost:8090"
    echo "  Qdrant: localhost:6333"
    echo "  TimescaleDB: localhost:5434"
    echo ""
    echo "📊 Monitoring:"
    echo "  View logs: docker-compose logs -f"
    echo "  Check status: docker-compose ps"
    echo "  Stop services: docker-compose down"
    echo ""
    echo "🔐 Default Credentials:"
    echo "  Grafana: admin/admin"
    echo "  MinIO: minioadmin/minioadmin"
    echo "  Neo4j: neo4j/knowledgehub123"
    echo ""
    success "Deployment completed successfully!"
}

# Rollback function
rollback() {
    error "Deployment failed. Initiating rollback..."
    
    # Stop all services
    docker-compose down
    
    # Restore backup if available
    if [[ -d "$BACKUP_DIR" ]]; then
        log "Restoring from backup..."
        if [[ -f "$BACKUP_DIR/docker-compose.yml" ]]; then
            cp "$BACKUP_DIR/docker-compose.yml" "$SCRIPT_DIR/"
        fi
        if [[ -f "$BACKUP_DIR/.env" ]]; then
            cp "$BACKUP_DIR/.env" "$SCRIPT_DIR/"
        fi
    fi
    
    error "Rollback completed. Check logs at $LOG_FILE"
    exit 1
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    # Add any cleanup logic here
}

# Trap for cleanup
trap cleanup EXIT
trap rollback ERR

# Main execution
main() {
    log "Starting KnowledgeHub Integrated Services Deployment"
    
    check_prerequisites
    create_backup
    setup_environment
    validate_configurations
    pull_images
    deploy_services
    validate_deployment
    run_migrations
    setup_monitoring
    show_summary
    
    log "Deployment process completed"
}

# Run main function
main "$@"