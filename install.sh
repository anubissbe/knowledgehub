#!/bin/bash

# KnowledgeHub Installation Script (FIXED VERSION)
# This script properly handles Docker Compose v2 and includes error handling

set -e

echo "🚀 KnowledgeHub Installation Script (Fixed Version)"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Detect Docker Compose version
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
    echo -e "${GREEN}✓${NC} Using Docker Compose v2"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
    echo -e "${YELLOW}⚠${NC} Using Docker Compose v1 (legacy)"
else
    echo -e "${RED}✗${NC} Docker Compose not found. Please install Docker Desktop or docker-compose"
    exit 1
fi

# Check prerequisites
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}✗${NC} $1 is not installed. Please install $1 first."
        exit 1
    else
        echo -e "${GREEN}✓${NC} $1 found"
    fi
}

echo "Checking prerequisites..."
check_command docker
check_command git
check_command curl

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo -e "${GREEN}✓${NC} Created .env file"
    echo -e "${YELLOW}⚠${NC} Please edit .env file with your configuration"
    sleep 3
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs webui/public webui/build
echo -e "${GREEN}✓${NC} Directories created"

# Fix port mismatch in AI service
echo "Fixing configuration issues..."
if [ -f "ai-service/Dockerfile" ]; then
    sed -i.bak 's/EXPOSE 8000/EXPOSE 8002/' ai-service/Dockerfile 2>/dev/null || \
    sed -i '' 's/EXPOSE 8000/EXPOSE 8002/' ai-service/Dockerfile 2>/dev/null || true
fi

# Build custom images
echo "Building KnowledgeHub services..."
$DOCKER_COMPOSE build --no-cache || {
    echo -e "${RED}✗${NC} Build failed. Trying without cache..."
    $DOCKER_COMPOSE build
}

# Start core services first
echo "Starting core services (PostgreSQL, Redis)..."
$DOCKER_COMPOSE up -d postgres redis

# Wait for PostgreSQL with timeout
echo "Waiting for PostgreSQL to be ready..."
TIMEOUT=60
ELAPSED=0
while ! $DOCKER_COMPOSE exec -T postgres pg_isready -U knowledgehub &>/dev/null; do
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo -e "${RED}✗${NC} PostgreSQL failed to start within ${TIMEOUT} seconds"
        echo "Checking logs..."
        $DOCKER_COMPOSE logs postgres --tail=20
        exit 1
    fi
    echo -n "."
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done
echo -e " ${GREEN}Ready!${NC}"

# Run database migrations with error handling
echo "Running database migrations..."
for migration in migrations/*.sql; do
    if [ -f "$migration" ]; then
        echo "Applying $(basename $migration)..."
        $DOCKER_COMPOSE exec -T postgres psql -U knowledgehub -d knowledgehub < "$migration" || {
            echo -e "${YELLOW}⚠${NC} Warning: Migration $(basename $migration) had issues (may already be applied)"
        }
    fi
done

# Start remaining services with health checks
echo "Starting all services..."
$DOCKER_COMPOSE up -d

# Wait for services with progress indicator
echo "Waiting for services to be ready..."
services=("api:3000/health" "webui:80" "weaviate:8080/v1/.well-known/ready" "neo4j:7474")
for service in "${services[@]}"; do
    name="${service%%:*}"
    url="${service#*:}"
    
    echo -n "  Checking $name..."
    TIMEOUT=30
    ELAPSED=0
    while ! curl -f -s "http://localhost:$url" > /dev/null 2>&1; do
        if [ $ELAPSED -ge $TIMEOUT ]; then
            echo -e " ${YELLOW}⚠ Timeout${NC}"
            break
        fi
        echo -n "."
        sleep 2
        ELAPSED=$((ELAPSED + 2))
    done
    if [ $ELAPSED -lt $TIMEOUT ]; then
        echo -e " ${GREEN}✓${NC}"
    fi
done

# Create helper script
cat > knowledgehub << 'EOF'
#!/bin/bash
# KnowledgeHub CLI Helper

case "$1" in
    start)
        docker compose up -d
        ;;
    stop)
        docker compose down
        ;;
    restart)
        docker compose restart
        ;;
    logs)
        docker compose logs -f ${2:-}
        ;;
    status)
        docker compose ps
        ;;
    *)
        echo "Usage: ./knowledgehub {start|stop|restart|logs|status}"
        exit 1
        ;;
esac
EOF
chmod +x knowledgehub

# Final status check
echo ""
echo "Installation Summary"
echo "==================="
echo ""

# Check each service
check_service() {
    if curl -f -s $2 > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $1 is running at $2"
        return 0
    else
        echo -e "${RED}✗${NC} $1 is not responding at $2"
        return 1
    fi
}

SERVICES_OK=0
check_service "API Gateway" "http://localhost:3000/health" && ((SERVICES_OK++))
check_service "Web UI" "http://localhost:3100" && ((SERVICES_OK++))
check_service "Weaviate" "http://localhost:8090/v1/.well-known/ready" && ((SERVICES_OK++))
check_service "Neo4j" "http://localhost:7474" && ((SERVICES_OK++))
check_service "MinIO" "http://localhost:9010/minio/health/live" && ((SERVICES_OK++))

echo ""
if [ $SERVICES_OK -ge 3 ]; then
    echo -e "${GREEN}🎉 Installation successful!${NC}"
    echo ""
    echo "Access KnowledgeHub at:"
    echo "  - Web UI: http://localhost:3100"
    echo "  - API Docs: http://localhost:3000/docs"
    echo "  - Neo4j Browser: http://localhost:7474"
    echo "  - MinIO Console: http://localhost:9011"
    echo ""
    echo "Quick commands:"
    echo "  ./knowledgehub status  - Check service status"
    echo "  ./knowledgehub logs    - View logs"
    echo "  ./knowledgehub stop    - Stop all services"
else
    echo -e "${YELLOW}⚠ Installation completed with warnings${NC}"
    echo ""
    echo "Some services may not be fully ready yet."
    echo "Check logs with: ./knowledgehub logs"
fi

echo ""
echo "Default credentials:"
echo "  - Neo4j: neo4j / knowledgehub123"
echo "  - MinIO: minioadmin / minioadmin"
echo ""
echo -e "${YELLOW}⚠${NC} Remember to update passwords in .env for production!"