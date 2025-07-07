#!/bin/bash

# KnowledgeHub Control Script
# Simple script to start, stop, restart, and check status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

case "$1" in
    start)
        echo -e "${BLUE}🚀 Starting KnowledgeHub...${NC}"
        ./deploy/deploy-local.sh
        ;;
    
    stop)
        echo -e "${YELLOW}🛑 Stopping KnowledgeHub...${NC}"
        docker compose down
        echo -e "${GREEN}✅ KnowledgeHub stopped${NC}"
        ;;
    
    restart)
        echo -e "${YELLOW}🔄 Restarting KnowledgeHub...${NC}"
        docker compose restart
        echo -e "${GREEN}✅ KnowledgeHub restarted${NC}"
        ;;
    
    status)
        echo -e "${BLUE}📊 KnowledgeHub Status:${NC}"
        docker compose ps
        echo ""
        ./scripts/check-health.sh
        ;;
    
    logs)
        shift
        docker compose logs -f "$@"
        ;;
    
    update)
        echo -e "${BLUE}🔄 Updating KnowledgeHub...${NC}"
        git pull
        docker compose build --pull
        docker compose up -d
        echo -e "${GREEN}✅ KnowledgeHub updated${NC}"
        ;;
    
    backup)
        echo -e "${BLUE}📦 Backing up KnowledgeHub...${NC}"
        ./scripts/backup.sh
        ;;
    
    clean)
        echo -e "${YELLOW}🧹 Cleaning up...${NC}"
        docker compose down -v
        docker system prune -f
        echo -e "${GREEN}✅ Cleanup complete${NC}"
        ;;
    
    *)
        echo "KnowledgeHub Control Script"
        echo "=========================="
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs|update|backup|clean}"
        echo ""
        echo "Commands:"
        echo "  start    - Start all KnowledgeHub services"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  status   - Show service status and health"
        echo "  logs     - Follow service logs (optional: service name)"
        echo "  update   - Pull latest code and redeploy"
        echo "  backup   - Create backup of data"
        echo "  clean    - Stop services and clean up volumes"
        echo ""
        echo "Examples:"
        echo "  $0 start"
        echo "  $0 logs api"
        echo "  $0 status"
        exit 1
        ;;
esac