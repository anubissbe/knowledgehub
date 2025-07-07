# KnowledgeHub Deployment

## Quick Start (Local Deployment)

```bash
# From the project root
./knowledgehub.sh start
```

## Available Scripts

### Local Deployment
- `deploy-local.sh` - Full local deployment with health checks
- `knowledgehub.service` - Systemd service file for auto-start

### Remote Deployment (Synology)
- `deploy.sh` - Full deployment with image export/import
- `deploy-simple.sh` - Simple rsync-based deployment

## Service Management

### Start Services
```bash
./knowledgehub.sh start
```

### Stop Services
```bash
./knowledgehub.sh stop
```

### Check Status
```bash
./knowledgehub.sh status
```

### View Logs
```bash
./knowledgehub.sh logs          # All services
./knowledgehub.sh logs api      # Specific service
```

## Access Points

After deployment, access KnowledgeHub at:
- **Web UI**: http://localhost:3100
- **API**: http://localhost:3000/docs
- **MCP Server**: http://localhost:3008
- **Grafana**: http://localhost:3030 (if monitoring enabled)

## Systemd Integration (Optional)

To enable auto-start on boot:

```bash
sudo cp deploy/knowledgehub.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable knowledgehub
sudo systemctl start knowledgehub
```

## Troubleshooting

### Services Won't Start
```bash
# Check logs
docker compose logs -f

# Check disk space
df -h

# Clean up and retry
docker system prune -f
./knowledgehub.sh clean
./knowledgehub.sh start
```

### Port Conflicts
Edit `docker-compose.yml` to change port mappings if needed.

### Memory Issues
Ensure at least 4GB RAM available for all services.