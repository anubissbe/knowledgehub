# Quick Start Guide

Get KnowledgeHub up and running in under 5 minutes! This guide provides the fastest path to a working installation.

## Prerequisites

- Docker 24.0+ with Docker Compose
- Git
- 8GB RAM minimum
- 20GB free disk space

## 🚀 5-Minute Setup

### Step 1: Clone and Configure

```bash
# Clone the repository
git clone https://github.com/anubissbe/knowledgehub.git
cd knowledgehub

# Copy environment configuration
cp .env.example .env
```

### Step 2: Start KnowledgeHub

```bash
# Start all services
docker compose up -d

# Watch the magic happen (optional)
docker compose logs -f
```

### Step 3: Wait for Services

Services take about 2-3 minutes to initialize. Check status:

```bash
# Check if services are ready
curl http://localhost:3000/health

# You should see:
# {"status":"healthy","services":{"api":"operational",...}}
```

### Step 4: Access the Application

Open your browser and navigate to:
- **Web UI**: http://localhost:3101
- **API Docs**: http://localhost:3000/docs

## 🎯 Your First Knowledge Source

### 1. Navigate to Sources

Click on "Sources" in the navigation menu.

### 2. Add a New Source

Click the "Add Source" button and fill in:
- **Name**: "Python Documentation"
- **URL**: `https://docs.python.org/3/`
- **Type**: Web
- **Max Depth**: 2
- **Max Pages**: 100

Click "Create Source" to start crawling.

### 3. Monitor Progress

Go to "Jobs" to watch the crawling progress in real-time. You'll see:
- Pages discovered
- Pages processed
- Processing speed
- Live log updates

### 4. Search Your Knowledge

Once crawling completes, go to "Search" and try:
- "python list comprehension"
- "async await tutorial"
- "exception handling"

## 📋 Common Tasks

### View All Sources
```bash
curl http://localhost:3000/api/v1/sources
```

### Check Job Status
```bash
curl http://localhost:3000/api/v1/jobs
```

### Search via API
```bash
curl -X POST http://localhost:3000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "python tutorial", "limit": 5}'
```

### Stop Services
```bash
docker compose down
```

### View Logs
```bash
# All services
docker compose logs

# Specific service
docker compose logs api

# Follow logs
docker compose logs -f scraper
```

## 🎨 Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Web UI | http://localhost:3101 | Main user interface |
| API | http://localhost:3000 | REST API |
| API Docs | http://localhost:3000/docs | Interactive API documentation |
| MinIO | http://localhost:9011 | Object storage console |
| Weaviate | http://localhost:8090 | Vector database |

## ⚡ Performance Tips

### Quick Crawl Settings
For faster initial testing:
- **Max Depth**: 1-2
- **Max Pages**: 50-100
- **JavaScript**: Disable if not needed

### Resource Optimization
```yaml
# Create docker-compose.override.yml
version: '3.8'
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
  scraper:
    deploy:
      resources:
        limits:
          memory: 2G
```

## 🔧 Troubleshooting

### Services Won't Start
```bash
# Check for port conflicts
netstat -tulpn | grep -E '3000|3101|8080|9000'

# Check Docker resources
docker system df
docker system prune -a
```

### Can't Access Web UI
1. Check if services are running: `docker compose ps`
2. Verify ports: `curl http://localhost:3101`
3. Check firewall: `sudo ufw status`

### Search Returns No Results
1. Ensure crawling completed: Check Jobs page
2. Verify data exists: `curl http://localhost:3000/api/v1/sources/{id}/stats`
3. Check logs: `docker compose logs rag-processor`

## 📚 Next Steps

Now that you have KnowledgeHub running:

1. **[User Guide](User-Guide)** - Learn all features
2. **[Configuration](Configuration)** - Customize your setup
3. **[API Documentation](API-Documentation)** - Integrate with your apps
4. **[Tutorials](Tutorials)** - Advanced use cases

## 🆘 Need Help?

- Check [FAQ](FAQ) for common questions
- Browse [Troubleshooting](Troubleshooting) for detailed solutions
- Search [GitHub Issues](https://github.com/anubissbe/knowledgehub/issues)
- Join our community discussions

---

**Congratulations!** 🎉 You now have a working KnowledgeHub installation. Start adding more sources and building your knowledge base!