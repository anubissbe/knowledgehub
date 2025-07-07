# KnowledgeHub Quick Start Guide

## 🚀 Get Started in 5 Minutes

This guide will get you up and running with KnowledgeHub quickly.

## Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available
- 10GB free disk space

## Step 1: Deploy KnowledgeHub

### Option A: One-Command Deployment
```bash
cd /opt/projects/knowledgehub
./knowledgehub.sh start
```

### Option B: Smart Deployment
```bash
cd /opt/projects/knowledgehub
./deploy/smart-deploy.sh
```

Wait for all services to start (usually 2-3 minutes).

## Step 2: Verify Installation

1. **Check Service Status**:
```bash
./knowledgehub.sh status
```

2. **Open KnowledgeHub**:
   - Go to: http://localhost:3100
   - You should see the dashboard

3. **Test API**:
```bash
curl http://localhost:3000/api/health
```

## Step 3: Add Your First Knowledge Source

### Using the Web Interface
1. Click **"Add Source"** on the dashboard
2. Enter a documentation URL (e.g., `https://docs.python.org`)
3. Give it a name: "Python Documentation"
4. Click **"Start Crawling"**

### Using the API
```bash
curl -X POST http://localhost:3000/api/v1/sources/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Python Docs",
    "url": "https://docs.python.org/3/",
    "type": "documentation"
  }'
```

## Step 4: Monitor Crawling Progress

1. **Web Interface**: Check the "Sources" page for status
2. **API**: 
```bash
curl http://localhost:3000/api/v1/sources/
```

3. **Logs**:
```bash
./knowledgehub.sh logs scraper
```

## Step 5: Search Your Content

### Web Search
1. Use the search box on the main page
2. Try searching for: "python functions"
3. View the results and click through to documents

### API Search
```bash
curl "http://localhost:3000/api/v1/search/?q=python%20functions"
```

## 🎯 Common Use Cases

### For Development Teams
1. **Add GitHub Repository**:
```bash
curl -X POST http://localhost:3000/api/v1/sources/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Project Wiki",
    "url": "https://github.com/yourorg/project/wiki",
    "type": "wiki"
  }'
```

2. **Add API Documentation**:
```bash
curl -X POST http://localhost:3000/api/v1/sources/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "API Docs",
    "url": "https://api.yourservice.com/docs",
    "type": "documentation"
  }'
```

### For Documentation Teams
1. **Add Multiple Doc Sites**:
```bash
# Add docs from config file
./scripts/import-sources.sh docs-config.json
```

2. **Set Up Automated Refreshes**:
   - Sources automatically refresh weekly
   - Manual refresh: Click "Refresh" button
   - Or via API: `curl -X POST http://localhost:3000/api/v1/sources/{id}/refresh`

### For Knowledge Management
1. **Search Across All Sources**:
   - Use the global search bar
   - Apply filters for specific sources
   - Use advanced search operators

2. **Monitor Source Health**:
   - Check dashboard for source status
   - Review failed crawls in the Jobs page
   - Set up alerts for crawl failures

## 📊 Monitoring Your Setup

### Health Dashboard
Visit: http://localhost:3000/api/health

Should show all services as "healthy":
- Database: ✅
- Cache: ✅
- Vector Store: ✅
- Storage: ✅

### Performance Monitoring
If you enabled monitoring during deployment:
- **Grafana**: http://localhost:3030
- **Prometheus**: http://localhost:9090

### Logs
```bash
# All services
./knowledgehub.sh logs

# Specific service
./knowledgehub.sh logs api
./knowledgehub.sh logs scraper
./knowledgehub.sh logs scheduler
```

## 🔧 Basic Configuration

### Environment Settings
Create `.env` file to customize:
```bash
# Basic settings
APP_ENV=production
DEBUG=false

# Database
DATABASE_URL=postgresql://khuser:khpassword@postgres:5432/knowledgehub

# Crawling
DEFAULT_CRAWL_DEPTH=3
MAX_CONCURRENT_CRAWLS=5
CRAWL_TIMEOUT=300

# Scheduling
SCHEDULER_ENABLED=true
REFRESH_SCHEDULE="0 2 * * 0"  # Sunday 2 AM
```

### Source Configuration
Common source types and their configs:

#### Documentation Sites
```json
{
  "name": "Service Docs",
  "url": "https://docs.service.com",
  "type": "documentation",
  "config": {
    "crawl_depth": 3,
    "file_types": ["html", "md"],
    "refresh_interval": 7
  }
}
```

#### GitHub Repositories
```json
{
  "name": "Project Repo",
  "url": "https://github.com/org/repo",
  "type": "repository",
  "config": {
    "include_wiki": true,
    "include_readme": true,
    "branch": "main"
  }
}
```

#### Wiki Systems
```json
{
  "name": "Company Wiki",
  "url": "https://wiki.company.com",
  "type": "wiki",
  "config": {
    "auth_required": true,
    "crawl_depth": 5
  }
}
```

## 🚨 Troubleshooting

### Services Won't Start
```bash
# Check Docker
docker --version
docker compose version

# Check ports
sudo lsof -i :3000
sudo lsof -i :3100

# Clean restart
./knowledgehub.sh stop
./knowledgehub.sh clean
./knowledgehub.sh start
```

### Source Crawling Fails
1. **Check URL is accessible**:
```bash
curl -I https://your-docs-url.com
```

2. **Review logs**:
```bash
./knowledgehub.sh logs scraper | grep ERROR
```

3. **Check source status**:
```bash
curl http://localhost:3000/api/v1/sources/ | jq '.sources[] | select(.status == "failed")'
```

### Search Returns No Results
1. **Check if crawling completed**:
   - Source status should be "completed"
   - Document count should be > 0

2. **Try broader search terms**:
   - Use single keywords instead of phrases
   - Check spelling

3. **Verify search service**:
```bash
curl http://localhost:8090/v1/.well-known/ready
```

### Performance Issues
1. **Check resource usage**:
```bash
docker stats --no-stream
```

2. **Optimize if needed**:
```bash
# Reduce concurrent crawls
# Increase memory limits
# Clean up old data
```

## 📚 Next Steps

### Learn More
- **User Guide**: [USER_GUIDE.md](USER_GUIDE.md)
- **Admin Guide**: [ADMIN_GUIDE.md](ADMIN_GUIDE.md)
- **API Documentation**: http://localhost:3000/docs
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)

### Advanced Features
- **Scheduled Refreshes**: Automatic source updates
- **Search Analytics**: Query statistics and insights
- **API Integration**: Programmatic access to all features
- **Monitoring**: Grafana dashboards and alerts
- **Backup/Recovery**: Automated data protection

### Get Support
- **Logs**: `./knowledgehub.sh logs`
- **Health Check**: http://localhost:3000/api/health
- **Documentation**: `/docs` directory
- **Issues**: Check GitHub issues for known problems

---

## 🎉 Success Checklist

After completing this guide, you should have:

- ✅ KnowledgeHub running at http://localhost:3100
- ✅ At least one knowledge source added and crawled
- ✅ Successful search results
- ✅ All services showing as healthy
- ✅ Understanding of basic operations

**Congratulations!** You now have a fully functional KnowledgeHub instance. Start adding your documentation sources and exploring the powerful search capabilities!

---

**Need help?** Check the [Troubleshooting](#troubleshooting) section above or review the [User Guide](USER_GUIDE.md) for detailed instructions.