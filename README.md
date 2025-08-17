# 🧠 KnowledgeHub

> **AI-Enhanced Development Intelligence Platform**

KnowledgeHub gives your AI coding tools persistent memory and intelligent context. It remembers your decisions, learns from your mistakes, and provides relevant suggestions across all your development sessions.

## 🎯 What It Does

**Memory & Learning**
- Remembers your development patterns and decisions
- Learns from errors and successful solutions
- Maintains context across coding sessions
- Tracks project evolution and code changes

**AI Enhancement**
- Works with Claude Code, GitHub Copilot, Cursor, and any AI tool
- Provides intelligent context injection
- Suggests relevant code patterns and solutions
- Analyzes and improves development workflows

**Enterprise Features**
- Vector search across all your code and documentation
- Real-time analytics and performance monitoring
- Multi-project isolation and team collaboration
- Advanced RAG (Retrieval-Augmented Generation) systems

## 🚀 Quick Start

```bash
# Clone and start
git clone https://github.com/anubissbe/knowledgehub.git
cd knowledgehub
docker-compose up -d

# Verify it's running
curl http://localhost:3000/health
```

**Web UI**: http://localhost:3100  
**API Docs**: http://localhost:3000/docs

## 🤖 Integration

### Claude Code
```bash
# Add to your project
source /path/to/knowledgehub/kh_code_helpers.sh
kh_setup
```

### Any AI Tool
```python
import requests

# Simple API integration
api = "http://localhost:3000"
response = requests.post(f"{api}/api/memory", json={
    "content": "Fixed auth bug with null check",
    "context": "authentication module"
})
```

## ✅ What KnowledgeHub IS

- **Universal AI Backend**: Works with any AI coding assistant
- **Memory System**: Persistent context across tools and sessions
- **Learning Platform**: Continuously improves from your patterns
- **Development Analytics**: Tracks and optimizes your workflow
- **Local-First**: Runs on your infrastructure, your data stays private

## ❌ What KnowledgeHub is NOT

- **Not an AI Model**: Doesn't generate code itself
- **Not a Code Editor**: Works with your existing tools
- **Not Cloud-Dependent**: Runs locally on your machine
- **Not Tool-Specific**: Not limited to any single AI assistant

## 🏗️ Architecture

```
Your AI Tools (Claude, Copilot, Cursor, etc.)
                    ↓
            KnowledgeHub API
                    ↓
    Memory • Analytics • Context • Learning
                    ↓
   PostgreSQL • Vector DB • Knowledge Graph
```

## 📚 Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Complete setup instructions
- **[AI Integration](docs/AI_INTEGRATION.md)** - Connect your AI tools
- **[API Documentation](http://localhost:3000/docs)** - REST API reference
- **[Enterprise Setup](docs/ENTERPRISE.md)** - Production deployment

## 🔧 Requirements

- Docker & Docker Compose
- 4GB+ RAM
- Python 3.11+ (for integrations)

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Made for developers who want smarter AI coding assistants** 🚀