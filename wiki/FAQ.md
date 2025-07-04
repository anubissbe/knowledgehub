# Frequently Asked Questions

## General Questions

### What is KnowledgeHub?

KnowledgeHub is an AI-powered knowledge management system that helps organizations capture, process, and search through their documentation intelligently. It uses advanced natural language processing and vector embeddings to provide semantic search capabilities across all your knowledge sources.

### How is it different from traditional search?

Unlike traditional keyword-based search, KnowledgeHub:
- **Understands meaning**: Uses AI to understand the context and intent of your queries
- **Learns continuously**: Improves search results based on usage patterns
- **Finds related content**: Discovers conceptually similar information even if exact keywords don't match
- **Maintains freshness**: Automatically updates content with 95%+ faster incremental crawling

### What types of content can it index?

KnowledgeHub currently supports:
- **Web pages**: Any publicly accessible website or internal documentation site
- **HTML content**: Static or dynamically generated pages
- **JavaScript-rendered sites**: Full support for modern SPAs
- Future: PDFs, Word documents, Confluence pages, and more

### Is it open source?

Yes! KnowledgeHub is open-source software released under the MIT license. You can freely use, modify, and distribute it for both commercial and non-commercial purposes.

## Installation & Setup

### What are the system requirements?

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 20GB free space
- Docker 24.0+ with Docker Compose

**Recommended for Production:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD
- GPU: Optional, improves embedding generation speed

### How long does installation take?

- **Quick Start (Docker)**: 5-10 minutes
- **Development Setup**: 20-30 minutes
- **Production Deployment**: 1-2 hours including configuration

### Can I run it on Windows/Mac?

Yes! KnowledgeHub runs on:
- Linux (all distributions)
- macOS (Intel and Apple Silicon)
- Windows 10/11 (via Docker Desktop or WSL2)

### Do I need a GPU?

No, GPU is optional. The system works perfectly on CPU, though GPU acceleration can improve embedding generation speed by 3-5x for large deployments.

## Usage & Features

### How do I add a new knowledge source?

1. Navigate to the Sources page
2. Click "Add Source"
3. Enter the URL and configuration
4. The system automatically starts crawling

See the [User Guide](User-Guide) for detailed instructions.

### How often should I refresh sources?

We recommend:
- **High-change content**: Daily or twice weekly
- **Documentation sites**: Weekly
- **Static content**: Monthly

The scheduler can automate this based on your preferences.

### What is incremental crawling?

Incremental crawling is our breakthrough feature that:
- Detects which pages have changed since last crawl
- Only processes new or modified content
- Achieves 95%+ faster updates
- Works automatically with no configuration

See [Incremental Crawling Guide](Incremental-Crawling) for details.

### How accurate is the search?

Search accuracy depends on:
- **Content quality**: Better source content = better results
- **Query clarity**: More specific queries yield better matches
- **Search type**: Hybrid search (default) provides best results

Typical relevance scores:
- 0.8-1.0: Excellent match
- 0.6-0.8: Good match
- 0.4-0.6: Moderate match

### Can I search across multiple sources?

Yes! By default, searches span all active sources. You can also:
- Filter by specific sources
- Exclude certain sources
- Create source groups for targeted searching

## Performance & Scaling

### How many documents can it handle?

KnowledgeHub has been tested with:
- **Single instance**: Up to 1 million documents
- **Clustered setup**: 10+ million documents
- **Search performance**: Sub-second for most queries

### How fast is crawling?

Crawling speed varies by site:
- **Typical**: 10-50 pages per minute
- **Fast sites**: Up to 100 pages per minute
- **Rate-limited sites**: 1-10 pages per minute

Incremental updates are 95%+ faster.

### Does it respect robots.txt?

Yes, KnowledgeHub:
- Honors robots.txt directives
- Implements configurable crawl delays
- Supports custom user agent strings
- Follows politeness policies

### How much storage does it need?

Storage requirements:
- **PostgreSQL**: ~1KB per document metadata
- **Weaviate**: ~2KB per vector embedding
- **MinIO**: Original content size (optional)
- **Total**: Approximately 3-5KB per document

Example: 100,000 documents ≈ 300-500MB

## Troubleshooting

### Search returns no results

Common causes:
1. **Sources not crawled yet**: Check Jobs page for status
2. **Services not running**: Verify all services are healthy
3. **Query too specific**: Try broader terms
4. **No matching content**: The information might not exist in sources

### Crawling is slow

Solutions:
1. **Reduce crawl delay**: Lower delay in source configuration
2. **Check rate limits**: Some sites limit request frequency
3. **Scale workers**: Add more scraper instances
4. **Use incremental**: Enable incremental crawling for updates

### High memory usage

Causes and fixes:
1. **Large crawls**: Reduce max_pages limit
2. **Memory leaks**: Restart services periodically
3. **Cache size**: Adjust Redis memory limits
4. **Too many workers**: Reduce concurrent workers

### Services won't start

Check:
1. **Port conflicts**: Ensure ports 3000, 3101, 8080, 9000 are free
2. **Docker resources**: Increase Docker memory allocation
3. **Disk space**: Ensure adequate free space
4. **Permissions**: Check file and directory permissions

## Security & Privacy

### Is my data secure?

KnowledgeHub provides:
- **Local deployment**: All data stays on your infrastructure
- **No external calls**: Fully self-contained system
- **Access control**: API key authentication
- **Encrypted storage**: Support for encryption at rest

### Does it send data externally?

No, KnowledgeHub:
- Runs entirely on your infrastructure
- Makes no external API calls
- Doesn't phone home
- Keeps all data local

### Can I restrict access?

Yes, through:
- API key authentication
- Network isolation
- Reverse proxy with auth
- Source-level permissions (roadmap)

### Is it GDPR compliant?

KnowledgeHub can be configured for GDPR compliance:
- All data stored locally
- Full data deletion capabilities
- No third-party data sharing
- Audit logging available

## Advanced Topics

### Can I customize the embedding model?

Yes! See [Tutorial 7](Tutorials#tutorial-7-custom-embedding-workflows) for:
- Using different models
- Fine-tuning for your domain
- Multi-model ensembles
- Custom embedding services

### Does it support multiple languages?

Currently:
- **UI**: English only
- **Content**: Any language (best results with English)
- **Search**: Works with any language, optimized for English

For multilingual support, use appropriate embedding models.

### Can I integrate with my existing tools?

Yes, through:
- **REST API**: Full programmatic access
- **Webhooks**: Event notifications
- **Export capabilities**: JSON, CSV formats
- **Custom integrations**: Via API

### How do I monitor system health?

Built-in monitoring includes:
- Health check endpoints
- Prometheus metrics
- Performance dashboards
- Log aggregation

See [Monitoring Guide](Monitoring) for details.

## Development & Contributing

### How can I contribute?

We welcome contributions! You can:
- Report bugs and request features
- Submit pull requests
- Improve documentation
- Share your use cases

See [Contributing Guidelines](Contributing) for details.

### What's the technology stack?

- **Backend**: Python, FastAPI
- **Frontend**: React, TypeScript, Vite
- **Databases**: PostgreSQL, Redis, Weaviate
- **AI/ML**: sentence-transformers, Playwright
- **Infrastructure**: Docker, Docker Compose

### Where can I get help?

- **Documentation**: This wiki
- **GitHub Issues**: Bug reports and features
- **Discussions**: Community forum
- **Email**: support@knowledgehub.dev (coming soon)

### What's on the roadmap?

Planned features:
- PDF and document support
- Real-time collaboration
- Advanced analytics
- Multi-tenant support
- Mobile applications

## Licensing & Commercial Use

### Can I use it commercially?

Yes! The MIT license allows:
- Commercial use
- Modification
- Distribution
- Private use

### Do I need to share my changes?

No, the MIT license doesn't require you to share modifications. However, we encourage contributing improvements back to the community.

### Is commercial support available?

Not currently, but planned for the future. The community provides excellent support through GitHub issues and discussions.

---

**Didn't find your answer?** 

- Search the [documentation](Home)
- Check [GitHub Issues](https://github.com/anubissbe/knowledgehub/issues)
- Ask in [Discussions](https://github.com/anubissbe/knowledgehub/discussions)