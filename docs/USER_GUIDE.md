# KnowledgeHub User Guide

## Table of Contents

- [Getting Started](#getting-started)
- [Managing Knowledge Sources](#managing-knowledge-sources)
- [Searching Content](#searching-content)
- [Monitoring Jobs](#monitoring-jobs)
- [Memory System](#memory-system)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Getting Started

### First Login

1. **Access the Web Interface**
   - Open your browser and navigate to `http://localhost:3101`
   - You should see the KnowledgeHub dashboard

2. **Dashboard Overview**
   The dashboard provides:
   - **System Health**: Current status of all services
   - **Recent Activity**: Latest crawling jobs and their status
   - **Quick Stats**: Sources, documents, and search metrics
   - **Performance Metrics**: Response times and system health

### Navigation

The main navigation includes:
- **🏠 Dashboard** - System overview and health
- **📚 Sources** - Manage knowledge sources
- **🔍 Search** - Find information across all sources
- **⚙️ Jobs** - Monitor background tasks
- **🧠 Memory** - Conversation context management
- **⚙️ Settings** - System configuration

## Managing Knowledge Sources

### Adding a New Source

1. **Navigate to Sources**
   - Click on **📚 Sources** in the main navigation
   - Click the **"+ Add Source"** button

2. **Fill in Source Details**
   ```
   Name: GitHub Documentation
   Description: Official GitHub documentation and guides
   Base URL: https://docs.github.com
   Source Type: Web
   ```

3. **Configure Crawling Settings**
   - **Max Depth**: How deep to crawl (recommended: 3-5)
   - **Max Pages**: Maximum pages to process (recommended: 1000-5000)
   - **Crawl Delay**: Delay between requests in seconds (recommended: 1.0)
   - **Follow Patterns**: URL patterns to include (e.g., `**` for all)
   - **Exclude Patterns**: URL patterns to exclude (e.g., `**/admin/**`)

4. **Start Crawling**
   - Click **"Create Source"**
   - The system will automatically start crawling
   - Monitor progress in the **Jobs** section

### Source Configuration Examples

#### Documentation Sites
```json
{
  "max_depth": 4,
  "max_pages": 2000,
  "crawl_delay": 1.0,
  "follow_patterns": ["**/docs/**", "**/guide/**"],
  "exclude_patterns": ["**/admin/**", "**/private/**", "**/*.pdf"]
}
```

#### API Documentation
```json
{
  "max_depth": 3,
  "max_pages": 500,
  "crawl_delay": 0.5,
  "follow_patterns": ["**/api/**", "**/reference/**"],
  "exclude_patterns": ["**/v1/**", "**/deprecated/**"]
}
```

#### Knowledge Base
```json
{
  "max_depth": 5,
  "max_pages": 5000,
  "crawl_delay": 1.5,
  "follow_patterns": ["**"],
  "exclude_patterns": ["**/search/**", "**/login/**", "**/admin/**"]
}
```

### Managing Existing Sources

#### Viewing Source Details
1. Go to **Sources** page
2. Click on any source name to view details
3. See crawling history, statistics, and configuration

#### Updating Sources
1. Click the **"Edit"** button next to a source
2. Modify configuration as needed
3. Click **"Update Source"**
4. Choose whether to trigger a new crawl

#### Refreshing Content
1. Click **"Refresh"** next to a source
2. The system will use **incremental crawling** by default
3. Only new or changed content will be processed
4. Monitor progress in the **Jobs** section

#### Deleting Sources
1. Click **"Delete"** next to a source
2. Confirm the deletion
3. All associated documents and chunks will be removed

## Searching Content

### Basic Search

1. **Navigate to Search**
   - Click **🔍 Search** in the main navigation

2. **Enter Your Query**
   ```
   How to deploy a Docker container
   ```

3. **Search Types**
   - **Hybrid** (Recommended): Combines semantic and keyword search
   - **Semantic**: AI-powered meaning-based search
   - **Keyword**: Traditional text matching

### Advanced Search Features

#### Search Filters
```
Query: "authentication methods"
Source: GitHub Documentation
Date Range: Last 30 days
Content Type: All
```

#### Search Operators
- **Exact phrases**: `"exact phrase"`
- **Exclude terms**: `authentication -oauth`
- **Required terms**: `+docker +deployment`
- **Wildcards**: `deploy*` (matches deploy, deployment, deploying)

### Understanding Search Results

Each search result shows:
- **Title**: Document title and source
- **Relevance Score**: How well it matches your query
- **Summary**: Key content snippet
- **Source**: Which knowledge source it came from
- **URL**: Direct link to original content
- **Last Updated**: When the content was last crawled

### Search Tips

#### For Best Results
1. **Use natural language**: "How to configure SSL certificates"
2. **Be specific**: Instead of "error", use "database connection error"
3. **Try different phrasings**: "setup guide" vs "installation instructions"
4. **Use context**: "React hooks useState" vs just "useState"

#### Semantic Search Examples
- **Question format**: "What is the difference between Docker and Kubernetes?"
- **Task-based**: "How to set up CI/CD pipeline with GitHub Actions"
- **Concept exploration**: "Best practices for API security"

## Monitoring Jobs

### Job Dashboard

1. **Navigate to Jobs**
   - Click **⚙️ Jobs** in the main navigation

2. **Job Status Overview**
   - **Pending**: Waiting to be processed
   - **Running**: Currently being executed
   - **Completed**: Successfully finished
   - **Failed**: Encountered an error
   - **Cancelled**: Manually stopped

### Real-time Progress

#### Live Updates
- Jobs update in real-time via WebSocket connection
- Progress bars show completion percentage
- Detailed status messages explain current activity

#### Progress Indicators
```
Status: Running
Progress: 75% (750/1000 pages)
Current Activity: Processing https://docs.example.com/api/auth
Elapsed Time: 5m 30s
Estimated Remaining: 1m 45s
```

### Job Types

#### Crawl Jobs
- **Initial Crawl**: First-time processing of a source
- **Incremental Crawl**: Updates using delta detection
- **Full Refresh**: Complete re-processing of all content

#### Processing Jobs
- **RAG Processing**: Converting crawled content to searchable chunks
- **Embedding Generation**: Creating vector representations
- **Index Updates**: Updating search indexes

#### Maintenance Jobs
- **Cleanup**: Removing orphaned data
- **Optimization**: Index compaction and optimization
- **Backup**: Data backup operations

### Managing Jobs

#### Viewing Job Details
1. Click on any job ID to see detailed information
2. View logs, errors, and processing statistics
3. See which pages were processed or skipped

#### Cancelling Jobs
1. Click **"Cancel"** next to a running job
2. The job will stop gracefully after completing current operations
3. Partial progress is preserved

#### Retrying Failed Jobs
1. Click **"Retry"** next to a failed job
2. The system will attempt to process only the failed portions
3. Previous successful work is preserved

## Memory System

### Understanding Memory

The memory system helps maintain context across conversations and sessions:

- **Conversation Memory**: Tracks discussion context
- **Decision Memory**: Records important choices made
- **Code Memory**: Remembers code patterns and solutions
- **Error Memory**: Learns from past issues

### Using Memory Features

#### Viewing Memory
1. Navigate to **🧠 Memory** section
2. Browse memory items by type and priority
3. Search through stored contexts

#### Memory Types
- **Critical**: Essential system decisions
- **High**: Important technical choices
- **Medium**: Useful reference information
- **Low**: General context
- **Trivial**: Basic operational data

#### Memory Operations
```bash
# Add important information
./memory-cli add "API endpoint changed to v2" -t decision -p high

# Search memory
./memory-cli search -t code --contains "authentication"

# Get context for current session
./memory-cli context --max-tokens 4000
```

## Advanced Features

### Incremental Crawling

#### How It Works
1. **Content Hashing**: Each page gets a SHA-256 hash
2. **Change Detection**: Compares current hash with stored hash
3. **Selective Processing**: Only processes changed content
4. **Link Discovery**: Still follows links to find new pages

#### Performance Benefits
- **95%+ faster updates** for existing content
- **Reduced server load** on target websites
- **Efficient resource usage** for your system
- **Automatic optimization** with no configuration needed

#### Monitoring Incremental Crawls
```
Incremental Crawl Results:
- Pages Checked: 1,838
- Unchanged: 1,826 (99.3%)
- Updated: 8 (0.4%)
- New: 4 (0.2%)
- Time Saved: 24m 45s (97.2% faster)
```

### Automated Scheduling

#### Weekly Refresh
- Sources are automatically refreshed weekly
- Uses incremental crawling by default
- Scheduled during low-usage periods (2 AM Sunday)
- Intelligent batching to distribute load

#### Configuration
```json
{
  "scheduler": {
    "enabled": true,
    "refresh_schedule": "0 2 * * 0",
    "batch_size": 5,
    "delay_between_batches": 300
  }
}
```

### API Integration

#### Authentication
```bash
# Get API key from settings
export API_KEY="your-api-key-here"

# Make authenticated requests
curl -H "X-API-Key: $API_KEY" http://localhost:3000/api/v1/sources
```

#### Common API Operations
```bash
# Create a source
curl -X POST http://localhost:3000/api/v1/sources \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "name": "My Documentation",
    "base_url": "https://docs.example.com",
    "source_type": "web"
  }'

# Search content
curl -X POST http://localhost:3000/api/v1/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "query": "installation guide",
    "search_type": "hybrid",
    "limit": 10
  }'
```

## Troubleshooting

### Common Issues

#### Source Won't Crawl
**Symptoms**: Job stays in "Pending" status
**Solutions**:
1. Check that the URL is accessible
2. Verify crawl configuration is valid
3. Ensure no rate limiting is blocking requests
4. Check for JavaScript requirements

#### Search Returns No Results
**Symptoms**: Search queries return empty results
**Solutions**:
1. Verify sources have been successfully crawled
2. Check that RAG processing jobs completed
3. Try different search terms or types
4. Ensure Weaviate service is running

#### Slow Performance
**Symptoms**: Searches or crawls are slow
**Solutions**:
1. Check system resource usage
2. Verify database indexes are in place
3. Clear cache if needed
4. Consider scaling worker processes

#### Jobs Keep Failing
**Symptoms**: Repeated job failures
**Solutions**:
1. Check error messages in job details
2. Verify network connectivity to target sites
3. Adjust crawl delays if rate limited
4. Check system resource availability

### Getting Help

#### System Health Check
```bash
# Check overall system health
curl http://localhost:3000/health

# Check specific services
curl http://localhost:3000/health/database
curl http://localhost:3000/health/redis
curl http://localhost:3000/health/weaviate
```

#### Log Investigation
```bash
# View API logs
docker compose logs -f api

# View scraper logs
docker compose logs -f scraper

# View all service logs
docker compose logs -f
```

#### Performance Monitoring
```bash
# Check container resource usage
docker stats

# Check queue depths
docker exec knowledgehub-redis redis-cli llen crawl_jobs:pending

# Check database performance
docker exec knowledgehub-postgres pg_stat_activity
```

## Best Practices

### Source Management

#### Choosing Good Sources
1. **Authoritative content**: Official documentation, established knowledge bases
2. **Well-structured sites**: Clear navigation, consistent formatting
3. **Regularly updated**: Active maintenance and fresh content
4. **Accessible content**: Public, non-login-required pages

#### Configuration Guidelines
1. **Start conservative**: Begin with lower max_pages and max_depth
2. **Respect rate limits**: Use appropriate crawl delays
3. **Use exclude patterns**: Skip non-content pages (admin, search, etc.)
4. **Monitor resource usage**: Watch for memory and CPU spikes

### Search Optimization

#### Query Best Practices
1. **Use natural language**: Write queries as questions or statements
2. **Be specific**: Include relevant context and details
3. **Try multiple approaches**: Rephrase if first search doesn't work
4. **Use filters**: Narrow down by source or date when helpful

#### Result Evaluation
1. **Check relevance scores**: Higher scores indicate better matches
2. **Review source URLs**: Verify content comes from expected sources
3. **Read full context**: Click through to original content when needed
4. **Provide feedback**: Note which results were most helpful

### Performance Optimization

#### System Maintenance
1. **Regular health checks**: Monitor system status dashboard
2. **Clean up old jobs**: Remove completed jobs periodically
3. **Monitor disk usage**: Ensure adequate storage space
4. **Update regularly**: Keep system components current

#### Scaling Considerations
1. **Worker scaling**: Add more workers for heavy crawling loads
2. **Database optimization**: Monitor query performance and add indexes
3. **Cache tuning**: Adjust Redis memory limits and eviction policies
4. **Resource monitoring**: Track CPU, memory, and disk usage trends

This user guide provides comprehensive information for effectively using KnowledgeHub. For additional technical details, refer to the API documentation and architecture guides.