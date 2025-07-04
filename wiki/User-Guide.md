# User Guide

Welcome to KnowledgeHub! This guide will help you get the most out of the platform, from basic usage to advanced features.

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
- Log messages appear as they happen

#### Job Details Include
- **Progress**: Current page / Total pages
- **Speed**: Pages per minute
- **Duration**: Time elapsed
- **Status Messages**: Real-time logs
- **Error Details**: If any issues occur

### Managing Jobs

#### Cancelling Jobs
1. Find the running job in the list
2. Click **"Cancel"** button
3. Job will stop at the next safe point

#### Retrying Failed Jobs
1. Find the failed job
2. Click **"Retry"** button
3. Job will restart from the beginning

#### Job History
- View all past jobs with filtering options
- Export job logs for debugging
- Track performance over time

## Memory System

### What is the Memory System?

The Memory System helps KnowledgeHub remember context across sessions, making interactions more intelligent and personalized.

### Using Memory

#### Automatic Memory
- Important conversations are automatically saved
- Search queries build understanding over time
- System learns from your usage patterns

#### Manual Memory Management
1. Go to **🧠 Memory** section
2. Add custom memories:
   ```
   Type: Decision
   Content: "Always use Python 3.11+ for new projects"
   Priority: High
   Tags: ["development", "standards"]
   ```

### Memory Types
- **Conversation**: Dialog and interactions
- **Decision**: Important choices made
- **Code**: Programming patterns and preferences
- **Error**: Issues and their solutions

### Memory Priority Levels
- **Critical**: Essential information
- **High**: Important context
- **Medium**: Useful background
- **Low**: Nice to have

## Advanced Features

### Scheduled Refreshes

#### Setting Up Automation
1. Go to **Settings** → **Automation**
2. Configure refresh schedule:
   ```
   Frequency: Weekly
   Day: Sunday
   Time: 02:00 AM
   Sources: All Active
   ```

### Bulk Operations

#### Bulk Import Sources
1. Prepare a CSV file with source configurations
2. Go to **Sources** → **Import**
3. Upload CSV and review
4. Start bulk crawling

#### Bulk Export
- Export all sources as JSON/CSV
- Export search results
- Export job history

### Webhooks Integration

#### Setting Up Webhooks
1. Go to **Settings** → **Integrations**
2. Add webhook endpoint:
   ```
   URL: https://your-app.com/webhook
   Events: ["job.completed", "job.failed"]
   Secret: your-secret-key
   ```

### API Access

#### Generating API Keys
1. Go to **Settings** → **API Keys**
2. Click **"Generate New Key"**
3. Name your key and set permissions
4. Copy the key (shown only once)

#### Using the API
```bash
curl -X POST "http://localhost:3000/api/v1/search" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "docker tutorial"}'
```

## Troubleshooting

### Common Issues

#### Search Returns No Results
1. Check if sources have been crawled
2. Verify search terms are spelled correctly
3. Try broader search terms
4. Check source status in Sources page

#### Crawling Stuck
1. Check job status in Jobs page
2. Look for error messages
3. Verify source URL is accessible
4. Check crawl configuration limits

#### Slow Performance
1. Reduce search result limit
2. Clear browser cache
3. Check system health in Dashboard
4. Contact administrator if persists

### Getting Help

#### Built-in Help
- Click **?** icon for contextual help
- Hover over settings for tooltips
- Check status messages for guidance

#### Support Resources
- [FAQ](FAQ) - Frequently asked questions
- [Troubleshooting Guide](Troubleshooting) - Detailed solutions
- [Community Forum](https://github.com/anubissbe/knowledgehub/discussions)
- [GitHub Issues](https://github.com/anubissbe/knowledgehub/issues)

## Best Practices

### Source Management
1. **Start Small**: Begin with a few key sources
2. **Test Configuration**: Use low limits initially
3. **Monitor Performance**: Check crawl times and sizes
4. **Regular Updates**: Schedule weekly refreshes

### Search Optimization
1. **Use Hybrid Search**: Best balance of results
2. **Refine Queries**: Iterate based on results
3. **Save Useful Queries**: Build a library
4. **Learn Patterns**: Understand what works

### System Maintenance
1. **Regular Monitoring**: Check dashboard weekly
2. **Clean Old Data**: Remove outdated sources
3. **Update Configuration**: Adjust based on usage
4. **Backup Important Data**: Export configurations

### Security Best Practices
1. **Secure API Keys**: Never share or commit keys
2. **Regular Updates**: Keep system updated
3. **Access Control**: Limit who can manage sources
4. **Monitor Usage**: Check for unusual activity

## Keyboard Shortcuts

- `Ctrl/Cmd + K`: Quick search
- `Ctrl/Cmd + /`: Toggle help
- `Esc`: Close dialogs
- `?`: Show keyboard shortcuts

## Next Steps

Now that you understand the basics:

1. [Add your first source](#adding-a-new-source)
2. [Try searching](#basic-search)
3. [Explore advanced features](#advanced-features)
4. Read [Best Practices](#best-practices)

For developers:
- [API Documentation](API-Documentation)
- [Configuration Guide](Configuration)
- [Architecture Overview](Architecture)