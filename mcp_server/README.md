# KnowledgeHub MCP Server

A Model Context Protocol (MCP) server that provides Claude Code with direct access to KnowledgeHub's AI-enhanced memory and intelligence systems.

## Features

### 🧠 AI-Enhanced Memory Operations
- **Semantic Search**: Find relevant memories using AI-powered similarity search
- **Smart Creation**: Automatic embedding generation and content analysis
- **Context Awareness**: Project and session-based memory organization
- **Rich Metadata**: Tags, types, and custom metadata support

### 🔄 Session Management & Continuity
- **Context Restoration**: Resume work exactly where you left off
- **Smart Sessions**: AI-enhanced session tracking and management
- **Cross-Session Learning**: Learn patterns across different coding sessions
- **Automatic Handoffs**: Seamless context transfer between sessions

### 🤖 AI Intelligence Features
- **Task Prediction**: AI predicts your next likely tasks based on context
- **Pattern Analysis**: Analyze code, usage, and error patterns using ML
- **Decision Tracking**: Record and learn from technical decisions
- **Error Learning**: Track errors and solutions for future reference
- **Intelligent Insights**: Get AI-generated recommendations and insights

### 📊 Real-time Analytics & Monitoring
- **Live Metrics**: Access real-time system and application metrics
- **Dashboard Data**: Comprehensive performance and usage dashboards
- **Alert Management**: Monitor and respond to system alerts
- **Performance Reports**: Generate detailed performance analysis reports

### 🔧 Utility & Integration
- **Context Synchronization**: Bidirectional context sync with Claude Code
- **System Health**: Monitor system status and perform health checks
- **API Information**: Discover available APIs and endpoints

## Tool Categories

### Memory Tools (5 tools)
- `create_memory` - Create AI-enhanced memories
- `search_memories` - Semantic search across memories
- `get_memory` - Retrieve specific memories with context
- `update_memory` - Update existing memories
- `get_memory_stats` - Memory system statistics

### Session Tools (5 tools)
- `init_session` - Initialize AI-enhanced sessions
- `get_session` - Get current session information
- `update_session_context` - Update session context
- `end_session` - End sessions with context preservation
- `get_session_history` - Session history and analytics

### AI Tools (6 tools)
- `predict_next_tasks` - AI task predictions
- `analyze_patterns` - ML-powered pattern analysis
- `get_ai_insights` - AI-generated insights and recommendations
- `record_decision` - Track technical decisions
- `track_error` - Learn from errors and solutions

### Analytics Tools (4 tools)
- `get_metrics` - Real-time metrics access
- `get_dashboard_data` - Dashboard and visualization data
- `get_alerts` - Alert monitoring and management
- `get_performance_report` - Performance analysis reports

### Utility Tools (4 tools)
- `sync_context` - Context synchronization
- `get_system_status` - System health monitoring
- `health_check` - Comprehensive health checks
- `get_api_info` - API discovery and documentation

## Installation & Setup

### Prerequisites

1. **KnowledgeHub Environment**: Ensure KnowledgeHub is running and accessible
2. **Python 3.8+**: Required for async support
3. **MCP Library**: Install the Model Context Protocol library

### Quick Start

1. **Install Dependencies**:
   ```bash
   cd /opt/projects/knowledgehub/mcp_server
   pip install -r requirements.txt
   ```

2. **Configure Claude Desktop**:
   Add the following to your Claude Desktop MCP configuration:
   ```json
   {
     "mcpServers": {
       "knowledgehub": {
         "command": "python",
         "args": ["/opt/projects/knowledgehub/mcp_server/start_server.py"],
         "env": {
           "PYTHONPATH": "/opt/projects/knowledgehub"
         }
       }
     }
   }
   ```

3. **Start the Server**:
   ```bash
   python start_server.py
   ```

4. **Test Connection**:
   ```bash
   python start_server.py --validate-only
   ```

### Configuration

The server uses `config.json` for configuration. Key settings:

```json
{
  "knowledgehub": {
    "api_base_url": "http://localhost:3000",
    "default_user_id": "claude_code",
    "session_timeout_minutes": 30
  },
  "tools": {
    "memory": {
      "enabled": true,
      "default_similarity_threshold": 0.7,
      "max_search_results": 50
    }
  }
}
```

## Usage Examples

### Memory Operations

**Create a Memory**:
```json
{
  "tool": "create_memory",
  "arguments": {
    "content": "Implemented OAuth2 authentication with JWT tokens",
    "memory_type": "code",
    "project_id": "my-project",
    "tags": ["auth", "security", "implementation"]
  }
}
```

**Search Memories**:
```json
{
  "tool": "search_memories",
  "arguments": {
    "query": "authentication implementation",
    "limit": 5,
    "similarity_threshold": 0.7
  }
}
```

### Session Management

**Initialize Session**:
```json
{
  "tool": "init_session",
  "arguments": {
    "session_type": "coding",
    "project_id": "my-project",
    "context_data": {
      "current_file": "auth.py",
      "working_on": "OAuth implementation"
    }
  }
}
```

### AI Intelligence

**Predict Next Tasks**:
```json
{
  "tool": "predict_next_tasks",
  "arguments": {
    "context": "Just implemented user authentication",
    "project_id": "my-project",
    "num_predictions": 5
  }
}
```

**Analyze Patterns**:
```json
{
  "tool": "analyze_patterns",
  "arguments": {
    "data": "class UserAuth:\n    def login(self, user, password):\n        # implementation",
    "analysis_type": "code",
    "project_id": "my-project"
  }
}
```

### Analytics

**Get Real-time Metrics**:
```json
{
  "tool": "get_metrics",
  "arguments": {
    "metric_names": ["response_time", "error_rate", "memory_usage"],
    "time_window": "1h",
    "aggregation": "avg"
  }
}
```

**Generate Performance Report**:
```json
{
  "tool": "get_performance_report",
  "arguments": {
    "report_type": "system",
    "time_range": "24h",
    "include_trends": true
  }
}
```

## Architecture

### Server Components

1. **KnowledgeHubMCPServer**: Main server class handling MCP protocol
2. **ToolRegistry**: Dynamic tool registration and management
3. **Handlers**: Specialized handlers for each tool category
4. **ResponseFormatter**: Consistent response formatting

### Integration Points

- **Memory Service**: Direct integration with KnowledgeHub's memory system
- **Session Service**: Session management and context preservation
- **AI Services**: Prediction, pattern analysis, and intelligence features
- **Analytics Service**: Real-time metrics and performance monitoring
- **WebSocket Events**: Real-time updates and notifications

### Security & Performance

- **Authentication**: Optional authentication support
- **Rate Limiting**: Configurable rate limiting per client
- **Caching**: Response caching for improved performance
- **Timeout Protection**: Tool execution timeout protection
- **Error Handling**: Comprehensive error handling and logging

## Development

### Adding New Tools

1. **Define Tool**: Add tool definition to appropriate category in `tools.py`
2. **Implement Handler**: Add method to appropriate handler in `handlers.py`
3. **Register Tool**: Tool is automatically registered via category definitions
4. **Test**: Use the validation script to test the new tool

### Testing

```bash
# Validate environment
python start_server.py --validate-only

# Test with debug logging
python start_server.py --log-level DEBUG

# Test specific configuration
python start_server.py --config test_config.json
```

### Debugging

- **Log Files**: Server logs to console and optionally to file
- **Health Checks**: Built-in health checking and diagnostics
- **Tool Validation**: Individual tool testing and validation
- **Context Tracing**: Full context and execution tracing

## Integration with Claude Code

### Setup in Claude Desktop

1. Copy `claude_desktop_config.json` example
2. Modify paths to match your installation
3. Restart Claude Desktop
4. Tools will appear in Claude Code interface

### Best Practices

1. **Initialize Sessions**: Always start with `init_session` for best context
2. **Use Semantic Search**: Leverage AI-powered memory search
3. **Track Decisions**: Record important technical decisions
4. **Monitor Performance**: Regular health checks and metrics review
5. **Context Sync**: Use bidirectional context synchronization

### Troubleshooting

**Server Won't Start**:
- Check Python version (3.8+ required)
- Verify KnowledgeHub API is accessible
- Review configuration file syntax

**Tools Not Working**:
- Check server logs for errors
- Verify tool arguments match schema
- Test with `health_check` tool

**Poor Performance**:
- Enable caching in configuration
- Adjust timeout settings
- Monitor system resources

## Support

For issues and questions:

1. Check server logs for error details
2. Use `health_check` tool for diagnostics
3. Validate environment with `--validate-only`
4. Review configuration settings

## License

Part of the KnowledgeHub AI-enhanced development environment.