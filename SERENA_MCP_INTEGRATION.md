# Serena MCP Integration - Code Intelligence for KnowledgeHub

## Overview

KnowledgeHub now includes **Serena-inspired code intelligence** capabilities, providing semantic code understanding, symbol analysis, and project-aware memory for Claude Code integration.

## What's Been Integrated

### ✅ Code Intelligence Service
- **File**: `/api/services/code_intelligence_service.py`
- **Purpose**: Core semantic code analysis engine
- **Features**:
  - Language detection (Python, TypeScript, JavaScript, Rust, Go)
  - Framework detection (FastAPI, Django, React, Vue, etc.)
  - LSP (Language Server Protocol) integration
  - Symbol extraction (classes, functions, methods)
  - Dependency analysis
  - Project-aware memory system

### ✅ REST API Endpoints
- **File**: `/api/routers/code_intelligence.py`
- **Base URL**: `http://192.168.1.25:3000/api/code-intelligence`
- **Authentication**: Exempt from API key requirement

### ✅ Available Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Service health check |
| `/activate-project` | POST | Initialize project analysis |
| `/symbols/overview` | GET | Get symbol overview |
| `/symbols/find` | POST | Find specific symbol |
| `/symbols/replace` | POST | Replace symbol body |
| `/search/pattern` | POST | Search for patterns |
| `/memory/save` | POST | Save project memory |
| `/memory/load` | GET | Load project memory |
| `/memory/list` | GET | List project memories |

### ✅ MCP Server Implementation
- **File**: `/api/services/code_intelligence_mcp.py`
- **Configuration**: `/opt/projects/knowledgehub/code-intelligence-mcp.json`
- **Tools**: 11 code intelligence tools available to Claude Code

## How to Use with Claude Code

### Option 1: Direct REST API Integration
The code intelligence service is available as REST endpoints and can be called directly:

```bash
# Activate a project
curl -X POST http://192.168.1.25:3000/api/code-intelligence/activate-project \
  -H "Content-Type: application/json" \
  -d '{"project_path": "/opt/projects/knowledgehub"}'

# Get symbols overview
curl -G http://192.168.1.25:3000/api/code-intelligence/symbols/overview \
  --data-urlencode "project_path=/opt/projects/knowledgehub" \
  --data-urlencode "file_path=/opt/projects/knowledgehub/api/main.py"
```

### Option 2: MCP Server Connection
For direct Claude Code integration, use the MCP server configuration:

#### Claude Desktop Configuration
Add to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "knowledgehub-code-intelligence": {
      "command": "python3",
      "args": [
        "/opt/projects/knowledgehub/api/services/code_intelligence_mcp.py"
      ],
      "env": {
        "PYTHONPATH": "/opt/projects/knowledgehub"
      }
    }
  }
}
```

## Available MCP Tools

### 1. **activate_project**
```json
{
  "name": "activate_project",
  "description": "Activate a project for code intelligence analysis",
  "parameters": {
    "project_path": "string (required)"
  }
}
```

### 2. **get_symbols_overview**
```json
{
  "name": "get_symbols_overview", 
  "description": "Get an overview of code symbols in a file or project",
  "parameters": {
    "relative_path": "string (optional)"
  }
}
```

### 3. **find_symbol**
```json
{
  "name": "find_symbol",
  "description": "Find a specific symbol by name or path",
  "parameters": {
    "name_path": "string (required)",
    "include_body": "boolean (default: false)",
    "include_references": "boolean (default: false)"
  }
}
```

### 4. **replace_symbol**
```json
{
  "name": "replace_symbol",
  "description": "Replace a symbol's body with new code",
  "parameters": {
    "symbol_path": "string (required)",
    "new_body": "string (required)"
  }
}
```

### 5. **search_pattern**
```json
{
  "name": "search_pattern",
  "description": "Search for a pattern in project files",
  "parameters": {
    "pattern": "string (required)",
    "file_pattern": "string (optional, e.g., '*.py')",
    "context_lines": "integer (default: 2)"
  }
}
```

### 6. **write_memory**
```json
{
  "name": "write_memory",
  "description": "Write project-specific memory",
  "parameters": {
    "name": "string (required)",
    "content": "string (required, markdown format)"
  }
}
```

### 7. **read_memory**
```json
{
  "name": "read_memory",
  "description": "Read a previously saved project memory",
  "parameters": {
    "name": "string (required)"
  }
}
```

### 8. **list_memories**
```json
{
  "name": "list_memories",
  "description": "List all saved memories for the current project"
}
```

### 9. **insert_after_symbol**
```json
{
  "name": "insert_after_symbol",
  "description": "Insert code after a symbol definition",
  "parameters": {
    "symbol_path": "string (required)",
    "code": "string (required)"
  }
}
```

### 10. **insert_before_symbol**
```json
{
  "name": "insert_before_symbol", 
  "description": "Insert code before a symbol definition",
  "parameters": {
    "symbol_path": "string (required)",
    "code": "string (required)"
  }
}
```

## Integration Status

### ✅ Completed Features
- [x] Code Intelligence Service with LSP support
- [x] Semantic code analysis endpoints  
- [x] Project-aware memory system
- [x] MCP tools for code operations
- [x] REST API deployment and testing
- [x] Authentication exemption for endpoints
- [x] Comprehensive tool documentation

### 🏗️ Architecture

```
KnowledgeHub API (Port 3000)
├── Code Intelligence Service
│   ├── Language Detection
│   ├── Framework Detection  
│   ├── Symbol Analysis (AST parsing)
│   ├── LSP Integration
│   └── Project Memory
├── REST Endpoints (/api/code-intelligence/*)
└── MCP Server (stdio transport)
    └── 11 Code Intelligence Tools
```

### 🎯 Key Benefits

1. **Semantic Understanding**: Understands code structure, not just text
2. **Project Context**: Maintains project-specific memory and context
3. **Framework Awareness**: Detects and adapts to different frameworks
4. **Symbol-Level Operations**: Work with classes, functions, methods directly
5. **Intelligent Search**: Pattern-based search with context
6. **Memory Persistence**: Save and recall project-specific insights

## Testing

The integration has been tested and verified:

- ✅ **Service Health**: `/api/code-intelligence/health` returns operational status
- ✅ **Project Activation**: Successfully activates projects for analysis
- ✅ **REST Endpoints**: All endpoints respond correctly
- ✅ **Authentication**: Endpoints exempted from API key requirements
- ✅ **Error Handling**: Graceful error handling and informative responses

## Example Usage

### Basic Workflow
1. **Activate Project**: Start analysis of a codebase
2. **Explore Symbols**: Get overview of classes, functions, etc.
3. **Search Patterns**: Find specific code patterns
4. **Modify Code**: Replace or insert code at symbol level
5. **Save Insights**: Store project-specific memories

### Sample Session
```python
# 1. Activate project
activate_project(project_path="/opt/projects/myapp")

# 2. Get symbols overview  
symbols = get_symbols_overview(relative_path="src/main.py")

# 3. Find specific function
func = find_symbol(name_path="MyClass/process_data", include_body=True)

# 4. Search for patterns
results = search_pattern(pattern="async def.*", file_pattern="*.py")

# 5. Save insights
write_memory(name="api_patterns", content="Found 15 async functions...")
```

## Next Steps

The Serena-inspired code intelligence is now fully integrated into KnowledgeHub. Claude Code can connect to it via:

1. **Direct API calls** to the REST endpoints
2. **MCP server connection** using the provided configuration
3. **Combined usage** with other KnowledgeHub AI features

This provides Claude Code with sophisticated code understanding capabilities while maintaining the project-aware memory and learning features that make KnowledgeHub unique.