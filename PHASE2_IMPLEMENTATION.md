# Phase 2 Implementation: Advanced Memory and Security

## Overview

Phase 2 of the KnowledgeHub RAG system adds two critical production-grade components:
1. **Zep Memory System** - Conversational memory with temporal knowledge graphs
2. **RBAC Security** - Role-based access control with multi-tenant isolation

## Components Implemented

### 1. Zep Memory System

#### Architecture
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Claude Code    │────▶│  KnowledgeHub    │────▶│   Zep Memory    │
│                 │     │  API Gateway     │     │   (Port 8000)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │                           │
                                ▼                           ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │   RAG System     │     │  Zep PostgreSQL │
                        │  (Qdrant)        │     │  (pgvector)     │
                        └──────────────────┘     └─────────────────┘
```

#### Features
- **Temporal Knowledge Graphs** - Track how facts change over time
- **Entity Extraction** - Automatic extraction of people, places, concepts
- **Conversation Summarization** - Configurable message window summaries
- **Hybrid Retrieval** - Combine RAG results with conversation memory
- **Multi-source Integration** - Ingest from various sources (Slack, email, etc.)

#### Docker Configuration
```yaml
services:
  zep:
    image: zepai/zep:latest
    ports:
      - "8000:8000"
    environment:
      - ZEP_STORE_TYPE=postgres
      - ZEP_STORE_POSTGRES_DSN=postgresql://zep:zep123@zep-postgres:5432/zep
      - ZEP_NLP_SPACY_MODEL=en_core_web_sm
      - ZEP_MEMORY_MESSAGE_WINDOW=12
      - ZEP_EXTRACTORS_MESSAGES_ENTITIES_ENABLED=true
      - ZEP_EXTRACTORS_MESSAGES_SUMMARIZER_ENABLED=true
```

#### API Endpoints
- `POST /api/zep/messages` - Add message to conversation memory
- `GET /api/zep/memory/{session_id}` - Get conversation memory
- `POST /api/zep/search` - Search across conversations
- `GET /api/zep/sessions` - Get user's sessions
- `POST /api/zep/hybrid-search` - Hybrid RAG + Memory search

### 2. RBAC (Role-Based Access Control)

#### Security Model
```
User ──▶ Role ──▶ Permissions ──▶ Resources
         │                           │
         └─────── Tenant ────────────┘
```

#### Roles and Permissions

| Role | Permissions |
|------|-------------|
| **Viewer** | Read documents, memories, query RAG |
| **User** | Viewer + Write documents/memories |
| **Developer** | User + Delete, Ingest, Scrape, Monitor |
| **Admin** | Developer + User management, Config |
| **Super Admin** | All permissions, cross-tenant access |

#### Permission Types
```python
class Permission(Enum):
    # Document permissions
    DOC_READ = "doc:read"
    DOC_WRITE = "doc:write"
    DOC_DELETE = "doc:delete"
    DOC_ADMIN = "doc:admin"
    
    # Memory permissions
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    MEMORY_DELETE = "memory:delete"
    MEMORY_ADMIN = "memory:admin"
    
    # RAG permissions
    RAG_QUERY = "rag:query"
    RAG_INGEST = "rag:ingest"
    RAG_SCRAPE = "rag:scrape"
    RAG_ADMIN = "rag:admin"
    
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_CONFIG = "system:config"
```

#### Security Features
- **Multi-tenant Isolation** - Data segregation by tenant
- **API Key Management** - Create keys with specific permissions
- **Audit Logging** - Track all access attempts
- **Permission Decorators** - Easy endpoint protection
- **Document-level Access** - Fine-grained document permissions

## Implementation Details

### 1. Zep Memory Service (`api/services/zep_memory.py`)

```python
# Add message to memory
await zep_service.add_message(
    session_id="session-123",
    role="user",
    content="How do I implement authentication?",
    user_id="user-456",
    metadata={"project": "myapp"}
)

# Search memory
results = await zep_service.search_memory(
    query="authentication",
    user_id="user-456",
    limit=5
)

# Hybrid retrieval (RAG + Memory)
combined_results = await zep_service.hybrid_retrieval(
    query="authentication methods",
    user_id="user-456",
    rag_results=rag_results,
    weight_memory=0.3,
    weight_rag=0.7
)
```

### 2. RBAC Service (`api/services/rbac_service.py`)

```python
# Check permissions
if rbac.has_permission(user, Permission.RAG_INGEST):
    # Allow document ingestion
    
# Filter by tenant
query = rbac.filter_by_tenant(user, query)

# Create API key
api_key = rbac.create_api_key(
    user=admin_user,
    name="ci-deployment",
    permissions=[Permission.RAG_QUERY, Permission.DOC_READ],
    expires_at=datetime.utcnow() + timedelta(days=30)
)
```

### 3. Permission Decorators (`api/middleware/rbac_middleware.py`)

```python
# Require single permission
@router.post("/ingest")
@require_permission(Permission.RAG_INGEST)
async def ingest_document(current_user: User = Depends(get_current_user)):
    ...

# Require any permission
@router.get("/data")
@require_any_permission([Permission.DOC_READ, Permission.MEMORY_READ])
async def get_data(current_user: User = Depends(get_current_user)):
    ...

# Document-specific permission
@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    user: User = Depends(check_document_permission(Permission.DOC_READ))
):
    ...
```

## Configuration

### Environment Variables
```bash
# Zep Memory System
ZEP_API_URL=http://localhost:8000
ZEP_API_KEY=your-zep-api-key  # Optional

# RBAC Settings
ENABLE_RBAC=true
ENABLE_MULTI_TENANT=true
DEFAULT_USER_ROLE=user
```

### Docker Deployment
```bash
# Start Zep and dependencies
docker compose up -d zep zep-postgres

# Verify Zep is running
curl http://localhost:8000/healthz

# Check Qdrant status
curl http://localhost:6333/
```

## Usage Examples

### 1. Conversational Memory

```python
# Track conversation
POST /api/zep/messages
{
    "session_id": "claude-session-123",
    "role": "user",
    "content": "I need to implement OAuth2 for my FastAPI app",
    "metadata": {
        "project": "myapp",
        "language": "python"
    }
}

# Get conversation history
GET /api/zep/memory/claude-session-123?limit=20

# Search across conversations
POST /api/zep/search
{
    "query": "OAuth2 implementation",
    "limit": 5
}
```

### 2. Hybrid RAG + Memory Search

```python
# First get RAG results
rag_results = await rag_service.query("OAuth2 FastAPI")

# Then combine with memory
POST /api/zep/hybrid-search
{
    "query": "OAuth2 FastAPI",
    "rag_results": rag_results,
    "weight_memory": 0.4,  # Give more weight to personal context
    "weight_rag": 0.6
}
```

### 3. API Key with Limited Permissions

```python
# Create read-only API key
POST /api/rbac/api-keys
{
    "name": "documentation-reader",
    "permissions": ["doc:read", "rag:query"],
    "expires_at": "2025-12-31T23:59:59Z"
}

# Use API key
curl -H "Authorization: Bearer kh_..." \
     http://localhost:3000/api/rag/query
```

## Security Best Practices

### 1. Tenant Isolation
- All queries automatically filtered by tenant
- Cross-tenant access only for super admins
- Tenant ID included in all audit logs

### 2. Permission Validation
- Validate at API gateway level
- Check document-level permissions
- Log all access attempts

### 3. API Key Security
- Keys inherit creator's permissions
- Automatic expiration
- Revocation capability
- Audit trail for key usage

## Monitoring and Debugging

### 1. Zep Memory Monitoring
```bash
# Check Zep health
curl http://localhost:8000/healthz

# View Zep logs
docker logs knowledgehub-zep-1 --tail 100

# Database queries
docker exec -it knowledgehub-zep-postgres-1 psql -U zep
```

### 2. RBAC Audit Logs
```python
# View access logs (admin only)
GET /api/rbac/audit-logs?user_id=xxx&granted=false

# Permission debugging
GET /api/rbac/debug/user/{user_id}/permissions
```

## Performance Considerations

### 1. Memory Optimization
- Zep message window: 12 messages
- Auto-summarization for older messages
- Efficient entity extraction
- Vector embeddings: 1024 dimensions

### 2. RBAC Caching
- Permission cache: 5 minutes
- API key cache: 24 hours
- Tenant filter cache: Per request

### 3. Hybrid Search
- Parallel retrieval from RAG and Zep
- Configurable result weights
- Duplicate detection
- Score-based ranking

## Next Steps: Phase 3

With Phase 2 complete, the system now has:
- ✅ Production RAG with real data sources
- ✅ Conversational memory with Zep
- ✅ RBAC and multi-tenant security

Phase 3 will add:
- 🔄 Multi-agent orchestrator for complex tasks
- 🕸️ GraphRAG with Neo4j PropertyGraphIndex
- 🤖 Specialized domain agents

The foundation is now ready for advanced multi-agent reasoning and graph-based knowledge representation.