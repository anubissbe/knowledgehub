# KnowledgeHub Developer Guide

## Getting Started

This guide will help you set up your development environment and start contributing to KnowledgeHub.

---

## Development Environment Setup

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm 9+
- Docker Desktop 20.10+
- Git 2.30+
- VS Code or PyCharm (recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/org/knowledgehub.git
cd knowledgehub
```

### 2. Python Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev  # Start development server
```

### 4. Docker Services
```bash
# Start all services
docker-compose up -d

# Or start specific services
docker-compose up -d postgres redis neo4j

# View logs
docker-compose logs -f api
```

### 5. Environment Configuration
```bash
# Copy example environment file
cp .env.example .env

# Edit with your configuration
nano .env
```

Required environment variables:
```env
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/knowledgehub
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET_KEY=your-secret-key-change-this
API_KEY_ENCRYPTION_KEY=your-encryption-key

# AI Services
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Vector Databases
WEAVIATE_URL=http://localhost:8080
QDRANT_URL=http://localhost:6333

# Graph Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

---

## Project Structure

```
knowledgehub/
├── api/                    # Backend API
│   ├── routers/           # API endpoints
│   ├── services/          # Business logic
│   ├── models/            # Database models
│   ├── middleware/        # Custom middleware
│   ├── security/          # Security modules
│   └── main.py           # FastAPI application
├── frontend/              # React frontend
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Page components
│   │   ├── services/     # API clients
│   │   ├── hooks/        # Custom hooks
│   │   └── utils/        # Utilities
│   └── vite.config.ts    # Vite configuration
├── tests/                 # Test suites
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
├── docs/                  # Documentation
├── scripts/              # Utility scripts
└── docker-compose.yml    # Docker services
```

---

## Development Workflow

### 1. Creating a New Feature

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# ... edit files ...

# Run tests
pytest tests/
npm test

# Commit changes
git add .
git commit -m "feat: add new feature"

# Push branch
git push origin feature/your-feature-name
```

### 2. Code Style Guidelines

#### Python (Backend)
- Follow PEP 8
- Use type hints
- Maximum line length: 88 (Black formatter)
- Docstrings for all public functions

```python
from typing import Optional, List, Dict, Any

async def search_documents(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Search documents using hybrid RAG.
    
    Args:
        query: Search query text
        filters: Optional filter criteria
        top_k: Number of results to return
        
    Returns:
        List of document dictionaries with scores
    """
    # Implementation
    pass
```

#### TypeScript (Frontend)
- Use TypeScript strict mode
- Functional components with hooks
- Interface over type when possible

```typescript
interface SearchProps {
  query: string;
  onSearch: (results: SearchResult[]) => void;
  className?: string;
}

export const SearchComponent: React.FC<SearchProps> = ({
  query,
  onSearch,
  className
}) => {
  const [loading, setLoading] = useState(false);
  
  const handleSearch = async () => {
    setLoading(true);
    try {
      const results = await searchAPI(query);
      onSearch(results);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    // JSX
  );
};
```

### 3. Testing

#### Unit Tests
```python
# tests/unit/test_rag_service.py
import pytest
from api.services.unified_rag_service import UnifiedRAGService

@pytest.mark.asyncio
async def test_search_returns_results():
    service = UnifiedRAGService()
    results = await service.search("test query", top_k=5)
    
    assert len(results) <= 5
    assert all("score" in r for r in results)
```

#### Integration Tests
```python
# tests/integration/test_api_endpoints.py
import pytest
from httpx import AsyncClient
from api.main import app

@pytest.mark.asyncio
async def test_rag_search_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/rag/search",
            json={"query": "test", "top_k": 10}
        )
        assert response.status_code == 200
        assert "results" in response.json()
```

#### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=api --cov-report=html

# Run specific test file
pytest tests/unit/test_rag_service.py

# Run with markers
pytest -m "not slow"
```

### 4. Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback one revision
alembic downgrade -1

# View migration history
alembic history
```

---

## API Development

### Adding a New Endpoint

1. **Create Router** (`api/routers/new_feature.py`):
```python
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from pydantic import BaseModel

router = APIRouter(prefix="/api/new-feature", tags=["new-feature"])

class NewFeatureRequest(BaseModel):
    field1: str
    field2: int = 10

class NewFeatureResponse(BaseModel):
    result: str
    metadata: dict

@router.post("/process", response_model=NewFeatureResponse)
async def process_new_feature(
    request: NewFeatureRequest,
    current_user: dict = Depends(get_current_user)
):
    """Process new feature request."""
    # Implementation
    return NewFeatureResponse(
        result="processed",
        metadata={"user": current_user["id"]}
    )
```

2. **Register Router** (`api/main.py`):
```python
from api.routers import new_feature

app.include_router(new_feature.router)
```

3. **Add Service Logic** (`api/services/new_feature_service.py`):
```python
class NewFeatureService:
    def __init__(self):
        self.db = get_database()
    
    async def process(self, data: dict) -> dict:
        # Business logic
        return result
```

---

## Frontend Development

### Creating a New Component

```tsx
// frontend/src/components/NewComponent.tsx
import React, { useState, useEffect } from 'react';
import { useApi } from '../hooks/useApi';
import { Card } from './ui/Card';

interface NewComponentProps {
  title: string;
  onUpdate?: (data: any) => void;
}

export const NewComponent: React.FC<NewComponentProps> = ({ 
  title, 
  onUpdate 
}) => {
  const [data, setData] = useState(null);
  const { loading, error, fetchData } = useApi();
  
  useEffect(() => {
    loadData();
  }, []);
  
  const loadData = async () => {
    const result = await fetchData('/api/new-feature/data');
    setData(result);
    onUpdate?.(result);
  };
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  
  return (
    <Card>
      <h2>{title}</h2>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </Card>
  );
};
```

### State Management

```tsx
// frontend/src/store/newFeatureSlice.ts
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { api } from '../services/api';

export const fetchFeatureData = createAsyncThunk(
  'newFeature/fetch',
  async (params: any) => {
    const response = await api.post('/new-feature/process', params);
    return response.data;
  }
);

const newFeatureSlice = createSlice({
  name: 'newFeature',
  initialState: {
    data: null,
    loading: false,
    error: null,
  },
  reducers: {
    clearData: (state) => {
      state.data = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchFeatureData.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchFeatureData.fulfilled, (state, action) => {
        state.loading = false;
        state.data = action.payload;
      })
      .addCase(fetchFeatureData.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message;
      });
  },
});

export const { clearData } = newFeatureSlice.actions;
export default newFeatureSlice.reducer;
```

---

## Debugging

### Backend Debugging

```python
# Enable debug mode in .env
DEBUG=true
LOG_LEVEL=DEBUG

# Use debugger
import pdb; pdb.set_trace()

# Or with ipdb
import ipdb; ipdb.set_trace()

# Async debugging
import asyncio
import aiodebug
aiodebug.log_slow_callbacks(0.1)
```

### Frontend Debugging

```typescript
// React DevTools
// Install browser extension

// Console debugging
console.log('Data:', data);
console.table(results);
console.time('Operation');
// ... code ...
console.timeEnd('Operation');

// Breakpoints
debugger; // Pause execution here

// Network debugging
// Use browser DevTools Network tab
```

### Docker Debugging

```bash
# View container logs
docker logs -f knowledgehub-api-1

# Execute commands in container
docker exec -it knowledgehub-api-1 bash

# Python shell in container
docker exec -it knowledgehub-api-1 python

# Database shell
docker exec -it knowledgehub-postgres-1 psql -U postgres -d knowledgehub
```

---

## Performance Optimization

### Backend Optimization

1. **Use Caching Decorator**:
```python
from api.services.caching_system import cached

@cached(ttl=600, prefix="search")
async def expensive_search(query: str):
    # Expensive operation
    return results
```

2. **Batch Database Operations**:
```python
from api.services.db_optimizer import db_optimizer

# Instead of multiple queries
results = await db_optimizer.batch_query([
    "SELECT * FROM table1 WHERE ...",
    "SELECT * FROM table2 WHERE ...",
])
```

3. **Async Operation Limits**:
```python
from api.services.async_optimizer import async_optimizer

# Process with concurrency limit
results = await async_optimizer.batch_process(
    items=large_list,
    processor=process_item,
    batch_size=50
)
```

### Frontend Optimization

1. **Lazy Loading**:
```tsx
const LazyComponent = React.lazy(() => import('./HeavyComponent'));

<Suspense fallback={<Loading />}>
  <LazyComponent />
</Suspense>
```

2. **Memoization**:
```tsx
const MemoizedComponent = React.memo(({ data }) => {
  return <ExpensiveRender data={data} />;
});

const memoizedValue = useMemo(() => {
  return expensiveComputation(data);
}, [data]);
```

3. **Virtual Scrolling**:
```tsx
import { VariableSizeList } from 'react-window';

<VariableSizeList
  height={600}
  itemCount={items.length}
  itemSize={getItemSize}
  width="100%"
>
  {Row}
</VariableSizeList>
```

---

## Common Issues and Solutions

### Issue: Database Connection Errors
```bash
# Solution: Check database is running
docker-compose ps
docker-compose up -d postgres

# Check connection string
echo $DATABASE_URL
```

### Issue: Import Errors
```python
# Solution: Add project to Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Issue: CORS Errors
```python
# Solution: Update CORS settings in main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue: Memory Leaks
```python
# Solution: Properly close connections
async with aiohttp.ClientSession() as session:
    # Use session
    pass  # Auto-closes

# Clear caches periodically
cache_system.local_cache.clear()
```

---

## Contributing

### Pull Request Process

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'feat: add amazing feature'`
4. **Push branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request** with description

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

Example:
```
feat(rag): add cross-encoder reranking

Implement cross-encoder reranking for improved search relevance.
Adds new reranking_optimizer service with configurable models.

Closes #123
```

---

## Resources

### Documentation
- [System Documentation](./SYSTEM_DOCUMENTATION.md)
- [API Documentation](./API_DOCUMENTATION.md)
- [Architecture Overview](./ARCHITECTURE.md)

### External Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Docker Documentation](https://docs.docker.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

### Support
- GitHub Issues: [Report bugs](https://github.com/org/knowledgehub/issues)
- Discussions: [Ask questions](https://github.com/org/knowledgehub/discussions)
- Discord: [Join community](https://discord.gg/knowledgehub)

---

*Last Updated: August 17, 2025*  
*Version: 1.0.0*