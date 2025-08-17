# KnowledgeHub Service Consolidation Migration Guide

## Overview
This guide helps migrate from individual services to consolidated services for better performance and maintainability.

## 1. Import Migration

### Before (Multiple imports across files):
```python
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import logging
import json
```

### After (Consolidated imports):
```python
from api.shared import *
# All common imports are now available
```

## 2. AI Services Migration

### Before (Multiple AI services):
```python
from api.services.embedding_service import EmbeddingService
from api.services.real_embeddings_service import RealEmbeddingService
from api.services.semantic_engine import SemanticEngine
# ... many more imports
```

### After (Consolidated AI service):
```python
from api.services.consolidated_ai import ai_orchestrator

# Usage:
embeddings = await ai_orchestrator.process_intelligence_request("embedding", text)
analysis = await ai_orchestrator.process_intelligence_request("semantic", content)
rag_response = await ai_orchestrator.process_intelligence_request("rag", query)
```

## 3. Memory Services Migration

### Before (Multiple memory services):
```python
from api.services.memory_service import MemoryService
from api.services.hybrid_memory_service import HybridMemoryService
from api.services.object_storage_service import ObjectStorageService
# ... many more
```

### After (Consolidated memory service):
```python
from api.services.consolidated_memory import memory_service, MemoryType

# Usage:
await memory_service.store(MemoryType.SESSION, "key", data)
data = await memory_service.retrieve(MemoryType.PERSISTENT, "key")
await memory_service.sync_memories(MemoryType.SESSION, MemoryType.PERSISTENT, "key")
```

## 4. Middleware Migration

### Before (Multiple middleware files):
```python
from api.middleware.security_headers import SecurityHeadersMiddleware
from api.middleware.rate_limit import RateLimitMiddleware  
from api.middleware.validation import ValidationMiddleware
```

### After (Consolidated middleware):
```python
from api.middleware.consolidated import ConsolidatedMiddleware, ValidationMiddleware, rate_limiter

# Usage in main app:
app.add_middleware(ConsolidatedMiddleware)

# Usage in routes:
@rate_limiter.limit_requests(max_requests=100, window_seconds=60)
@ValidationMiddleware.validate_request_size(max_size=5*1024*1024)
async def my_endpoint():
    pass
```

## 5. Performance Benefits

### Expected Improvements:
- **Import time reduction**: 30-40% faster startup
- **Memory usage**: 25-35% less memory footprint  
- **Code duplication**: 60-70% reduction in duplicate code
- **Maintenance**: Single source of truth for common functionality

## 6. Migration Steps

1. **Phase 1**: Install consolidated modules (already done)
2. **Phase 2**: Update imports in high-traffic routes
3. **Phase 3**: Migrate service instantiation
4. **Phase 4**: Update middleware configuration
5. **Phase 5**: Remove old service files (after validation)

## 7. Validation

After migration, verify:
- All endpoints still function correctly
- Performance metrics show improvement
- No increase in error rates
- Memory usage has decreased

## 8. Rollback Plan

If issues occur:
1. Keep old service files as backup
2. Revert imports to original pattern
3. Restart services
4. Investigate and fix consolidated services

## 9. Files Affected

The consolidation affects these categories:
- **AI Services**: 24 files → 1 consolidated file
- **Memory Services**: 10 files → 1 consolidated file  
- **Middleware**: 8+ functions → 1 consolidated file
- **Common Imports**: 500+ duplicate imports → 1 shared module
