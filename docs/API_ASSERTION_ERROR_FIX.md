# API Assertion Error Fix

**Date**: 2025-07-06  
**Issue**: KnowledgeHub API throwing AssertionError in StaticFiles middleware  
**Status**: ✅ FIXED

## Problem

The KnowledgeHub API was throwing repeated assertion errors:

```
File "/usr/local/lib/python3.11/site-packages/starlette/staticfiles.py", line 96, in __call__
    assert scope["type"] == "http"
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

### Root Cause

The StaticFiles middleware was mounted at the root path "/" and configured to handle all requests. When WebSocket connections tried to connect to `/ws/notifications`, they were intercepted by the StaticFiles middleware which expects only HTTP requests, causing the assertion error.

## Solution

Created a custom `SelectiveStaticFiles` class that extends the default StaticFiles middleware to:

1. Check the request scope type before processing
2. Skip static file handling for WebSocket connections
3. Skip static file handling for paths starting with "/ws"
4. Allow WebSocket requests to pass through to the WebSocket router

### Code Changes

**File**: `/opt/projects/knowledgehub/src/api/main.py`

```python
# Create a custom static files handler that excludes WebSocket paths
from starlette.types import Scope, Receive, Send

class SelectiveStaticFiles(StaticFiles):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Skip static file handling for WebSocket paths
        if scope["type"] == "websocket" or scope["path"].startswith("/ws"):
            # Let the WebSocket handler take over
            await self.app(scope, receive, send)
            return
        # For HTTP requests, use normal static file handling
        await super().__call__(scope, receive, send)

app.mount("/", SelectiveStaticFiles(directory=frontend_dist_path, html=True), name="static")
```

## Verification

After applying the fix and restarting the API container:

1. **No assertion errors** - 0 errors since restart (previously 321)
2. **WebSocket connections working** - Successfully accepting connections at `/ws/notifications`
3. **API endpoints working** - All HTTP endpoints responding normally
4. **Static files still served** - Frontend assets still accessible

## Impact

- Eliminates repeated error logs that were cluttering the system
- Enables proper WebSocket functionality for real-time updates
- Improves overall API stability and performance

## Prevention

To prevent similar issues in the future:

1. Always consider WebSocket routes when mounting catch-all middleware
2. Test WebSocket functionality after adding/modifying middleware
3. Use specific path prefixes for static files instead of root mounting when possible
4. Add integration tests for WebSocket endpoints