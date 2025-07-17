# KnowledgeHub Memory API Fix

## Solution Applied

The memory system API is running on port 8003, not 3000. I've updated the configuration to use this working API.

### Changes Made:

1. **API Service Discovery**:
   - Found memory API running on port 8003 at `/opt/projects/memory-system`
   - Endpoint: `GET /api/v1/memories/search?q=&limit=100`

2. **Frontend Updates**:
   - Updated `/frontend/src/services/api.ts` to use port 8003 for LAN access
   - Modified `/frontend/src/pages/MemorySystem.tsx` to:
     - Use correct endpoint path
     - Map API response format to component's Memory interface
     - Handle different field names (memory_type vs type, created_at vs timestamp)

3. **Vite Configuration**:
   - Added proxy rule for `/api/v1/memories` to route to port 8003
   - Maintains existing proxy for other APIs on port 3000

## To Apply Changes:

1. **Restart Vite dev server** (required for proxy changes):
   ```bash
   cd /opt/projects/knowledgehub/frontend
   # Kill existing process
   pkill -f "vite.*--port 3101"
   # Restart
   npm run dev -- --host 0.0.0.0 --port 3101
   ```

2. **Verify Memory API** is running:
   ```bash
   curl http://localhost:8003/health
   ```

## API Response Format

The memory API returns data in this format:
```json
{
  "id": "uuid",
  "content": "memory content",
  "memory_type": "CODE|CONTEXT|DECISION|etc",
  "tags": ["tag1", "tag2"],
  "user_id": "user",
  "created_at": "ISO timestamp",
  "metadata": {
    "session_id": "session_123",
    "timestamp": 123456789,
    // other metadata
  }
}
```

The frontend now correctly maps this to its expected format.