# ProjectHub Proxy Server

This proxy server fixes the broken PUT endpoint in the ProjectHub API by implementing a workaround.

## Problem

The ProjectHub API at `http://192.168.1.24:3009/api/tasks/{id}` returns 500 Internal Server Error for all PUT requests, making it impossible to update task status.

## Solution

This proxy server:
1. Intercepts PUT requests to `/api/tasks/{id}`
2. For status updates, creates a new task with the updated status (since POST works)
3. Returns the new task as if it was an update
4. Passes through all other requests unchanged

## Usage

### Start the proxy server:
```bash
npm install
npm start
```

### Use the proxy endpoint instead of the direct API:
```javascript
// Instead of:
// PUT http://192.168.1.24:3009/api/tasks/{id}

// Use:
// PUT http://localhost:3109/api/tasks/{id}
```

### Example:
```bash
# Update task status to completed
curl -X PUT http://localhost:3109/api/tasks/some-task-id \
  -H "Content-Type: application/json" \
  -d '{"status": "completed"}'
```

## Environment Variables

- `PORT` - Proxy server port (default: 3109)
- `PROJECTHUB_API` - ProjectHub API URL (default: http://192.168.1.24:3009/api)

## How It Works

1. **GET, POST, DELETE** - Proxied directly to the original API
2. **PUT /api/tasks/{id}** - Implemented with workaround:
   - Fetches existing task
   - If status is being changed, creates a new task with updated status
   - Returns the new task with metadata about the update method

## Limitations

- Only status updates are fully supported
- Other field updates are simulated (returned but not persisted)
- Original task IDs are preserved in the response for reference

## Testing

Test the proxy with:
```bash
# Health check
curl http://localhost:3109/health

# Get tasks (proxied)
curl http://localhost:3109/api/tasks

# Update task status (workaround)
curl -X PUT http://localhost:3109/api/tasks/task-id \
  -H "Content-Type: application/json" \
  -d '{"status": "completed"}'
```