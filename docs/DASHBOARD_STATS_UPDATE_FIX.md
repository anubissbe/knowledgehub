# Dashboard Stats Update Fix Documentation

## Problem Statement
The KnowledgeHub dashboard was not updating statistics (total chunks and documents) when jobs were re-run. Users had to manually refresh their browser to see updated counts.

## Solution Overview
Implemented a multi-layered solution to ensure dashboard stats update automatically through three mechanisms:
1. Query cache invalidation on user actions
2. Automatic periodic refresh
3. Real-time WebSocket notifications

## Implementation Details

### 1. Frontend Query Cache Invalidation

**File: `/src/web-ui/src/pages/Jobs.tsx`**

Added cache invalidation for dashboard stats and sources when job actions are performed:

```typescript
const retryMutation = useMutation({
  mutationFn: api.retryJob,
  onSuccess: () => {
    // Invalidate multiple queries to ensure UI updates
    queryClient.invalidateQueries({ queryKey: ['jobs'] })
    queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] })
    queryClient.invalidateQueries({ queryKey: ['sources'] })
  },
})
```

This ensures immediate updates when users retry or cancel jobs.

### 2. Dashboard Auto-Refresh

**File: `/src/web-ui/src/pages/Dashboard.tsx`**

Added automatic refresh and manual refresh button:

```typescript
const { data: stats, isLoading, refetch, isRefetching } = useQuery({
  queryKey: ['dashboard-stats'],
  queryFn: api.getDashboardStats,
  refetchInterval: 30000, // Refresh every 30 seconds
  refetchIntervalInBackground: true, // Continue refreshing even when tab is not active
})
```

Added refresh button in the UI:
```typescript
<Tooltip title="Refresh stats">
  <IconButton onClick={() => refetch()} disabled={isRefetching} color="primary">
    <RefreshIcon />
  </IconButton>
</Tooltip>
```

### 3. WebSocket Real-time Updates

**File: `/src/web-ui/src/contexts/WebSocketContext.tsx`** (New)

Created a WebSocket context provider that:
- Establishes WebSocket connection to the backend
- Listens for job status updates
- Invalidates relevant queries when updates are received
- Handles reconnection automatically

Key message handlers:
```typescript
case 'job_completed':
case 'job_failed':
case 'job_cancelled':
  // Invalidate queries when job status changes
  queryClient.invalidateQueries({ queryKey: ['jobs'] })
  queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] })
  queryClient.invalidateQueries({ queryKey: ['sources'] })
  break

case 'stats_updated':
  // Invalidate dashboard stats
  queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] })
  break
```

### 4. Backend WebSocket Broadcasting

**File: `/src/api/services/job_service.py`**

Modified `update_job_status` to broadcast WebSocket notifications:

```python
# Broadcast WebSocket notifications
if notification_type:
    # Send notification to all connected clients
    asyncio.create_task(broadcast_to_all({
        "type": notification_type,
        "job_id": str(job_id),
        "status": status.lower(),
        "stats": stats
    }))
    
    # Also send stats update notification
    asyncio.create_task(broadcast_to_all({
        "type": "stats_updated",
        "source_id": str(job.source_id) if job.source_id else None
    }))
```

**File: `/src/api/routers/websocket.py`**

Added `broadcast_to_all` function to send messages to all connected clients:

```python
async def broadcast_to_all(message: Dict):
    """Broadcast message to all connected clients"""
    disconnected_clients = []
    for client_id, websocket in clients.items():
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send to {client_id}: {e}")
            disconnected_clients.append(client_id)
```

### 5. Integration Points

**File: `/src/web-ui/src/main.tsx`**

Added WebSocketProvider to the application:

```typescript
<WebSocketProvider>
  <App />
</WebSocketProvider>
```

**File: `/src/web-ui/src/services/api.ts`**

Updated WebSocket connection to use correct endpoint:

```typescript
connectWebSocket: () => {
  const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:3000'
  return new WebSocket(`${wsUrl}/ws/notifications`)
}
```

## How It Works

1. **User Action Flow**:
   - User clicks retry/cancel on a job
   - Frontend immediately invalidates cache → UI updates
   - Backend processes the action
   - Backend broadcasts WebSocket notification
   - Frontend receives notification → invalidates cache again

2. **Background Updates**:
   - Dashboard automatically refreshes every 30 seconds
   - WebSocket notifications trigger immediate updates
   - Manual refresh button available for instant updates

3. **Job Completion Flow**:
   - Worker completes job processing
   - Worker calls API to update job status
   - Backend updates database and broadcasts WebSocket message
   - All connected clients receive notification
   - Dashboard stats update automatically

## Benefits

1. **Real-time Updates**: Stats update immediately when jobs complete
2. **No Manual Refresh**: Users don't need to refresh the browser
3. **Multiple Update Paths**: Ensures updates even if WebSocket fails
4. **Background Updates**: Stats stay current even when user is on different page
5. **Scalable**: All connected clients receive updates simultaneously

## Testing

To verify the implementation:

1. Open the dashboard in a browser
2. Start a new job or retry a failed job
3. Watch the dashboard stats - they should update:
   - Immediately when you take action
   - Automatically every 30 seconds
   - When jobs complete (via WebSocket)
4. Open multiple browser tabs - all should update simultaneously

## Future Improvements

1. Add visual feedback when stats are updating
2. Implement differential updates (only changed stats)
3. Add connection status indicator for WebSocket
4. Consider using Server-Sent Events as fallback
5. Add metrics for update latency