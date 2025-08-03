# WebSocket and SSE Authentication Guide

## Overview

KnowledgeHub supports both WebSocket and Server-Sent Events (SSE) for real-time communication. Both protocols support optional authentication for enhanced functionality.

## WebSocket Authentication

### Endpoint
- **URL**: `ws://[host]/ws/notifications`
- **Protocol**: WebSocket

### Authentication Methods

#### 1. Anonymous Connection
```javascript
const ws = new WebSocket('ws://localhost:3000/ws/notifications');
```

#### 2. Token-based Authentication
```javascript
const token = 'your-auth-token';
const ws = new WebSocket(`ws://localhost:3000/ws/notifications?token=${token}`);
```

### Connection Flow

1. **Origin Validation**: The server validates the connection origin to prevent CSRF attacks
2. **Authentication** (optional): If a token is provided, it's validated
3. **Welcome Message**: Server sends connection details:
```json
{
  "type": "connected",
  "client_id": "uuid",
  "user_id": "authenticated_user",
  "authenticated": true,
  "message": "WebSocket connection established"
}
```

### Security Features

- **Origin Validation**: Only allows connections from whitelisted origins
- **CSRF Protection**: Validates origin headers
- **Optional Authentication**: Supports both anonymous and authenticated connections
- **User Association**: Tracks authenticated users for targeted messaging

## Server-Sent Events (SSE) Authentication

### Endpoint
- **URL**: `http://[host]/api/realtime/stream`
- **Protocol**: HTTP with text/event-stream

### Authentication Methods

#### 1. Anonymous Connection
```javascript
const eventSource = new EventSource('/api/realtime/stream');
```

#### 2. Token-based Authentication
```javascript
const token = 'your-auth-token';
const eventSource = new EventSource(`/api/realtime/stream?token=${token}`);
```

### Event Types
- `CODE_CHANGE`: Code modifications
- `MEMORY_CREATED`: New memory items
- `DECISION_MADE`: Decision records
- `ERROR_OCCURRED`: Error events
- `PATTERN_DETECTED`: Pattern recognition

### CORS Support
The SSE endpoint includes CORS headers for cross-origin access:
```
Access-Control-Allow-Origin: *
```

## Implementation Details

### WebSocket Handler (`/api/routers/websocket.py`)
- Maintains client connections in memory
- Supports subscription-based messaging
- Handles disconnections gracefully
- Tracks user associations for authenticated connections

### SSE Handler (`/api/routers/realtime_learning.py`)
- Streams events from the real-time learning pipeline
- Supports event type filtering
- Handles connection drops automatically
- Provides retry mechanism for clients

## Client Examples

### JavaScript WebSocket Client
```javascript
class WebSocketClient {
  constructor(token = null) {
    const url = token 
      ? `ws://localhost:3000/ws/notifications?token=${token}`
      : 'ws://localhost:3000/ws/notifications';
    
    this.ws = new WebSocket(url);
    
    this.ws.onopen = () => {
      console.log('WebSocket connected');
    };
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('Received:', data);
    };
    
    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }
  
  subscribe(jobId) {
    this.ws.send(JSON.stringify({
      type: 'subscribe',
      job_id: jobId
    }));
  }
}
```

### JavaScript SSE Client
```javascript
class SSEClient {
  constructor(token = null, eventTypes = []) {
    const params = new URLSearchParams();
    if (token) params.append('token', token);
    if (eventTypes.length) params.append('event_types', eventTypes.join(','));
    
    const url = `/api/realtime/stream${params.toString() ? '?' + params : ''}`;
    this.eventSource = new EventSource(url);
    
    this.eventSource.onopen = () => {
      console.log('SSE connected');
    };
    
    this.eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('SSE event:', data);
    };
    
    this.eventSource.onerror = (error) => {
      console.error('SSE error:', error);
    };
  }
}
```

## Troubleshooting

### Common Issues

1. **Origin Not Allowed**: Ensure your client origin is in the allowed list
2. **Connection Refused**: Check if the API is running and accessible
3. **Authentication Failed**: Verify token format and validity
4. **No Events Received**: Check event type filters and pipeline status

### Debug Tips

- Check browser console for connection errors
- Monitor server logs for authentication issues
- Use browser DevTools to inspect WebSocket frames
- Test with curl for SSE: `curl -N http://localhost:3000/api/realtime/stream`

## Security Considerations

1. **Token Transmission**: Tokens are sent as query parameters due to WebSocket/SSE limitations
2. **HTTPS Required**: Use WSS/HTTPS in production to encrypt token transmission
3. **Token Rotation**: Implement token expiration and rotation for long-lived connections
4. **Rate Limiting**: Consider implementing connection rate limits per user
5. **Resource Management**: Monitor and limit concurrent connections