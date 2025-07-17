// API Configuration for multiple backend services

export interface ServiceEndpoint {
  baseUrl: string
  endpoints: string[]
}

// Define which endpoints belong to which service
export const SERVICE_ROUTES: Record<string, ServiceEndpoint> = {
  memory: {
    baseUrl: 'http://localhost:8003',
    endpoints: [
      '/api/v1/memories',
      '/api/v1/graph',
      '/api/v1/analytics'
    ]
  },
  main: {
    baseUrl: 'http://localhost:3000',
    endpoints: [
      '/api/memory/stats',
      '/api/claude-auto',
      '/api/performance',
      '/api/ai-features',
      '/api/decisions',
      '/api/code-evolution',
      '/api/patterns',
      '/api/claude-workflow',
      '/api/mistakes',
      '/api/proactive'
    ]
  }
}

// Function to determine which service an endpoint belongs to
export function getServiceForEndpoint(path: string): string {
  // Check each service's endpoints
  for (const [serviceName, config] of Object.entries(SERVICE_ROUTES)) {
    for (const endpoint of config.endpoints) {
      if (path.startsWith(endpoint)) {
        return serviceName
      }
    }
  }
  return 'main' // Default to main service
}

// Function to get the base URL for a given path
export function getBaseUrlForPath(path: string, hostname: string): string {
  // Check if we have a configured API URL from environment
  const apiUrl = import.meta.env.VITE_API_URL
  if (apiUrl) {
    return apiUrl
  }
  
  // For localhost/development, let Vite proxy handle it
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return ''  // Let Vite proxy handle it
  }
  
  // For LAN access without configured API URL, default to API server
  if (hostname.startsWith('192.168.1.')) {
    return 'http://192.168.1.25:3000'
  }
  
  return ''  // Default to relative URLs
}

// Mock data for unavailable endpoints
export const MOCK_RESPONSES: Record<string, any> = {
  '/api/memory/stats': {
    total_memories: 0,
    memory_types: {},
    recent_activity: 0,
    storage_used: 0,
    sync_status: 'offline'
  },
  '/api/claude-auto/session/current': {
    session_id: 'mock-session',
    start_time: new Date().toISOString(),
    status: 'active',
    memories_count: 0
  },
  '/api/performance/report': {
    metrics: {
      response_time_avg: 100,
      requests_per_minute: 0,
      error_rate: 0,
      uptime: 100
    }
  },
  '/api/ai-features/summary': {
    features: {
      session_continuity: { status: 'offline', usage: 0 },
      mistake_learning: { status: 'offline', errors_tracked: 0 },
      proactive_assistance: { status: 'offline', suggestions_made: 0 },
      decision_reasoning: { status: 'offline', decisions_tracked: 0 },
      code_evolution: { status: 'offline', changes_tracked: 0 },
      performance_optimization: { status: 'offline', optimizations: 0 },
      workflow_integration: { status: 'offline', workflows_captured: 0 },
      pattern_recognition: { status: 'offline', patterns_found: 0 }
    }
  },
  '/api/code-evolution/recent': {
    changes: [],
    total: 0,
    last_update: new Date().toISOString()
  },
  '/api/decisions/recent': {
    decisions: [],
    total: 0,
    last_update: new Date().toISOString()
  },
  '/api/claude-workflow/active': {
    workflows: [],
    active_count: 0,
    last_update: new Date().toISOString()
  },
  '/api/patterns/recent': {
    patterns: [],
    total: 0,
    last_update: new Date().toISOString()
  },
  '/api/mistakes/recent': {
    mistakes: [],
    total: 0,
    resolved: 0,
    pending: 0
  },
  '/api/proactive/suggestions': {
    suggestions: [],
    total: 0,
    accepted: 0,
    declined: 0
  },
  '/api/memory/recent': {
    memories: [],
    total: 0,
    last_update: new Date().toISOString()
  },
  '/api/claude-auto/stats': {
    sessions: 0,
    memories: 0,
    decisions: 0,
    errors: 0,
    performance_metrics: {}
  }
}