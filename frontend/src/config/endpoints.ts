// API endpoint mappings for KnowledgeHub

export const API_ENDPOINTS = {
  // Memory endpoints
  MEMORY_STATS: '/api/claude-auto/memory/stats',
  MEMORY_LIST: '/api/v1/memories',
  MEMORY_SEARCH: '/api/memory/search',
  
  // Session endpoints
  SESSION_CURRENT: '/api/claude-auto/session/current',
  SESSION_CREATE: '/api/claude-auto/session/create',
  
  // AI Intelligence endpoints
  AI_FEATURES: {
    SESSION_CONTINUITY: '/api/claude-auto/session/current',
    MISTAKE_LEARNING: '/api/mistake-learning/lessons',
    PROACTIVE_ASSISTANT: '/api/proactive/predictions',
    DECISION_REASONING: '/api/decisions/search',
    CODE_EVOLUTION: '/api/code-evolution/history',
    PERFORMANCE: '/api/performance/stats',
    PATTERN_RECOGNITION: '/api/patterns/recognize',
    WORKFLOW_INTEGRATION: '/api/claude-workflow/stats'
  },
  
  // Performance endpoints
  PERFORMANCE_METRICS: '/api/performance/report',
  PERFORMANCE_HOURLY: '/api/performance/stats',
  
  // Activity endpoints
  ACTIVITY_RECENT: '/api/claude-workflow/activity',
  
  // Sources endpoints
  SOURCES_LIST: '/api/v1/sources/',
  SOURCES_REFRESH: (id: string) => `/api/v1/sources/${id}/refresh`,
  SOURCES_DELETE: (id: string) => `/api/v1/sources/${id}`,
  
  // Search endpoints
  SEARCH_PUBLIC: '/api/public/search',
  SEARCH_UNIFIED: '/api/v1/search/unified',
  
  // Decision endpoints
  DECISIONS_CATEGORIES: '/api/decisions/categories',
  DECISIONS_SEARCH: '/api/decisions/search',
  
  // Settings endpoints
  SETTINGS_GET: '/api/settings',
  SETTINGS_UPDATE: '/api/settings',
} as const

// Mock data for missing endpoints
export const MOCK_RESPONSES = {
  AI_FEATURES_SUMMARY: {
    total_features: 8,
    active_features: 8,
    performance: {
      accuracy: 92,
      speed: 95,
      reliability: 98
    }
  },
  ACTIVITY_RECENT: {
    activities: [
      { type: 'memory_created', timestamp: new Date().toISOString() },
      { type: 'search_performed', timestamp: new Date().toISOString() }
    ]
  }
}