
// Updated API configuration for new backend implementation
export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_URL || 'http://192.168.1.25:3000',
  ENDPOINTS: {
    HEALTH: '/health',
    API_INFO: '/api',
    SOURCES: '/api/v1/sources',
    MEMORY: '/api/memory',
    SESSION: '/api/memory/session',
    RAG_QUERY: '/api/rag/query',
    RAG_INDEX: '/api/rag/index',
    LLAMAINDEX: '/api/llamaindex',
    GRAPHRAG: '/api/graphrag',
    WEBSOCKET: '/ws'
  },
  TIMEOUT: 10000,
  RETRY_ATTEMPTS: 3
};

// Authentication configuration for JWT
export const AUTH_CONFIG = {
  TOKEN_KEY: 'knowledgehub_token',
  TOKEN_REFRESH_KEY: 'knowledgehub_refresh_token',
  LOGIN_ENDPOINT: '/auth/login',
  REFRESH_ENDPOINT: '/auth/refresh',
  LOGOUT_ENDPOINT: '/auth/logout'
};

// Performance monitoring configuration
export const PERFORMANCE_CONFIG = {
  ENABLE_MONITORING: true,
  METRICS_ENDPOINT: '/api/metrics',
  HEALTH_CHECK_INTERVAL: 30000
};

// Utility function for base URL configuration
export const getBaseUrlForPath = (path: string = "/", hostname?: string): string => {
  const currentHostname = hostname || window.location.hostname;
  
  // LAN environment detection
  if (currentHostname.startsWith("192.168.1.")) {
    return "http://192.168.1.25:3000";
  }
  
  // Development environment
  if (currentHostname === "localhost" || currentHostname === "127.0.0.1") {
    return "http://192.168.1.25:3000";
  }
  
  // Default fallback
  return API_CONFIG.BASE_URL;
};

// URL helper functions
export const getApiUrl = (endpoint: string): string => {
  const baseUrl = getBaseUrlForPath();
  return `${baseUrl}${endpoint}`;
};

export const getWebSocketUrl = (): string => {
  const baseUrl = getBaseUrlForPath();
  return baseUrl.replace(/^http/, "ws") + "/ws";
};

export default API_CONFIG;
