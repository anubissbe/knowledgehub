// Core types for API and WebSocket services

export interface ApiResponse<T = any> {
  data: T;
  message?: string;
  status: "success" | "error";
  timestamp?: string;
}

export interface ApiError {
  message: string;
  code?: string | number;
  details?: any;
  timestamp?: string;
  requestId?: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

export interface RequestConfig {
  timeout?: number;
  retries?: number;
  retryDelay?: number;
  cache?: boolean;
  cacheKey?: string;
  cacheTTL?: number;
}

// WebSocket types
export interface WebSocketMessage<T = any> {
  type: string;
  payload: T;
  timestamp?: string;
  id?: string;
}

export interface WebSocketError {
  code: number;
  message: string;
  reason?: string;
  timestamp: string;
}

export interface ConnectionState {
  isConnected: boolean;
  isConnecting: boolean;
  lastConnectedAt?: Date;
  reconnectAttempts: number;
  error?: WebSocketError;
}

// Service-specific types
export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}

export interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  preferences: UserPreferences;
  permissions: string[];
}

export interface UserPreferences {
  theme: "light" | "dark" | "auto";
  notifications: boolean;
  autoSave: boolean;
  language: string;
}

// Memory Service types
export interface Memory {
  id: string;
  type: string;
  content: string;
  metadata: Record<string, any>;
  tags: string[];
  created_at: string;
  updated_at: string;
}

export interface MemorySearchQuery {
  query?: string;
  type?: string;
  tags?: string[];
  dateFrom?: string;
  dateTo?: string;
  limit?: number;
  offset?: number;
}

// AI Service types
export interface AIFeatureStatus {
  name: string;
  status: "active" | "inactive" | "error";
  metrics?: Record<string, number>;
  lastUpdate?: string;
}

export interface AIInsight {
  id: string;
  type: string;
  content: string;
  confidence: number;
  created_at: string;
  metadata?: Record<string, any>;
}

// System Service types
export interface SystemHealth {
  status: "healthy" | "degraded" | "down";
  services: Record<string, {
    status: "up" | "down";
    responseTime: number;
    lastCheck: string;
  }>;
  uptime: number;
  version: string;
}

export interface PerformanceMetrics {
  responseTime: {
    avg: number;
    p95: number;
    p99: number;
  };
  requestsPerSecond: number;
  errorRate: number;
  activeConnections: number;
  memoryUsage: number;
  timestamp: string;
}

// Environment configuration
export interface Environment {
  API_BASE_URL: string;
  WS_BASE_URL: string;
  TIMEOUT: number;
  RETRY_ATTEMPTS: number;
  RETRY_DELAY: number;
  CACHE_ENABLED: boolean;
  CACHE_TTL: number;
  LOG_LEVEL: "debug" | "info" | "warn" | "error";
}
