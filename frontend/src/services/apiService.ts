// Enhanced API Service for KnowledgeHub Phase 5
// Integrates all Phase 1-4 backend APIs with optimized caching and error handling

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import { api as baseApi } from './api'

export interface APIConfig {
  baseURL: string
  timeout: number
  retryAttempts: number
  retryDelay: number
}

export interface CacheConfig {
  ttl: number // Time to live in milliseconds
  maxSize: number // Maximum cache size
  enabled: boolean
}

export interface APIResponse<T = any> {
  data: T
  success: boolean
  message?: string
  timestamp: string
  cached?: boolean
  phase?: 'phase1' | 'phase2' | 'phase3' | 'phase4'
}

// API Endpoints for all phases
export const API_ENDPOINTS = {
  // Phase 1: RAG System
  RAG: {
    SEARCH: '/api/rag/search',
    VECTORIZE: '/api/rag/vectorize',
    EMBED: '/api/rag/embed',
    SIMILARITY: '/api/rag/similarity',
    CHUNKS: '/api/rag/chunks',
    PERFORMANCE: '/api/rag/performance',
  },

  // Phase 2: AI Intelligence
  AI: {
    PATTERNS: '/api/ai/patterns',
    LEARNING: '/api/ai/learning',
    INSIGHTS: '/api/ai/insights',
    PREDICTIONS: '/api/ai/predictions',
    DECISIONS: '/api/ai/decisions',
    FEEDBACK: '/api/ai/feedback',
    ANALYTICS: '/api/ai/analytics',
    SESSION_CONTINUITY: '/api/claude-auto/session/current',
    MISTAKE_LEARNING: '/api/mistake-learning/lessons',
    PROACTIVE_ASSISTANT: '/api/proactive/predictions',
    DECISION_REASONING: '/api/decisions/search',
    CODE_EVOLUTION: '/api/code-evolution/history',
    PERFORMANCE_METRICS: '/api/performance/stats',
    PATTERN_RECOGNITION: '/api/patterns/recognize',
    WORKFLOW_INTEGRATION: '/api/claude-workflow/stats',
  },

  // Phase 3: Analytics
  ANALYTICS: {
    DASHBOARD: '/api/analytics/dashboard',
    METRICS: '/api/analytics/metrics',
    TIMESERIES: '/api/analytics/timeseries',
    PERFORMANCE: '/api/analytics/performance',
    REPORTS: '/api/analytics/reports',
    REALTIME: '/api/analytics/realtime',
    TIME_SERIES: '/api/time-series-analytics/metrics',
    PERFORMANCE_TRACKING: '/api/performance-metrics/track',
  },

  // Phase 4: Enterprise Features
  ENTERPRISE: {
    TENANTS: '/api/enterprise/tenants',
    RBAC: '/api/enterprise/rbac',
    SCALING: '/api/enterprise/scaling',
    AUDIT: '/api/enterprise/audit',
    COMPLIANCE: '/api/enterprise/compliance',
    MONITORING: '/api/enterprise/monitoring',
    MULTI_TENANT: '/api/multi-tenant',
    SCALING_METRICS: '/api/scaling/metrics',
  },

  // Core Services
  MEMORY: {
    LIST: '/api/v1/memories',
    SEARCH: '/api/memory/search',
    CREATE: '/api/memory/create',
    STATS: '/api/claude-auto/memory/stats',
    CONTEXT: '/api/memory/context',
  },

  SESSIONS: {
    CURRENT: '/api/claude-auto/session/current',
    CREATE: '/api/claude-auto/session/create',
    HISTORY: '/api/sessions/history',
  },

  SOURCES: {
    LIST: '/api/v1/sources/',
    REFRESH: (id: string) => `/api/v1/sources/${id}/refresh`,
    DELETE: (id: string) => `/api/v1/sources/${id}`,
  },

  SEARCH: {
    PUBLIC: '/api/public/search',
    UNIFIED: '/api/v1/search/unified',
  },

  // Health and monitoring
  HEALTH: '/api/health',
  METRICS: '/api/metrics',
  STATUS: '/api/status',
} as const

class EnhancedAPIService {
  private cache: Map<string, { data: any; timestamp: number; ttl: number }> = new Map()
  private retryQueue: Map<string, { config: AxiosRequestConfig; attempts: number }> = new Map()
  
  private config: APIConfig = {
    baseURL: this.determineBaseURL(),
    timeout: 30000,
    retryAttempts: 3,
    retryDelay: 1000,
  }

  private cacheConfig: CacheConfig = {
    ttl: 5 * 60 * 1000, // 5 minutes default
    maxSize: 1000,
    enabled: true,
  }

  constructor() {
    this.setupInterceptors()
    this.startCacheCleanup()
  }

  private determineBaseURL(): string {
    const hostname = typeof window !== 'undefined' ? window.location.hostname : 'localhost'
    
    if (hostname.startsWith('192.168.1.')) {
      return `http://192.168.1.25:3000`
    } else if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return `http://localhost:3000`
    } else {
      return `http://${hostname}:3000`
    }
  }

  private setupInterceptors(): void {
    // Request interceptor
    baseApi.interceptors.request.use(
      (config) => {
        // Add timestamp and correlation ID
        config.headers = {
          ...config.headers,
          'X-Request-ID': this.generateRequestId(),
          'X-Timestamp': new Date().toISOString(),
          'X-Phase': 'phase5-ui',
        }
        
        return config
      },
      (error) => Promise.reject(error)
    )

    // Response interceptor
    baseApi.interceptors.response.use(
      (response) => {
        // Cache successful responses
        if (this.shouldCache(response.config)) {
          this.setCache(response.config.url || '', response.data)
        }
        
        return response
      },
      async (error) => {
        if (error.response?.status >= 500) {
          // Retry server errors
          return this.retryRequest(error.config, error)
        }
        
        // Try cache for failed requests
        const cachedData = this.getCache(error.config.url || '')
        if (cachedData) {
          return {
            ...error.response,
            data: cachedData,
            fromCache: true,
          }
        }
        
        return Promise.reject(error)
      }
    )
  }

  // Phase 1: RAG System APIs
  async searchRAG(query: string, options?: {
    limit?: number
    threshold?: number
    filters?: Record<string, any>
  }): Promise<APIResponse<any[]>> {
    return this.request({
      method: 'POST',
      url: API_ENDPOINTS.RAG.SEARCH,
      data: { query, ...options },
      cache: { ttl: 2 * 60 * 1000 }, // 2 minutes for search results
    })
  }

  async getRAGPerformance(): Promise<APIResponse<{
    latency: number
    throughput: number
    accuracy: number
    speedup: number
  }>> {
    return this.request({
      url: API_ENDPOINTS.RAG.PERFORMANCE,
      cache: { ttl: 30 * 1000 }, // 30 seconds for performance metrics
    })
  }

  // Phase 2: AI Intelligence APIs
  async getAIInsights(): Promise<APIResponse<any[]>> {
    return this.request({
      url: API_ENDPOINTS.AI.INSIGHTS,
      cache: { ttl: 5 * 60 * 1000 },
    })
  }

  async getPatternRecognition(): Promise<APIResponse<any[]>> {
    return this.request({
      url: API_ENDPOINTS.AI.PATTERN_RECOGNITION,
      cache: { ttl: 10 * 60 * 1000 },
    })
  }

  async getLearningMetrics(): Promise<APIResponse<any>> {
    return this.request({
      url: API_ENDPOINTS.AI.LEARNING,
      cache: { ttl: 2 * 60 * 1000 },
    })
  }

  async getSessionContinuity(): Promise<APIResponse<any>> {
    return this.request({
      url: API_ENDPOINTS.AI.SESSION_CONTINUITY,
      cache: { ttl: 30 * 1000 },
    })
  }

  async getMistakeLearning(): Promise<APIResponse<any[]>> {
    return this.request({
      url: API_ENDPOINTS.AI.MISTAKE_LEARNING,
      cache: { ttl: 5 * 60 * 1000 },
    })
  }

  async getProactiveAssistant(): Promise<APIResponse<any[]>> {
    return this.request({
      url: API_ENDPOINTS.AI.PROACTIVE_ASSISTANT,
      cache: { ttl: 1 * 60 * 1000 },
    })
  }

  // Phase 3: Analytics APIs
  async getDashboardMetrics(): Promise<APIResponse<any>> {
    return this.request({
      url: API_ENDPOINTS.ANALYTICS.DASHBOARD,
      cache: { ttl: 30 * 1000 },
    })
  }

  async getTimeSeriesAnalytics(timeRange: string = '24h'): Promise<APIResponse<any[]>> {
    return this.request({
      url: `${API_ENDPOINTS.ANALYTICS.TIME_SERIES}?range=${timeRange}`,
      cache: { ttl: 60 * 1000 },
    })
  }

  async getPerformanceTracking(): Promise<APIResponse<{
    response_time: number
    throughput: number
    error_rate: number
    uptime: number
  }>> {
    return this.request({
      url: API_ENDPOINTS.ANALYTICS.PERFORMANCE_TRACKING,
      cache: { ttl: 15 * 1000 },
    })
  }

  async getRealtimeMetrics(): Promise<APIResponse<any>> {
    return this.request({
      url: API_ENDPOINTS.ANALYTICS.REALTIME,
      cache: { ttl: 5 * 1000 }, // Very short cache for real-time data
    })
  }

  // Phase 4: Enterprise APIs
  async getMultiTenantMetrics(): Promise<APIResponse<any>> {
    return this.request({
      url: API_ENDPOINTS.ENTERPRISE.MULTI_TENANT,
      cache: { ttl: 60 * 1000 },
    })
  }

  async getScalingMetrics(): Promise<APIResponse<any>> {
    return this.request({
      url: API_ENDPOINTS.ENTERPRISE.SCALING_METRICS,
      cache: { ttl: 30 * 1000 },
    })
  }

  async getAuditLogs(filters?: Record<string, any>): Promise<APIResponse<any[]>> {
    return this.request({
      url: API_ENDPOINTS.ENTERPRISE.AUDIT,
      params: filters,
      cache: { ttl: 2 * 60 * 1000 },
    })
  }

  // Memory System APIs
  async getMemoryStats(): Promise<APIResponse<any>> {
    return this.request({
      url: API_ENDPOINTS.MEMORY.STATS,
      cache: { ttl: 30 * 1000 },
    })
  }

  async searchMemories(query: string, limit = 20): Promise<APIResponse<any[]>> {
    return this.request({
      url: API_ENDPOINTS.MEMORY.SEARCH,
      params: { q: query, limit },
      cache: { ttl: 2 * 60 * 1000 },
    })
  }

  // System Health APIs
  async getSystemHealth(): Promise<APIResponse<{
    status: 'healthy' | 'degraded' | 'critical'
    uptime: number
    version: string
    services: Record<string, boolean>
  }>> {
    return this.request({
      url: API_ENDPOINTS.HEALTH,
      cache: { ttl: 15 * 1000 },
    })
  }

  async getSystemMetrics(): Promise<APIResponse<any>> {
    return this.request({
      url: API_ENDPOINTS.METRICS,
      cache: { ttl: 10 * 1000 },
    })
  }

  // Generic request method with enhanced features
  async request<T = any>(config: AxiosRequestConfig & {
    cache?: { ttl?: number; key?: string }
    retry?: { attempts?: number; delay?: number }
  }): Promise<APIResponse<T>> {
    const requestId = this.generateRequestId()
    const startTime = Date.now()
    
    try {
      // Check cache first
      if (config.cache?.ttl && this.cacheConfig.enabled) {
        const cacheKey = config.cache.key || this.getCacheKey(config)
        const cached = this.getCache(cacheKey)
        
        if (cached) {
          return {
            data: cached,
            success: true,
            timestamp: new Date().toISOString(),
            cached: true,
          }
        }
      }

      // Make request
      const response = await baseApi(config)
      const duration = Date.now() - startTime

      // Cache response
      if (config.cache?.ttl) {
        const cacheKey = config.cache.key || this.getCacheKey(config)
        this.setCache(cacheKey, response.data, config.cache.ttl)
      }

      return {
        data: response.data,
        success: true,
        timestamp: new Date().toISOString(),
        cached: false,
      }
    } catch (error: any) {
      
      // Try cache as fallback
      if (config.cache?.ttl) {
        const cacheKey = config.cache.key || this.getCacheKey(config)
        const cached = this.getCache(cacheKey)
        
        if (cached) {
          return {
            data: cached,
            success: true,
            message: 'Returned cached data due to network error',
            timestamp: new Date().toISOString(),
            cached: true,
          }
        }
      }
      
      throw error
    }
  }

  // Batch requests for improved performance
  async batchRequest<T = any>(requests: Array<{
    key: string
    config: AxiosRequestConfig
  }>): Promise<Record<string, APIResponse<T> | Error>> {
    const results: Record<string, APIResponse<T> | Error> = {}
    
    await Promise.allSettled(
      requests.map(async ({ key, config }) => {
        try {
          results[key] = await this.request(config)
        } catch (error) {
          results[key] = error as Error
        }
      })
    )
    
    return results
  }

  // Cache management
  private getCacheKey(config: AxiosRequestConfig): string {
    const url = config.url || ''
    const method = config.method || 'GET'
    const params = config.params ? JSON.stringify(config.params) : ''
    const data = config.data ? JSON.stringify(config.data) : ''
    
    return `${method}:${url}:${params}:${data}`
  }

  private setCache(key: string, data: any, ttl = this.cacheConfig.ttl): void {
    if (!this.cacheConfig.enabled) return
    
    // Clean cache if it's too large
    if (this.cache.size >= this.cacheConfig.maxSize) {
      this.cleanExpiredCache()
      
      if (this.cache.size >= this.cacheConfig.maxSize) {
        // Remove oldest entries
        const entries = Array.from(this.cache.entries())
        entries.sort((a, b) => a[1].timestamp - b[1].timestamp)
        
        for (let i = 0; i < Math.floor(this.cacheConfig.maxSize * 0.2); i++) {
          this.cache.delete(entries[i][0])
        }
      }
    }
    
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
    })
  }

  private getCache(key: string): any | null {
    if (!this.cacheConfig.enabled) return null
    
    const cached = this.cache.get(key)
    if (!cached) return null
    
    if (Date.now() - cached.timestamp > cached.ttl) {
      this.cache.delete(key)
      return null
    }
    
    return cached.data
  }

  private cleanExpiredCache(): void {
    const now = Date.now()
    
    for (const [key, value] of this.cache.entries()) {
      if (now - value.timestamp > value.ttl) {
        this.cache.delete(key)
      }
    }
  }

  private startCacheCleanup(): void {
    setInterval(() => {
      this.cleanExpiredCache()
    }, 60 * 1000) // Clean every minute
  }

  private shouldCache(config?: AxiosRequestConfig): boolean {
    if (!config || !this.cacheConfig.enabled) return false
    
    const method = config.method?.toUpperCase()
    return method === 'GET'
  }

  private async retryRequest(config: AxiosRequestConfig, error: any): Promise<any> {
    const key = this.getCacheKey(config)
    const existing = this.retryQueue.get(key)
    
    if (existing && existing.attempts >= this.config.retryAttempts) {
      return Promise.reject(error)
    }
    
    const attempts = (existing?.attempts || 0) + 1
    this.retryQueue.set(key, { config, attempts })
    
    const delay = this.config.retryDelay * Math.pow(2, attempts - 1)
    await new Promise(resolve => setTimeout(resolve, delay))
    
    try {
      const response = await baseApi(config)
      this.retryQueue.delete(key)
      return response
    } catch (retryError) {
      if (attempts >= this.config.retryAttempts) {
        this.retryQueue.delete(key)
        return Promise.reject(retryError)
      }
      
      return this.retryRequest(config, retryError)
    }
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  // Public methods for cache management
  clearCache(): void {
    this.cache.clear()
  }

  getCacheStats(): {
    size: number
    maxSize: number
    hitRate: number
    memoryUsage: number
  } {
    const memoryUsage = JSON.stringify([...this.cache.entries()]).length
    
    return {
      size: this.cache.size,
      maxSize: this.cacheConfig.maxSize,
      hitRate: 0, // Would need tracking for real implementation
      memoryUsage,
    }
  }

  updateConfig(config: Partial<APIConfig>): void {
    this.config = { ...this.config, ...config }
  }

  updateCacheConfig(config: Partial<CacheConfig>): void {
    this.cacheConfig = { ...this.cacheConfig, ...config }
  }
}

// Create singleton instance
export const apiService = new EnhancedAPIService()
export default apiService