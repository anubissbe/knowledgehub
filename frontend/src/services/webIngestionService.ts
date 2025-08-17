/**
 * Web Ingestion Service
 * Handles Firecrawl job monitoring and web content ingestion
 */

import { api } from './api'

// Types and Interfaces
export enum IngestionStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
  PAUSED = 'paused'
}

export enum IngestionMode {
  SINGLE_PAGE = 'single_page',
  SITEMAP = 'sitemap',
  CRAWL = 'crawl',
  BATCH_URLS = 'batch_urls'
}

export interface IngestionJob {
  job_id: string
  name: string
  status: IngestionStatus
  mode: IngestionMode
  source_urls: string[]
  target_urls?: string[]
  config: {
    max_pages?: number
    depth_limit?: number
    include_patterns?: string[]
    exclude_patterns?: string[]
    respect_robots_txt?: boolean
    rate_limit?: number
    timeout?: number
    extract_images?: boolean
    extract_links?: boolean
    extract_metadata?: boolean
  }
  progress: {
    total_urls: number
    processed_urls: number
    successful_urls: number
    failed_urls: number
    percentage: number
  }
  start_time: string
  end_time?: string
  duration?: number
  created_by?: string
  metadata?: Record<string, any>
}

export interface IngestionResult {
  url: string
  status: 'success' | 'failed' | 'skipped'
  title?: string
  content?: string
  metadata?: {
    word_count?: number
    language?: string
    content_type?: string
    last_modified?: string
    images?: string[]
    links?: string[]
    headers?: Record<string, string>
  }
  error?: string
  processing_time?: number
  extracted_at: string
}

export interface IngestionPipeline {
  id: string
  name: string
  description: string
  steps: {
    name: string
    type: 'extraction' | 'processing' | 'validation' | 'storage'
    config: Record<string, any>
    enabled: boolean
  }[]
  default_config: Record<string, any>
  created_at: string
  updated_at: string
}

export interface IngestionAnalytics {
  total_jobs: number
  active_jobs: number
  completed_jobs: number
  failed_jobs: number
  total_pages_processed: number
  success_rate: number
  avg_processing_time: number
  data_volume: {
    total_size: number
    avg_page_size: number
    content_types: Record<string, number>
  }
  performance_metrics: {
    pages_per_hour: number
    peak_processing_rate: number
    error_rate: number
  }
  timeline: {
    date: string
    jobs_started: number
    jobs_completed: number
    pages_processed: number
    errors: number
  }[]
}

export interface WebSource {
  id: string
  name: string
  base_url: string
  type: 'website' | 'api' | 'rss' | 'sitemap'
  status: 'active' | 'paused' | 'error'
  config: {
    crawl_frequency: 'hourly' | 'daily' | 'weekly' | 'monthly'
    last_crawl?: string
    next_crawl?: string
    include_patterns?: string[]
    exclude_patterns?: string[]
    auth_config?: Record<string, any>
  }
  statistics: {
    total_pages: number
    last_update: string
    avg_content_freshness: number
    error_count: number
  }
  created_at: string
  updated_at: string
}

export interface ContentValidation {
  url: string
  validations: {
    type: 'content_quality' | 'duplicate_detection' | 'language_detection' | 'format_validation'
    status: 'passed' | 'failed' | 'warning'
    score?: number
    message: string
    details?: Record<string, any>
  }[]
  overall_score: number
  recommendation: 'accept' | 'review' | 'reject'
}

export class WebIngestionService {
  private baseUrl = '/api/v1/ingestion'

  /**
   * Create a new ingestion job
   */
  async createJob(jobConfig: {
    name: string
    mode: IngestionMode
    source_urls: string[]
    config?: Partial<IngestionJob['config']>
    pipeline_id?: string
  }): Promise<IngestionJob> {
    const response = await api.post(`${this.baseUrl}/jobs`, jobConfig)
    return response.data
  }

  /**
   * Get ingestion job status
   */
  async getJob(job_id: string): Promise<IngestionJob> {
    const response = await api.get(`${this.baseUrl}/jobs/${job_id}`)
    return response.data
  }

  /**
   * Get all ingestion jobs
   */
  async getJobs(params?: {
    status?: IngestionStatus
    mode?: IngestionMode
    limit?: number
    offset?: number
    sort_by?: 'created_at' | 'start_time' | 'progress'
    sort_order?: 'asc' | 'desc'
  }): Promise<{
    jobs: IngestionJob[]
    total: number
    has_more: boolean
  }> {
    const searchParams = new URLSearchParams()
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString())
        }
      })
    }

    const response = await api.get(`${this.baseUrl}/jobs?${searchParams}`)
    return response.data
  }

  /**
   * Start an ingestion job
   */
  async startJob(job_id: string): Promise<{ success: boolean }> {
    const response = await api.post(`${this.baseUrl}/jobs/${job_id}/start`)
    return response.data
  }

  /**
   * Pause an ingestion job
   */
  async pauseJob(job_id: string): Promise<{ success: boolean }> {
    const response = await api.post(`${this.baseUrl}/jobs/${job_id}/pause`)
    return response.data
  }

  /**
   * Cancel an ingestion job
   */
  async cancelJob(job_id: string): Promise<{ success: boolean }> {
    const response = await api.post(`${this.baseUrl}/jobs/${job_id}/cancel`)
    return response.data
  }

  /**
   * Get job results
   */
  async getJobResults(
    job_id: string,
    params?: {
      status?: 'success' | 'failed' | 'skipped'
      limit?: number
      offset?: number
    }
  ): Promise<{
    results: IngestionResult[]
    total: number
    has_more: boolean
  }> {
    const searchParams = new URLSearchParams()
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString())
        }
      })
    }

    const response = await api.get(`${this.baseUrl}/jobs/${job_id}/results?${searchParams}`)
    return response.data
  }

  /**
   * Get real-time job monitoring data
   */
  async getJobMonitoring(job_id: string): Promise<{
    current_url?: string
    processing_rate: number
    estimated_completion: string
    resource_usage: {
      cpu_usage: number
      memory_usage: number
      network_usage: number
    }
    recent_errors: {
      url: string
      error: string
      timestamp: string
    }[]
    queue_status: {
      pending: number
      processing: number
      completed: number
    }
  }> {
    const response = await api.get(`${this.baseUrl}/jobs/${job_id}/monitoring`)
    return response.data
  }

  /**
   * Get ingestion analytics
   */
  async getAnalytics(timeframe: string = '24h'): Promise<IngestionAnalytics> {
    const response = await api.get(`${this.baseUrl}/analytics?timeframe=${timeframe}`)
    return response.data
  }

  /**
   * Get web sources
   */
  async getWebSources(): Promise<WebSource[]> {
    const response = await api.get(`${this.baseUrl}/sources`)
    return response.data
  }

  /**
   * Create a web source
   */
  async createWebSource(source: Omit<WebSource, 'id' | 'statistics' | 'created_at' | 'updated_at'>): Promise<WebSource> {
    const response = await api.post(`${this.baseUrl}/sources`, source)
    return response.data
  }

  /**
   * Update a web source
   */
  async updateWebSource(id: string, updates: Partial<WebSource>): Promise<WebSource> {
    const response = await api.put(`${this.baseUrl}/sources/${id}`, updates)
    return response.data
  }

  /**
   * Delete a web source
   */
  async deleteWebSource(id: string): Promise<{ success: boolean }> {
    const response = await api.delete(`${this.baseUrl}/sources/${id}`)
    return response.data
  }

  /**
   * Trigger source crawl
   */
  async triggerSourceCrawl(id: string): Promise<IngestionJob> {
    const response = await api.post(`${this.baseUrl}/sources/${id}/crawl`)
    return response.data
  }

  /**
   * Get ingestion pipelines
   */
  async getPipelines(): Promise<IngestionPipeline[]> {
    const response = await api.get(`${this.baseUrl}/pipelines`)
    return response.data
  }

  /**
   * Create ingestion pipeline
   */
  async createPipeline(pipeline: Omit<IngestionPipeline, 'id' | 'created_at' | 'updated_at'>): Promise<IngestionPipeline> {
    const response = await api.post(`${this.baseUrl}/pipelines`, pipeline)
    return response.data
  }

  /**
   * Validate content
   */
  async validateContent(
    urls: string[],
    validation_types?: string[]
  ): Promise<ContentValidation[]> {
    const response = await api.post(`${this.baseUrl}/validate`, {
      urls,
      validation_types
    })
    return response.data
  }

  /**
   * Preview ingestion result
   */
  async previewIngestion(
    url: string,
    config?: Partial<IngestionJob['config']>
  ): Promise<{
    preview: IngestionResult
    estimated_processing_time: number
    suggested_config: Record<string, any>
  }> {
    const response = await api.post(`${this.baseUrl}/preview`, {
      url,
      config
    })
    return response.data
  }

  /**
   * Get system status
   */
  async getSystemStatus(): Promise<{
    status: 'healthy' | 'degraded' | 'down'
    active_jobs: number
    queue_length: number
    processing_rate: number
    error_rate: number
    system_load: {
      cpu_usage: number
      memory_usage: number
      disk_usage: number
    }
    services: {
      firecrawl: 'up' | 'down'
      database: 'up' | 'down'
      queue: 'up' | 'down'
      storage: 'up' | 'down'
    }
  }> {
    const response = await api.get(`${this.baseUrl}/status`)
    return response.data
  }
}

export const webIngestionService = new WebIngestionService()