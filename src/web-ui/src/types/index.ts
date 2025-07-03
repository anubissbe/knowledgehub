// Source types
export interface Source {
  id: string
  url: string
  name: string
  type: string
  status: string
  authentication?: Record<string, any>
  crawl_config?: {
    max_depth: number
    max_pages: number
    follow_patterns: string[]
    exclude_patterns: string[]
  }
  refresh_interval: number
  statistics?: {
    total_pages: number
    total_chunks: number
    error_count: number
    last_crawl_pages?: number
    last_crawl_chunks?: number
  }
  created_at: string
  updated_at: string
  last_crawled_at?: string
}

export interface SourceCreate {
  url: string
  name: string
  type: string
  authentication?: Record<string, any>
  crawl_config?: Record<string, any>
  refresh_interval: number
}

// Search types
export interface SearchResult {
  content: string
  source_name: string
  url: string
  score: number
  chunk_type: string
  metadata?: Record<string, any>
}

export interface SearchResponse {
  query: string
  results: SearchResult[]
  total: number
  search_time_ms: number
}

// Job types
export interface Job {
  id: string
  source_id: string
  type: string
  status: string
  config: Record<string, any>
  result?: Record<string, any>
  error?: string
  created_at: string
  started_at?: string
  completed_at?: string
  duration?: number
}

// Memory types
export interface Memory {
  id: string
  content: string
  content_hash: string
  tags: string[]
  metadata: Record<string, any>
  embedding_id?: string
  access_count: number
  created_at: string
  updated_at: string
  accessed_at: string
}

// Dashboard types
export interface DashboardStats {
  sources: number
  activeJobs: number
  totalChunks: number
  memories: number
  systemHealth: {
    api: boolean
    database: boolean
    redis: boolean
    weaviate: boolean
  }
}

// MCP types
export interface Tool {
  name: string
  description: string
  parameters: Record<string, any>
}

export interface ToolCall {
  tool: string
  arguments: Record<string, any>
}