/**
 * Hybrid RAG Service
 * Handles communication with the enhanced RAG API endpoints
 */

import { api } from './api'

// Types and Interfaces
export enum RetrievalMode {
  VECTOR_ONLY = 'vector_only',
  SPARSE_ONLY = 'sparse_only',
  GRAPH_ONLY = 'graph_only',
  HYBRID_VECTOR_SPARSE = 'hybrid_vector_sparse',
  HYBRID_VECTOR_GRAPH = 'hybrid_vector_graph',
  HYBRID_ALL = 'hybrid_all'
}

export enum RerankingModel {
  CROSS_ENCODER = 'cross_encoder',
  COHERE = 'cohere',
  JINA = 'jina'
}

export interface HybridRAGRequest {
  query: string
  retrieval_mode: RetrievalMode
  top_k?: number
  include_reasoning?: boolean
  enable_reranking?: boolean
  session_id?: string
  context?: Record<string, any>
}

export interface RetrievalResult {
  id: string
  content: string
  metadata: Record<string, any>
  score: number
  source: string
  retrieval_method: string
}

export interface ReasoningStep {
  step: number
  description: string
  method: string
  results_count: number
  processing_time: number
}

export interface HybridRAGResponse {
  query: string
  mode: RetrievalMode
  results: RetrievalResult[]
  reasoning_steps?: ReasoningStep[]
  total_results: number
  processing_time: number
  session_id: string
  metadata: {
    vector_results?: number
    sparse_results?: number
    graph_results?: number
    reranking_applied?: boolean
    query_expansion?: string[]
  }
}

export interface RAGAnalytics {
  total_queries: number
  avg_response_time: number
  mode_distribution: Record<RetrievalMode, number>
  quality_metrics: {
    relevance_score: number
    precision_at_k: number
    recall_at_k: number
    mrr: number // Mean Reciprocal Rank
  }
  performance_metrics: {
    avg_vector_time: number
    avg_sparse_time: number
    avg_graph_time: number
    avg_reranking_time: number
  }
}

export interface ComparisonResult {
  query: string
  modes: RetrievalMode[]
  results: Record<RetrievalMode, RetrievalResult[]>
  analysis: {
    overlap_percentage: number
    unique_results: Record<RetrievalMode, number>
    quality_comparison: Record<RetrievalMode, number>
  }
}

export class HybridRAGService {
  private baseUrl = '/api/rag/enhanced'

  /**
   * Execute a hybrid RAG query
   */
  async query(request: HybridRAGRequest): Promise<HybridRAGResponse> {
    const response = await api.post(`${this.baseUrl}/query`, request)
    return response.data
  }

  /**
   * Execute a streaming RAG query
   */
  async queryStream(request: HybridRAGRequest): Promise<ReadableStream> {
    const response = await fetch(`${this.baseUrl}/query/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })

    if (!response.body) {
      throw new Error('No response body for streaming query')
    }

    return response.body
  }

  /**
   * Compare different retrieval modes for the same query
   */
  async compareRetrievalModes(
    query: string,
    modes: RetrievalMode[],
    top_k: number = 10
  ): Promise<ComparisonResult> {
    const response = await api.post(`${this.baseUrl}/compare`, {
      query,
      modes,
      top_k
    })
    return response.data
  }

  /**
   * Get analytics for RAG performance
   */
  async getAnalytics(
    timeframe: string = '24h',
    session_id?: string
  ): Promise<RAGAnalytics> {
    const params = new URLSearchParams({ timeframe })
    if (session_id) params.append('session_id', session_id)
    
    const response = await api.get(`${this.baseUrl}/analytics?${params}`)
    return response.data
  }

  /**
   * Get query suggestions based on current context
   */
  async getSuggestions(
    partial_query: string,
    context?: Record<string, any>
  ): Promise<string[]> {
    const response = await api.post(`${this.baseUrl}/suggestions`, {
      partial_query,
      context
    })
    return response.data.suggestions
  }

  /**
   * Explain retrieval results
   */
  async explainResults(
    query: string,
    result_ids: string[]
  ): Promise<{
    explanations: Record<string, {
      relevance_factors: string[]
      score_breakdown: Record<string, number>
      similar_queries: string[]
    }>
  }> {
    const response = await api.post(`${this.baseUrl}/explain`, {
      query,
      result_ids
    })
    return response.data
  }

  /**
   * Provide feedback on retrieval results
   */
  async provideFeedback(
    query: string,
    session_id: string,
    feedback: {
      result_id: string
      rating: number // 1-5
      comments?: string
      is_relevant: boolean
    }[]
  ): Promise<{ success: boolean }> {
    const response = await api.post(`${this.baseUrl}/feedback`, {
      query,
      session_id,
      feedback
    })
    return response.data
  }

  /**
   * Get real-time retrieval metrics
   */
  async getRealtimeMetrics(): Promise<{
    active_queries: number
    avg_response_time: number
    error_rate: number
    cache_hit_rate: number
    queue_length: number
  }> {
    const response = await api.get(`${this.baseUrl}/metrics/realtime`)
    return response.data
  }

  /**
   * Configure retrieval parameters
   */
  async updateConfiguration(config: {
    default_mode?: RetrievalMode
    default_top_k?: number
    enable_caching?: boolean
    reranking_threshold?: number
    similarity_threshold?: number
  }): Promise<{ success: boolean }> {
    const response = await api.put(`${this.baseUrl}/config`, config)
    return response.data
  }
}

export const hybridRAGService = new HybridRAGService()