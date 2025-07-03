import axios from 'axios'
import { Source, SourceCreate, Job, Memory } from '@/types'

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'dev-api-key-123', // Development API key
  },
})

// Request interceptor for auth
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export const api = {
  // Dashboard - compute stats from sources and jobs
  getDashboardStats: async () => {
    // Since there's no dedicated dashboard endpoint, compute stats from other endpoints
    try {
      const [sourcesResponse, jobsResponse] = await Promise.all([
        apiClient.get('/v1/sources/'),
        apiClient.get('/v1/jobs/')
      ])
      
      const sources = sourcesResponse.data.sources || []
      const jobs = jobsResponse.data.jobs || []
      
      return {
        total_sources: sources.length,
        active_sources: sources.filter((s: any) => s.status === 'active').length,
        total_documents: sources.reduce((acc: number, s: any) => acc + (s.stats?.documents || 0), 0),
        total_chunks: sources.reduce((acc: number, s: any) => acc + (s.stats?.chunks || 0), 0),
        pending_jobs: jobs.filter((j: any) => j.status === 'pending').length,
        running_jobs: jobs.filter((j: any) => j.status === 'running').length,
        completed_jobs: jobs.filter((j: any) => j.status === 'completed').length,
        failed_jobs: jobs.filter((j: any) => j.status === 'failed').length
      }
    } catch (error) {
      console.error('Failed to fetch dashboard stats:', error)
      return {
        total_sources: 0,
        active_sources: 0,
        total_documents: 0,
        total_chunks: 0,
        pending_jobs: 0,
        running_jobs: 0,
        completed_jobs: 0,
        failed_jobs: 0
      }
    }
  },

  // Sources
  getSources: async () => {
    const { data } = await apiClient.get('/v1/sources/')
    return data
  },

  getSource: async (id: string) => {
    const { data } = await apiClient.get<Source>(`/v1/sources/${id}`)
    return data
  },

  createSource: async (source: SourceCreate) => {
    const { data } = await apiClient.post<Source>('/v1/sources/', source)
    return data
  },

  updateSource: async (id: string, source: Partial<SourceCreate>) => {
    const { data } = await apiClient.patch<Source>(`/v1/sources/${id}`, source)
    return data
  },

  deleteSource: async (id: string) => {
    await apiClient.delete(`/v1/sources/${id}`)
  },

  refreshSource: async (id: string) => {
    const { data } = await apiClient.post(`/v1/sources/${id}/refresh`)
    return data
  },

  // Search
  search: async (query: string, searchType: string = 'hybrid') => {
    const { data } = await apiClient.post('/v1/search', {
      query,
      search_type: searchType,
      limit: 20,
    })
    return data
  },

  // Jobs - Note: API doesn't have a list endpoint, so we'll return empty for now
  getJobs: async (_params?: { source_id?: string; status?: string }) => {
    // The API doesn't have a jobs list endpoint, return empty array
    return { jobs: [], total: 0 }
  },

  getJob: async (id: string) => {
    const { data } = await apiClient.get<Job>(`/v1/jobs/${id}`)
    return data
  },

  retryJob: async (id: string) => {
    const { data } = await apiClient.post(`/v1/jobs/${id}/retry`)
    return data
  },

  cancelJob: async (id: string) => {
    const { data } = await apiClient.post(`/v1/jobs/${id}/cancel`)
    return data
  },

  // Memories
  getMemories: async (params?: { source_id?: string; search?: string }) => {
    const { data } = await apiClient.get<Memory[]>('/v1/memories/', { params })
    return data
  },

  getMemory: async (id: string) => {
    const { data } = await apiClient.get<Memory>(`/v1/memories/${id}`)
    return data
  },

  createMemory: async (memory: Partial<Memory>) => {
    const { data } = await apiClient.post<Memory>('/v1/memories/', memory)
    return data
  },

  updateMemory: async (id: string, memory: Partial<Memory>) => {
    const { data } = await apiClient.patch<Memory>(`/v1/memories/${id}`, memory)
    return data
  },

  deleteMemory: async (id: string) => {
    await apiClient.delete(`/v1/memories/${id}`)
  },

  // WebSocket connection for real-time updates
  connectWebSocket: () => {
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:3000'
    return new WebSocket(`${wsUrl}/ws`)
  },
}