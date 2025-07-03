import axios from 'axios'
import { Source, SourceCreate, Job, Memory } from '@/types'

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
  headers: {
    'Content-Type': 'application/json',
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
    try {
      const [sourcesResponse, jobsResponse] = await Promise.all([
        apiClient.get('/api/v1/sources/'),
        apiClient.get('/api/v1/jobs/')
      ])
      
      // Ensure sources is always an array
      const sourcesData = sourcesResponse.data
      const sources = Array.isArray(sourcesData?.sources) ? sourcesData.sources : []
      
      // Ensure jobs is always an array
      const jobsData = jobsResponse.data
      const jobs = Array.isArray(jobsData?.jobs) ? jobsData.jobs : []
      
      return {
        total_sources: sources.length,
        active_sources: sources.filter((s: any) => s.status === 'active').length,
        pending_sources: sources.filter((s: any) => s.status === 'pending').length,
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
        pending_sources: 0,
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
    try {
      const { data } = await apiClient.get('/api/v1/sources/')
      // Ensure we always return the full response with sources array
      return {
        sources: Array.isArray(data?.sources) ? data.sources : [],
        total: data?.total || 0,
        skip: data?.skip || 0,
        limit: data?.limit || 100
      }
    } catch (error) {
      console.error('Error fetching sources:', error)
      return {
        sources: [],
        total: 0,
        skip: 0,
        limit: 100
      }
    }
  },

  getSource: async (id: string) => {
    const { data } = await apiClient.get<Source>(`/api/v1/sources/${id}`)
    return data
  },

  createSource: async (source: SourceCreate) => {
    const { data } = await apiClient.post<Source>('/api/v1/sources/', source)
    return data
  },

  updateSource: async (id: string, source: Partial<SourceCreate>) => {
    const { data } = await apiClient.patch<Source>(`/api/v1/sources/${id}`, source)
    return data
  },

  deleteSource: async (id: string) => {
    await apiClient.delete(`/api/v1/sources/${id}`)
  },

  refreshSource: async (id: string) => {
    const { data } = await apiClient.post(`/api/v1/sources/${id}/refresh`)
    return data
  },

  // Search
  search: async (query: string, searchType: string = 'hybrid') => {
    const { data } = await apiClient.post('/api/v1/search', {
      query,
      search_type: searchType,
      limit: 20,
    })
    return data
  },

  // Jobs
  getJobs: async (params?: { source_id?: string; status?: string; skip?: number; limit?: number }) => {
    try {
      const { data } = await apiClient.get('/api/v1/jobs/', { params })
      return {
        jobs: Array.isArray(data?.jobs) ? data.jobs : [],
        total: data?.total || 0,
        skip: data?.skip || 0,
        limit: data?.limit || 100
      }
    } catch (error) {
      console.error('Error fetching jobs:', error)
      return {
        jobs: [],
        total: 0,
        skip: 0,
        limit: 100
      }
    }
  },

  getJob: async (id: string) => {
    const { data } = await apiClient.get<Job>(`/api/v1/jobs/${id}`)
    return data
  },

  retryJob: async (id: string) => {
    const { data } = await apiClient.post(`/api/v1/jobs/${id}/retry`)
    return data
  },

  cancelJob: async (id: string) => {
    const { data } = await apiClient.post(`/api/v1/jobs/${id}/cancel`)
    return data
  },

  // Memories - endpoint may not exist, return empty array
  getMemories: async (params?: { source_id?: string; search?: string }) => {
    try {
      const { data } = await apiClient.get('/api/v1/memories/', { params })
      return Array.isArray(data) ? data : (data?.memories || [])
    } catch (error) {
      console.warn('Memories endpoint not available:', error)
      return []
    }
  },

  getMemory: async (id: string) => {
    const { data } = await apiClient.get<Memory>(`/api/v1/memories/${id}`)
    return data
  },

  createMemory: async (memory: Partial<Memory>) => {
    const { data } = await apiClient.post<Memory>('/api/v1/memories/', memory)
    return data
  },

  updateMemory: async (id: string, memory: Partial<Memory>) => {
    const { data } = await apiClient.patch<Memory>(`/api/v1/memories/${id}`, memory)
    return data
  },

  deleteMemory: async (id: string) => {
    await apiClient.delete(`/api/v1/memories/${id}`)
  },

  // WebSocket connection for real-time updates
  connectWebSocket: () => {
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:3000'
    return new WebSocket(`${wsUrl}/ws`)
  },
}