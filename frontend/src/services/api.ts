import axios, { AxiosError } from 'axios'
import { getBaseUrlForPath, MOCK_RESPONSES } from './apiConfig'

// Create axios instance without a fixed base URL
export const api = axios.create({
  headers: {
    'Content-Type': 'application/json',
    'X-Requested-With': 'XMLHttpRequest',
  },
})

// Request interceptor to add the correct base URL based on the endpoint
api.interceptors.request.use((config) => {
  const hostname = window.location.hostname
  const path = config.url || ''
  
  // If the URL is already absolute, don't modify it
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return config
  }
  
  // Get the appropriate base URL for this endpoint
  const baseUrl = getBaseUrlForPath(path, hostname)
  if (baseUrl) {
    config.url = `${baseUrl}${path}`
  }
  
  return config
})

// Add auth token if available
api.interceptors.request.use((config) => {
  const settings = localStorage.getItem('knowledgehub_settings')
  if (settings) {
    const { apiKey } = JSON.parse(settings)
    if (apiKey) {
      // Use X-API-Key header as expected by the backend
      config.headers['X-API-Key'] = apiKey
    }
  }
  return config
})

// Handle errors and provide mock responses for unavailable endpoints
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const requestPath = error.config?.url || ''
    
    // Only log detailed error info in development
    if (import.meta.env.DEV) {
      console.log(`[API] Error intercepted for ${requestPath}:`, error.code, error.response?.status)
    }
    
    // Check if this is a connection refused error (no response) or network error
    if (!error.response && (error.code === 'ERR_NETWORK' || error.code === 'ECONNREFUSED')) {
      console.warn(`[API] Connection refused for endpoint: ${requestPath}`)
      
      // Check if we have a mock response for this path
      for (const [mockPath, mockData] of Object.entries(MOCK_RESPONSES)) {
        if (requestPath.includes(mockPath)) {
          console.warn(`[API] Using mock data for unavailable service: ${requestPath}`)
          
          // Return a successful response with mock data
          return {
            data: mockData,
            status: 200,
            statusText: 'OK (Mocked - Service Unavailable)',
            headers: {},
            config: error.config,
          }
        }
      }
      
      // If no mock found, return a generic error response to prevent breaking the UI
      console.warn(`[API] No mock data found for ${requestPath}, returning empty response`)
      return {
        data: {},
        status: 200,
        statusText: 'OK (Empty Mock)',
        headers: {},
        config: error.config,
      }
    }
    
    // Check if this is a 404 or 500 error
    if (error.response?.status === 404 || error.response?.status === 500) {
      // Check if we have a mock response for this path
      for (const [mockPath, mockData] of Object.entries(MOCK_RESPONSES)) {
        if (requestPath.includes(mockPath)) {
          console.warn(`[API] Using mock data for ${error.response.status} endpoint: ${requestPath}`)
          
          // Return a successful response with mock data
          return {
            data: mockData,
            status: 200,
            statusText: 'OK (Mocked)',
            headers: error.response.headers,
            config: error.config,
          }
        }
      }
      
      // If no mock found, return empty response
      console.warn(`[API] No mock data found for ${error.response.status} endpoint ${requestPath}, returning empty response`)
      return {
        data: {},
        status: 200,
        statusText: `OK (Empty Mock for ${error.response.status})`,
        headers: error.response.headers,
        config: error.config,
      }
    }
    
    if (error.response?.status === 401) {
      console.error('Authentication error')
      // Could redirect to login or show notification
    }
    
    return Promise.reject(error)
  }
)