import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || window.location.origin

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add auth token if available
api.interceptors.request.use((config) => {
  const settings = localStorage.getItem('knowledgehub_settings')
  if (settings) {
    const { apiKey } = JSON.parse(settings)
    if (apiKey) {
      config.headers.Authorization = `Bearer ${apiKey}`
    }
  }
  return config
})

// Handle errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      console.error('Authentication error')
      // Could redirect to login or show notification
    }
    return Promise.reject(error)
  }
)