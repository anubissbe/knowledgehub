import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for auth token if needed
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

export const apiService = {
  // Health check
  getHealth: async () => {
    const response = await apiClient.get('/health');
    return response.data;
  },

  // AI Features
  getAiFeaturesSummary: async () => {
    const response = await apiClient.get('/api/ai-features/summary');
    return response.data;
  },

  // Session Management
  startSession: async (cwd: string) => {
    const response = await apiClient.post('/api/claude-auto/session/start', null, {
      params: { cwd },
    });
    return response.data;
  },

  endSession: async (sessionId: string) => {
    const response = await apiClient.post(`/api/claude-auto/session/end/${sessionId}`);
    return response.data;
  },

  // Mistake Learning
  getMistakes: async (userId?: string) => {
    const response = await apiClient.get('/api/mistake-learning/mistakes', {
      params: { user_id: userId },
    });
    return response.data;
  },

  trackMistake: async (mistake: any) => {
    const response = await apiClient.post('/api/mistake-learning/track', mistake);
    return response.data;
  },

  // Decision Tracking
  getDecisions: async (userId?: string) => {
    const response = await apiClient.get('/api/decisions', {
      params: { user_id: userId },
    });
    return response.data;
  },

  recordDecision: async (decision: any) => {
    const response = await apiClient.post('/api/decisions/record', decision);
    return response.data;
  },

  // Performance Metrics
  getPerformanceMetrics: async (userId?: string) => {
    const response = await apiClient.get('/api/performance', {
      params: { user_id: userId },
    });
    return response.data;
  },

  // Search
  search: async (query: string, filters?: any) => {
    const response = await apiClient.post('/api/search', {
      query,
      ...filters,
    });
    return response.data;
  },

  // Memory System
  getMemories: async (sessionId?: string) => {
    const response = await apiClient.get('/api/memory', {
      params: { session_id: sessionId },
    });
    return response.data;
  },

  createMemory: async (memory: any) => {
    const response = await apiClient.post('/api/memory', memory);
    return response.data;
  },

  // Knowledge Sources
  getSources: async () => {
    const response = await apiClient.get('/api/sources');
    return response.data;
  },

  createSource: async (source: any) => {
    const response = await apiClient.post('/api/sources', source);
    return response.data;
  },

  syncSource: async (sourceId: string) => {
    const response = await apiClient.post(`/api/sources/${sourceId}/sync`);
    return response.data;
  },

  // WebSocket connection for real-time updates
  connectWebSocket: (onMessage: (data: any) => void) => {
    const ws = new WebSocket(`${API_BASE_URL.replace('http', 'ws')}/ws`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return ws;
  },
};

export default apiService;