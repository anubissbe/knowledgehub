
import axios from 'axios';
import { API_CONFIG } from './apiConfig';

// Enhanced API service with caching and performance optimization
class APIService {
  private cache = new Map<string, { data: any; timestamp: number }>();
  private cacheTimeout = 5 * 60 * 1000; // 5 minutes

  constructor() {
    this.setupAxiosConfig();
  }

  private setupAxiosConfig() {
    axios.defaults.baseURL = API_CONFIG.BASE_URL;
    axios.defaults.timeout = API_CONFIG.TIMEOUT;
    
    // Add performance monitoring
    axios.interceptors.request.use((config) => {
      config.metadata = { startTime: new Date() };
      return config;
    });

    axios.interceptors.response.use(
      (response) => {
        const endTime = new Date();
        const duration = endTime.getTime() - response.config.metadata.startTime.getTime();
        console.log(`API Call: ${response.config.url} took ${duration}ms`);
        return response;
      },
      (error) => {
        console.error('API Error:', error);
        return Promise.reject(error);
      }
    );
  }

  private getCacheKey(url: string, params?: any): string {
    return `${url}_${JSON.stringify(params || {})}`;
  }

  private isValidCache(timestamp: number): boolean {
    return Date.now() - timestamp < this.cacheTimeout;
  }

  async get<T>(url: string, params?: any, useCache = true): Promise<{ data: T }> {
    const cacheKey = this.getCacheKey(url, params);
    
    if (useCache && this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey)!;
      if (this.isValidCache(cached.timestamp)) {
        return { data: cached.data };
      }
    }

    try {
      const response = await axios.get(url, { params });
      
      if (useCache) {
        this.cache.set(cacheKey, {
          data: response.data,
          timestamp: Date.now()
        });
      }
      
      return { data: response.data };
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async post<T>(url: string, data?: any): Promise<{ data: T }> {
    try {
      const response = await axios.post(url, data);
      return { data: response.data };
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async put<T>(url: string, data?: any): Promise<{ data: T }> {
    try {
      const response = await axios.put(url, data);
      return { data: response.data };
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async delete<T>(url: string): Promise<{ data: T }> {
    try {
      const response = await axios.delete(url);
      return { data: response.data };
    } catch (error) {
      throw this.handleError(error);
    }
  }

  private handleError(error: any): Error {
    if (error.response) {
      // Server responded with error status
      const message = error.response.data?.message || error.response.statusText;
      return new Error(`API Error (${error.response.status}): ${message}`);
    } else if (error.request) {
      // Request was made but no response received
      return new Error('Network Error: No response from server');
    } else {
      // Something else happened
      return new Error(`Request Error: ${error.message}`);
    }
  }

  // Health check method
  async healthCheck(): Promise<boolean> {
    try {
      await this.get('/health', {}, false);
      return true;
    } catch {
      return false;
    }
  }

  // Clear cache method
  clearCache() {
    this.cache.clear();
  }
}

export const apiService = new APIService();
export const api = apiService; // Alias for backward compatibility
export default apiService;
