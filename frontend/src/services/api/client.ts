// Unified HTTP API client with interceptors, caching, and error handling

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from "axios";
import { ApiResponse, ApiError, RequestConfig } from "../types";
import { environment } from "../config/environment";
import { ErrorHandler, AuthError, TimeoutError, NetworkError, withRetry } from "../errors";

// Request cache interface
interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

// Response interceptor interface
type ResponseInterceptor<T = any> = (
  response: AxiosResponse<T>
) => AxiosResponse<T> | Promise<AxiosResponse<T>>;

type RequestInterceptor = (
  config: AxiosRequestConfig
) => AxiosRequestConfig | Promise<AxiosRequestConfig>;

// Unified API Client class
export class ApiClient {
  private client: AxiosInstance;
  private cache = new Map<string, CacheEntry<any>>();
  private requestInterceptors: RequestInterceptor[] = [];
  private responseInterceptors: ResponseInterceptor[] = [];

  constructor() {
    this.client = axios.create({
      baseURL: environment.API_BASE_URL,
      timeout: environment.TIMEOUT,
      headers: {
        "Content-Type": "application/json",
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor for auth token
    this.client.interceptors.request.use(
      (config) => {
        const token = this.getAuthToken();
        if (token) {
          config.headers = config.headers || {};
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(this.handleError(error))
    );

    // Response interceptor for error handling and token refresh
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };

        // Handle 401 errors with token refresh
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            await this.refreshToken();
            const token = this.getAuthToken();
            if (token && originalRequest.headers) {
              originalRequest.headers.Authorization = `Bearer ${token}`;
            }
            return this.client(originalRequest);
          } catch (refreshError) {
            this.clearAuth();
            throw new AuthError("Session expired. Please log in again.");
          }
        }

        throw this.handleError(error);
      }
    );
  }

  // Auth token management
  private getAuthToken(): string | null {
    return localStorage.getItem("knowledgehub_token");
  }

  private async refreshToken(): Promise<void> {
    const refreshToken = localStorage.getItem("knowledgehub_refresh_token");
    if (!refreshToken) {
      throw new AuthError("No refresh token available");
    }

    const response = await this.client.post("/api/auth/refresh", {
      refreshToken,
    });

    const { accessToken, refreshToken: newRefreshToken } = response.data;
    localStorage.setItem("knowledgehub_token", accessToken);
    if (newRefreshToken) {
      localStorage.setItem("knowledgehub_refresh_token", newRefreshToken);
    }
  }

  private clearAuth(): void {
    localStorage.removeItem("knowledgehub_token");
    localStorage.removeItem("knowledgehub_refresh_token");
  }

  // Error handling
  private handleError(error: any): Error {
    if (axios.isAxiosError(error)) {
      if (error.code === "ECONNABORTED") {
        return new TimeoutError("Request timeout");
      }

      if (!error.response) {
        return new NetworkError("Network error", { originalError: error.message });
      }

      const apiError: ApiError = {
        message: error.response.data?.message || error.message,
        code: error.response.status,
        details: error.response.data,
        timestamp: new Date().toISOString(),
      };

      return ErrorHandler.handle(apiError);
    }

    return ErrorHandler.handle(error);
  }

  // Cache management
  private getCacheKey(method: string, url: string, params?: any): string {
    const paramString = params ? JSON.stringify(params) : "";
    return `${method.toUpperCase()}:${url}:${paramString}`;
  }

  private getFromCache<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    const now = Date.now();
    if (now > entry.timestamp + entry.ttl) {
      this.cache.delete(key);
      return null;
    }

    return entry.data;
  }

  private setCache<T>(key: string, data: T, ttl: number): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
    });
  }

  private clearCache(pattern?: string): void {
    if (!pattern) {
      this.cache.clear();
      return;
    }

    const regex = new RegExp(pattern);
    for (const key of this.cache.keys()) {
      if (regex.test(key)) {
        this.cache.delete(key);
      }
    }
  }

  // HTTP methods with caching and retry logic
  async get<T>(
    url: string,
    config: RequestConfig & AxiosRequestConfig = {}
  ): Promise<ApiResponse<T>> {
    const cacheKey = this.getCacheKey("GET", url, config.params);

    // Check cache first
    if (environment.CACHE_ENABLED && (config.cache ?? true)) {
      const cached = this.getFromCache<ApiResponse<T>>(cacheKey);
      if (cached) {
        return cached;
      }
    }

    const operation = async () => {
      const response = await this.client.get<ApiResponse<T>>(url, {
        ...config,
        timeout: config.timeout || environment.TIMEOUT,
      });

      const apiResponse: ApiResponse<T> = {
        data: response.data.data || response.data,
        message: response.data.message,
        status: "success",
        timestamp: new Date().toISOString(),
      };

      // Cache successful responses
      if (environment.CACHE_ENABLED && (config.cache ?? true)) {
        const ttl = config.cacheTTL || environment.CACHE_TTL;
        this.setCache(cacheKey, apiResponse, ttl);
      }

      return apiResponse;
    };

    return withRetry(operation, config.retries || environment.RETRY_ATTEMPTS, config.retryDelay || environment.RETRY_DELAY);
  }

  async post<T>(
    url: string,
    data?: any,
    config: RequestConfig & AxiosRequestConfig = {}
  ): Promise<ApiResponse<T>> {
    const operation = async () => {
      const response = await this.client.post<ApiResponse<T>>(url, data, {
        ...config,
        timeout: config.timeout || environment.TIMEOUT,
      });

      const apiResponse: ApiResponse<T> = {
        data: response.data.data || response.data,
        message: response.data.message,
        status: "success",
        timestamp: new Date().toISOString(),
      };

      // Clear related cache entries
      this.clearCache(url);

      return apiResponse;
    };

    return withRetry(operation, config.retries || environment.RETRY_ATTEMPTS, config.retryDelay || environment.RETRY_DELAY);
  }

  async put<T>(
    url: string,
    data?: any,
    config: RequestConfig & AxiosRequestConfig = {}
  ): Promise<ApiResponse<T>> {
    const operation = async () => {
      const response = await this.client.put<ApiResponse<T>>(url, data, {
        ...config,
        timeout: config.timeout || environment.TIMEOUT,
      });

      const apiResponse: ApiResponse<T> = {
        data: response.data.data || response.data,
        message: response.data.message,
        status: "success",
        timestamp: new Date().toISOString(),
      };

      // Clear related cache entries
      this.clearCache(url);

      return apiResponse;
    };

    return withRetry(operation, config.retries || environment.RETRY_ATTEMPTS, config.retryDelay || environment.RETRY_DELAY);
  }

  async patch<T>(
    url: string,
    data?: any,
    config: RequestConfig & AxiosRequestConfig = {}
  ): Promise<ApiResponse<T>> {
    const operation = async () => {
      const response = await this.client.patch<ApiResponse<T>>(url, data, {
        ...config,
        timeout: config.timeout || environment.TIMEOUT,
      });

      const apiResponse: ApiResponse<T> = {
        data: response.data.data || response.data,
        message: response.data.message,
        status: "success",
        timestamp: new Date().toISOString(),
      };

      // Clear related cache entries
      this.clearCache(url);

      return apiResponse;
    };

    return withRetry(operation, config.retries || environment.RETRY_ATTEMPTS, config.retryDelay || environment.RETRY_DELAY);
  }

  async delete<T>(
    url: string,
    config: RequestConfig & AxiosRequestConfig = {}
  ): Promise<ApiResponse<T>> {
    const operation = async () => {
      const response = await this.client.delete<ApiResponse<T>>(url, {
        ...config,
        timeout: config.timeout || environment.TIMEOUT,
      });

      const apiResponse: ApiResponse<T> = {
        data: response.data.data || response.data,
        message: response.data.message,
        status: "success",
        timestamp: new Date().toISOString(),
      };

      // Clear related cache entries
      this.clearCache(url);

      return apiResponse;
    };

    return withRetry(operation, config.retries || environment.RETRY_ATTEMPTS, config.retryDelay || environment.RETRY_DELAY);
  }

  // Utility methods
  setAuthToken(token: string): void {
    localStorage.setItem("knowledgehub_token", token);
  }

  removeAuthToken(): void {
    this.clearAuth();
  }

  addRequestInterceptor(interceptor: RequestInterceptor): void {
    this.requestInterceptors.push(interceptor);
  }

  addResponseInterceptor<T>(interceptor: ResponseInterceptor<T>): void {
    this.responseInterceptors.push(interceptor as ResponseInterceptor);
  }

  clearCacheByPattern(pattern: string): void {
    this.clearCache(pattern);
  }

  clearAllCache(): void {
    this.cache.clear();
  }

  getCacheStats(): { size: number; keys: string[] } {
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys()),
    };
  }
}

// Export singleton instance
export const apiClient = new ApiClient();

// Export for testing and advanced usage
