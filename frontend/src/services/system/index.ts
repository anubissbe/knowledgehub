// SystemService for health checks, monitoring, and system management

import { ApiResponse, SystemHealth, PerformanceMetrics } from "../types";
import { apiClient } from "../api/client";
import { ErrorHandler } from "../errors";
import { wsManager } from "../websocket/manager";

export interface ServiceStatus {
  name: string;
  status: "up" | "down" | "degraded";
  responseTime: number;
  lastCheck: string;
  version?: string;
  uptime?: number;
  errorRate?: number;
}

export interface SystemOverview {
  status: "healthy" | "degraded" | "down";
  services: ServiceStatus[];
  totalServices: number;
  healthyServices: number;
  degradedServices: number;
  downServices: number;
  lastUpdate: string;
}

export interface ResourceUsage {
  cpu: {
    usage: number;
    cores: number;
    loadAverage: number[];
  };
  memory: {
    used: number;
    total: number;
    percentage: number;
  };
  disk: {
    used: number;
    total: number;
    percentage: number;
  };
  network: {
    bytesIn: number;
    bytesOut: number;
    packetsIn: number;
    packetsOut: number;
  };
  timestamp: string;
}

export interface LogEntry {
  id: string;
  timestamp: string;
  level: "debug" | "info" | "warn" | "error";
  service: string;
  message: string;
  metadata?: Record<string, any>;
}

export interface AlertRule {
  id: string;
  name: string;
  metric: string;
  operator: ">" | "<" | "=" | "!=";
  threshold: number;
  enabled: boolean;
  notifications: string[];
  created_at: string;
}

export class SystemService {
  constructor() {
    this.setupRealtimeListeners();
  }

  // Health Monitoring
  async getSystemHealth(): Promise<SystemHealth> {
    try {
      const response = await apiClient.get<SystemHealth>("/api/system/health", {
        cacheTTL: 10000, // Cache for 10 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get system health");
    }
  }

  async getSystemOverview(): Promise<SystemOverview> {
    try {
      const response = await apiClient.get<SystemOverview>("/api/system/overview", {
        cacheTTL: 15000, // Cache for 15 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get system overview");
    }
  }

  async checkService(serviceName: string): Promise<ServiceStatus> {
    try {
      const response = await apiClient.get<ServiceStatus>(`/api/system/services/${serviceName}/health`, {
        cacheTTL: 5000, // Cache for 5 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, `Check ${serviceName} service`);
    }
  }

  async getAllServices(): Promise<ServiceStatus[]> {
    try {
      const response = await apiClient.get<ServiceStatus[]>("/api/system/services", {
        cacheTTL: 15000, // Cache for 15 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get all services");
    }
  }

  // Performance Monitoring
  async getResourceUsage(): Promise<ResourceUsage> {
    try {
      const response = await apiClient.get<ResourceUsage>("/api/system/resources", {
        cacheTTL: 5000, // Cache for 5 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get resource usage");
    }
  }

  async getPerformanceHistory(params: {
    metric?: string;
    duration?: string;
    granularity?: string;
  } = {}): Promise<PerformanceMetrics[]> {
    try {
      const response = await apiClient.get<PerformanceMetrics[]>("/api/system/performance/history", {
        params: {
          metric: params.metric || "all",
          duration: params.duration || "1h",
          granularity: params.granularity || "1m",
        },
        cacheTTL: 30000, // Cache for 30 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get performance history");
    }
  }

  // Logging
  async getLogs(params: {
    service?: string;
    level?: string;
    limit?: number;
    offset?: number;
    dateFrom?: string;
    dateTo?: string;
  } = {}): Promise<{ logs: LogEntry[]; total: number }> {
    try {
      const response = await apiClient.get<{ logs: LogEntry[]; total: number }>("/api/system/logs", {
        params: {
          service: params.service,
          level: params.level,
          limit: params.limit || 100,
          offset: params.offset || 0,
          date_from: params.dateFrom,
          date_to: params.dateTo,
        },
        cacheTTL: 10000, // Cache for 10 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get logs");
    }
  }

  async searchLogs(query: string, params: {
    service?: string;
    level?: string;
    limit?: number;
    dateFrom?: string;
    dateTo?: string;
  } = {}): Promise<{ logs: LogEntry[]; total: number }> {
    try {
      const response = await apiClient.post<{ logs: LogEntry[]; total: number }>(
        "/api/system/logs/search",
        {
          query,
          service: params.service,
          level: params.level,
          limit: params.limit || 100,
          date_from: params.dateFrom,
          date_to: params.dateTo,
        },
        {
          cacheTTL: 10000, // Cache for 10 seconds
          cacheKey: `${query}-${JSON.stringify(params)}`,
        }
      );
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Search logs");
    }
  }

  // Configuration
  async getSystemConfig(): Promise<Record<string, any>> {
    try {
      const response = await apiClient.get<Record<string, any>>("/api/system/config", {
        cacheTTL: 300000, // Cache for 5 minutes
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get system config");
    }
  }

  async updateSystemConfig(config: Record<string, any>): Promise<Record<string, any>> {
    try {
      const response = await apiClient.put<Record<string, any>>("/api/system/config", config, {
        cache: false,
      });
      
      // Clear config cache
      apiClient.clearCacheByPattern("GET:/api/system/config");
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Update system config");
    }
  }

  // Alerts and Notifications
  async getAlertRules(): Promise<AlertRule[]> {
    try {
      const response = await apiClient.get<AlertRule[]>("/api/system/alerts/rules", {
        cacheTTL: 60000, // Cache for 1 minute
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get alert rules");
    }
  }

  async createAlertRule(rule: Omit<AlertRule, "id" | "created_at">): Promise<AlertRule> {
    try {
      const response = await apiClient.post<AlertRule>("/api/system/alerts/rules", rule, {
        cache: false,
      });
      
      // Clear alert rules cache
      apiClient.clearCacheByPattern("GET:/api/system/alerts/rules");
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Create alert rule");
    }
  }

  async updateAlertRule(id: string, updates: Partial<AlertRule>): Promise<AlertRule> {
    try {
      const response = await apiClient.patch<AlertRule>(`/api/system/alerts/rules/${id}`, updates, {
        cache: false,
      });
      
      // Clear alert rules cache
      apiClient.clearCacheByPattern("GET:/api/system/alerts/rules");
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Update alert rule");
    }
  }

  async deleteAlertRule(id: string): Promise<void> {
    try {
      await apiClient.delete(`/api/system/alerts/rules/${id}`, {
        cache: false,
      });
      
      // Clear alert rules cache
      apiClient.clearCacheByPattern("GET:/api/system/alerts/rules");
    } catch (error) {
      throw ErrorHandler.handle(error, "Delete alert rule");
    }
  }

  async getActiveAlerts(): Promise<any[]> {
    try {
      const response = await apiClient.get<any[]>("/api/system/alerts/active", {
        cacheTTL: 15000, // Cache for 15 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get active alerts");
    }
  }

  // System Operations
  async restartService(serviceName: string): Promise<void> {
    try {
      await apiClient.post(`/api/system/services/${serviceName}/restart`, {}, {
        cache: false,
      });
      
      // Clear service status cache
      apiClient.clearCacheByPattern(`GET:/api/system/services/${serviceName}`);
      apiClient.clearCacheByPattern("GET:/api/system/services");
    } catch (error) {
      throw ErrorHandler.handle(error, `Restart ${serviceName} service`);
    }
  }

  async stopService(serviceName: string): Promise<void> {
    try {
      await apiClient.post(`/api/system/services/${serviceName}/stop`, {}, {
        cache: false,
      });
      
      // Clear service status cache
      apiClient.clearCacheByPattern(`GET:/api/system/services/${serviceName}`);
      apiClient.clearCacheByPattern("GET:/api/system/services");
    } catch (error) {
      throw ErrorHandler.handle(error, `Stop ${serviceName} service`);
    }
  }

  async startService(serviceName: string): Promise<void> {
    try {
      await apiClient.post(`/api/system/services/${serviceName}/start`, {}, {
        cache: false,
      });
      
      // Clear service status cache
      apiClient.clearCacheByPattern(`GET:/api/system/services/${serviceName}`);
      apiClient.clearCacheByPattern("GET:/api/system/services");
    } catch (error) {
      throw ErrorHandler.handle(error, `Start ${serviceName} service`);
    }
  }

  // Real-time monitoring
  private setupRealtimeListeners(): void {
    // Listen for service status changes
    wsManager.on("service_status_change", (data: { service: string; status: ServiceStatus }) => {
      // Clear service cache
      apiClient.clearCacheByPattern(`GET:/api/system/services/${data.service}`);
      apiClient.clearCacheByPattern("GET:/api/system/services");
      apiClient.clearCacheByPattern("GET:/api/system/health");
    });

    // Listen for system alerts
    wsManager.on("system_alert", (data: any) => {
      // Clear alerts cache
      apiClient.clearCacheByPattern("GET:/api/system/alerts");
    });

    // Listen for resource usage updates
    wsManager.on("resource_update", (data: ResourceUsage) => {
      // Clear resource cache
      apiClient.clearCacheByPattern("GET:/api/system/resources");
    });
  }

  // WebSocket connection management
  getWebSocketStatus() {
    return wsManager.getHealthStatus();
  }

  async connectWebSocket(): Promise<void> {
    return wsManager.connect();
  }

  disconnectWebSocket(): void {
    wsManager.disconnect();
  }

  // Cache management
  clearCache(): void {
    apiClient.clearCacheByPattern("GET:/api/system");
  }

  // System information
  async getSystemInfo(): Promise<{
    version: string;
    buildDate: string;
    environment: string;
    features: string[];
  }> {
    try {
      const response = await apiClient.get("/api/system/info", {
        cacheTTL: 3600000, // Cache for 1 hour
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get system info");
    }
  }
}

// Export singleton instance
export const systemService = new SystemService();
