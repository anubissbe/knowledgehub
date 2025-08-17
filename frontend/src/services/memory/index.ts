// MemoryService with CRUD operations, search functionality, and context management

import { ApiResponse, Memory, MemorySearchQuery, PaginatedResponse } from "../types";
import { apiClient } from "../api/client";
import { ErrorHandler } from "../errors";

export interface CreateMemoryRequest {
  type: string;
  content: string;
  metadata?: Record<string, any>;
  tags?: string[];
}

export interface UpdateMemoryRequest {
  type?: string;
  content?: string;
  metadata?: Record<string, any>;
  tags?: string[];
}

export interface MemoryStats {
  total_memories: number;
  memory_types: Record<string, number>;
  recent_activity: number;
  storage_used: number;
  sync_status: string;
}

export class MemoryService {
  // CRUD Operations
  async createMemory(data: CreateMemoryRequest): Promise<Memory> {
    try {
      const response = await apiClient.post<Memory>("/api/v1/memories", data, {
        cache: false,
      });
      
      // Clear cache for memory lists
      apiClient.clearCacheByPattern("GET:/api/v1/memories");
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Create memory");
    }
  }

  async getMemory(id: string): Promise<Memory> {
    try {
      const response = await apiClient.get<Memory>(`/api/v1/memories/${id}`);
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get memory");
    }
  }

  async updateMemory(id: string, data: UpdateMemoryRequest): Promise<Memory> {
    try {
      const response = await apiClient.patch<Memory>(`/api/v1/memories/${id}`, data, {
        cache: false,
      });
      
      // Clear cache for this memory and lists
      apiClient.clearCacheByPattern(`GET:/api/v1/memories/${id}`);
      apiClient.clearCacheByPattern("GET:/api/v1/memories");
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Update memory");
    }
  }

  async deleteMemory(id: string): Promise<void> {
    try {
      await apiClient.delete(`/api/v1/memories/${id}`, { cache: false });
      
      // Clear cache for this memory and lists
      apiClient.clearCacheByPattern(`GET:/api/v1/memories/${id}`);
      apiClient.clearCacheByPattern("GET:/api/v1/memories");
    } catch (error) {
      throw ErrorHandler.handle(error, "Delete memory");
    }
  }

  // List and Search Operations
  async getMemories(
    params: {
      page?: number;
      limit?: number;
      type?: string;
      tags?: string[];
    } = {}
  ): Promise<PaginatedResponse<Memory>> {
    try {
      const response = await apiClient.get<PaginatedResponse<Memory>>("/api/v1/memories", {
        params: {
          page: params.page || 1,
          limit: params.limit || 20,
          ...(params.type && { type: params.type }),
          ...(params.tags && { tags: params.tags.join(",") }),
        },
        cacheTTL: 60000, // Cache for 1 minute
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get memories");
    }
  }

  async searchMemories(query: MemorySearchQuery): Promise<PaginatedResponse<Memory>> {
    try {
      const response = await apiClient.post<PaginatedResponse<Memory>>(
        "/api/memory/search",
        query,
        {
          cacheTTL: 30000, // Cache search results for 30 seconds
          cacheKey: JSON.stringify(query),
        }
      );
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Search memories");
    }
  }

  async getMemoryTypes(): Promise<string[]> {
    try {
      const response = await apiClient.get<string[]>("/api/v1/memories/types", {
        cacheTTL: 300000, // Cache for 5 minutes
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get memory types");
    }
  }

  async getMemoryTags(): Promise<string[]> {
    try {
      const response = await apiClient.get<string[]>("/api/v1/memories/tags", {
        cacheTTL: 60000, // Cache for 1 minute
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get memory tags");
    }
  }

  // Statistics and Analytics
  async getMemoryStats(): Promise<MemoryStats> {
    try {
      const response = await apiClient.get<MemoryStats>("/api/memory/stats", {
        cacheTTL: 30000, // Cache for 30 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get memory stats");
    }
  }

  async getRecentMemories(limit = 10): Promise<Memory[]> {
    try {
      const response = await apiClient.get<Memory[]>("/api/memory/recent", {
        params: { limit },
        cacheTTL: 15000, // Cache for 15 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get recent memories");
    }
  }

  // Context Operations
  async getContextMemories(
    contextId: string,
    params: { page?: number; limit?: number } = {}
  ): Promise<PaginatedResponse<Memory>> {
    try {
      const response = await apiClient.get<PaginatedResponse<Memory>>(
        `/api/memory/context/${contextId}`,
        {
          params: {
            page: params.page || 1,
            limit: params.limit || 20,
          },
          cacheTTL: 30000, // Cache for 30 seconds
        }
      );
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get context memories");
    }
  }

  async addToContext(memoryId: string, contextId: string): Promise<void> {
    try {
      await apiClient.post(
        `/api/memory/${memoryId}/context`,
        { contextId },
        { cache: false }
      );
      
      // Clear related cache
      apiClient.clearCacheByPattern(`GET:/api/memory/context/${contextId}`);
    } catch (error) {
      throw ErrorHandler.handle(error, "Add to context");
    }
  }

  async removeFromContext(memoryId: string, contextId: string): Promise<void> {
    try {
      await apiClient.delete(`/api/memory/${memoryId}/context/${contextId}`, {
        cache: false,
      });
      
      // Clear related cache
      apiClient.clearCacheByPattern(`GET:/api/memory/context/${contextId}`);
    } catch (error) {
      throw ErrorHandler.handle(error, "Remove from context");
    }
  }

  // Bulk Operations
  async bulkCreateMemories(memories: CreateMemoryRequest[]): Promise<Memory[]> {
    try {
      const response = await apiClient.post<Memory[]>(
        "/api/v1/memories/bulk",
        { memories },
        { cache: false }
      );
      
      // Clear cache for memory lists
      apiClient.clearCacheByPattern("GET:/api/v1/memories");
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Bulk create memories");
    }
  }

  async bulkDeleteMemories(ids: string[]): Promise<void> {
    try {
      await apiClient.post(
        "/api/v1/memories/bulk-delete",
        { ids },
        { cache: false }
      );
      
      // Clear cache for memory lists and individual memories
      apiClient.clearCacheByPattern("GET:/api/v1/memories");
      ids.forEach(id => {
        apiClient.clearCacheByPattern(`GET:/api/v1/memories/${id}`);
      });
    } catch (error) {
      throw ErrorHandler.handle(error, "Bulk delete memories");
    }
  }

  // Export/Import
  async exportMemories(format: "json" | "csv" = "json"): Promise<Blob> {
    try {
      const response = await apiClient.get("/api/v1/memories/export", {
        params: { format },
        responseType: "blob" as any,
        cache: false,
      });
      
      return response.data as any;
    } catch (error) {
      throw ErrorHandler.handle(error, "Export memories");
    }
  }

  async importMemories(file: File): Promise<{ imported: number; errors: string[] }> {
    try {
      const formData = new FormData();
      formData.append("file", file);
      
      const response = await apiClient.post<{ imported: number; errors: string[] }>(
        "/api/v1/memories/import",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          cache: false,
        }
      );
      
      // Clear cache after import
      apiClient.clearCacheByPattern("GET:/api/v1/memories");
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Import memories");
    }
  }

  // Cache management
  clearCache(): void {
    apiClient.clearCacheByPattern("GET:/api/v1/memories");
    apiClient.clearCacheByPattern("GET:/api/memory");
  }
}

// Export singleton instance
export const memoryService = new MemoryService();
