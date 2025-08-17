import { StateCreator } from "zustand";
import { RootState, MemoryState } from "../types";
import { memoryService } from "../../services";
import { Memory, MemorySearchQuery, CreateMemoryRequest, UpdateMemoryRequest } from "../../services/types";

export interface MemorySlice extends MemoryState {
  // Actions
  fetchMemories: (params?: { page?: number; limit?: number; type?: string; tags?: string[] }) => Promise<void>;
  searchMemories: (query: MemorySearchQuery) => Promise<void>;
  createMemory: (memory: CreateMemoryRequest) => Promise<void>;
  updateMemory: (id: string, memory: UpdateMemoryRequest) => Promise<void>;
  deleteMemory: (memoryId: string) => Promise<void>;
  setSelectedMemory: (memory: Memory | null) => void;
  setSearchQuery: (query: string) => void;
  addRecentQuery: (query: string) => void;
  fetchMemoryStats: () => Promise<void>;
  fetchRecentMemories: (limit?: number) => Promise<void>;
  bulkDeleteMemories: (ids: string[]) => Promise<void>;
  clearSearch: () => void;
  clearError: () => void;
}

const initialState: MemoryState = {
  memories: [],
  searchResults: null,
  recentMemories: [],
  memoryStats: null,
  memoryTypes: [],
  memoryTags: [],
  isLoading: false,
  isSearching: false,
  error: null,
  selectedMemory: null,
  currentSearchQuery: "",
  recentQueries: JSON.parse(localStorage.getItem("knowledgehub_recent_queries") || "[]"),
  pagination: {
    page: 1,
    limit: 20,
    total: 0,
    totalPages: 0,
    hasNext: false,
    hasPrev: false,
  },
};

export const createMemorySlice: StateCreator<
  RootState,
  [],
  [],
  MemorySlice
> = (set, get) => ({
  ...initialState,

  fetchMemories: async (params = {}) => {
    set((state) => ({
      memory: { ...state.memory, isLoading: true, error: null }
    }));

    try {
      const response = await memoryService.getMemories(params);
      
      set((state) => ({
        memory: {
          ...state.memory,
          memories: response.data,
          pagination: response.pagination,
          isLoading: false,
        }
      }));
    } catch (error: any) {
      set((state) => ({
        memory: {
          ...state.memory,
          isLoading: false,
          error: error.message || "Failed to fetch memories",
        }
      }));
    }
  },

  searchMemories: async (query: MemorySearchQuery) => {
    set((state) => ({
      memory: { 
        ...state.memory, 
        isSearching: true, 
        error: null,
        currentSearchQuery: query.query || "",
      }
    }));

    try {
      const response = await memoryService.searchMemories(query);
      
      // Add to recent queries if it is a text search
      if (query.query) {
        get().memory.addRecentQuery(query.query);
      }
      
      set((state) => ({
        memory: {
          ...state.memory,
          searchResults: response,
          isSearching: false,
        }
      }));
    } catch (error: any) {
      set((state) => ({
        memory: {
          ...state.memory,
          isSearching: false,
          error: error.message || "Search failed",
        }
      }));
    }
  },

  createMemory: async (memory: CreateMemoryRequest) => {
    set((state) => ({
      memory: { ...state.memory, isLoading: true, error: null }
    }));

    try {
      const newMemory = await memoryService.createMemory(memory);
      
      set((state) => ({
        memory: {
          ...state.memory,
          memories: [newMemory, ...state.memory.memories],
          isLoading: false,
        }
      }));

      // Refresh stats
      get().memory.fetchMemoryStats();
    } catch (error: any) {
      set((state) => ({
        memory: {
          ...state.memory,
          isLoading: false,
          error: error.message || "Failed to create memory",
        }
      }));
      throw error;
    }
  },

  // ... rest of methods ...
  
  setSelectedMemory: (memory: Memory | null) => {
    set((state) => ({
      memory: { ...state.memory, selectedMemory: memory }
    }));
  },

  setSearchQuery: (query: string) => {
    set((state) => ({
      memory: { ...state.memory, currentSearchQuery: query }
    }));
  },

  addRecentQuery: (query: string) => {
    if (!query.trim()) return;
    
    set((state) => {
      const recentQueries = [query, ...state.memory.recentQueries.filter(q => q !== query)]
        .slice(0, 10); // Keep only last 10 queries
      
      localStorage.setItem("knowledgehub_recent_queries", JSON.stringify(recentQueries));
      
      return {
        memory: { ...state.memory, recentQueries }
      };
    });
  },

  fetchMemoryStats: async () => {
    try {
      const stats = await memoryService.getMemoryStats();
      set((state) => ({
        memory: { ...state.memory, memoryStats: stats }
      }));
    } catch (error: any) {
    }
  },

  fetchRecentMemories: async (limit = 10) => {
    try {
      const recentMemories = await memoryService.getRecentMemories(limit);
      set((state) => ({
        memory: { ...state.memory, recentMemories }
      }));
    } catch (error: any) {
    }
  },

  updateMemory: async (id: string, updates: UpdateMemoryRequest) => {
    try {
      const updatedMemory = await memoryService.updateMemory(id, updates);
      set((state) => ({
        memory: {
          ...state.memory,
          memories: state.memory.memories.map(m => 
            m.id === id ? updatedMemory : m
          ),
          selectedMemory: state.memory.selectedMemory?.id === id 
            ? updatedMemory 
            : state.memory.selectedMemory,
        }
      }));
    } catch (error: any) {
      set((state) => ({
        memory: {
          ...state.memory,
          error: error.message || "Failed to update memory",
        }
      }));
      throw error;
    }
  },

  deleteMemory: async (memoryId: string) => {
    try {
      await memoryService.deleteMemory(memoryId);
      set((state) => ({
        memory: {
          ...state.memory,
          memories: state.memory.memories.filter(m => m.id !== memoryId),
          selectedMemory: state.memory.selectedMemory?.id === memoryId 
            ? null 
            : state.memory.selectedMemory,
        }
      }));
      get().memory.fetchMemoryStats();
    } catch (error: any) {
      set((state) => ({
        memory: {
          ...state.memory,
          error: error.message || "Failed to delete memory",
        }
      }));
      throw error;
    }
  },

  bulkDeleteMemories: async (ids: string[]) => {
    try {
      await memoryService.bulkDeleteMemories(ids);
      set((state) => ({
        memory: {
          ...state.memory,
          memories: state.memory.memories.filter(m => !ids.includes(m.id)),
          selectedMemory: state.memory.selectedMemory && ids.includes(state.memory.selectedMemory.id)
            ? null 
            : state.memory.selectedMemory,
        }
      }));
      get().memory.fetchMemoryStats();
    } catch (error: any) {
      set((state) => ({
        memory: {
          ...state.memory,
          error: error.message || "Failed to delete memories",
        }
      }));
      throw error;
    }
  },

  clearSearch: () => {
    set((state) => ({
      memory: {
        ...state.memory,
        searchResults: null,
        currentSearchQuery: "",
        isSearching: false,
      }
    }));
  },

  clearError: () => {
    set((state) => ({
      memory: { ...state.memory, error: null }
    }));
  },
});
