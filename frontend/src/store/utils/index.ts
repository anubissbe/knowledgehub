// Store utilities and helpers

// Local storage keys
export const STORAGE_KEYS = {
  TOKEN: 'knowledgehub_token',
  SETTINGS: 'knowledgehub_settings',
  THEME: 'knowledgehub_darkMode',
  RECENT_QUERIES: 'knowledgehub_recent_queries',
  USER_PREFERENCES: 'knowledgehub_user_preferences',
} as const;

// Default values
export const DEFAULT_VALUES = {
  USER_PREFERENCES: {
    theme: 'system' as const,
    language: 'en',
    notifications: {
      enabled: true,
      email: true,
      push: true,
      aiInsights: true,
    },
    dashboard: {
      layout: 'grid' as const,
      autoRefresh: true,
      refreshInterval: 30000, // 30 seconds
    },
  },
  NOTIFICATION_DURATION: {
    success: 5000,
    error: 0, // Persist until manually closed
    warning: 7000,
    info: 5000,
  },
} as const;

// Type guards
export const isValidTheme = (value: any): value is 'light' | 'dark' | 'system' => {
  return ['light', 'dark', 'system'].includes(value);
};

export const isValidNotificationType = (value: any): value is 'success' | 'error' | 'warning' | 'info' => {
  return ['success', 'error', 'warning', 'info'].includes(value);
};

// Storage utilities
export const storageUtils = {
  // Get data from localStorage with fallback
  get: <T>(key: string, defaultValue: T): T => {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      return defaultValue;
    }
  },

  // Set data in localStorage
  set: (key: string, value: any): void => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
    }
  },

  // Remove data from localStorage
  remove: (key: string): void => {
    try {
      localStorage.removeItem(key);
    } catch (error) {
    }
  },

  // Clear all KnowledgeHub data from localStorage
  clearAll: (): void => {
    Object.values(STORAGE_KEYS).forEach(key => {
      try {
        localStorage.removeItem(key);
      } catch (error) {
      }
    });
  },
};

// Performance utilities
export const performanceUtils = {
  // Measure component render time
  measureRender: (componentName: string, renderFn: () => void) => {
    if (!import.meta.env.DEV) {
      renderFn();
      return 0;
    }

    const start = performance.now();
    renderFn();
    const end = performance.now();
    const duration = end - start;

    if (duration > 16) { // Warn if render takes longer than 1 frame (16ms)
    }

    return duration;
  },

  // Debounce function
  debounce: <T extends (...args: any[]) => any>(
    func: T,
    wait: number,
    immediate?: boolean
  ): ((...args: Parameters<T>) => void) => {
    let timeout: NodeJS.Timeout | null = null;
    
    return (...args: Parameters<T>) => {
      const later = () => {
        timeout = null;
        if (!immediate) func(...args);
      };
      
      const callNow = immediate && !timeout;
      
      if (timeout) clearTimeout(timeout);
      timeout = setTimeout(later, wait);
      
      if (callNow) func(...args);
    };
  },

  // Throttle function
  throttle: <T extends (...args: any[]) => any>(
    func: T,
    limit: number
  ): ((...args: Parameters<T>) => void) => {
    let inThrottle: boolean;
    
    return (...args: Parameters<T>) => {
      if (!inThrottle) {
        func(...args);
        inThrottle = true;
        setTimeout(() => (inThrottle = false), limit);
      }
    };
  },
};

// API utilities
export const apiUtils = {
  // Build query string from object
  buildQueryString: (params: Record<string, any>): string => {
    const searchParams = new URLSearchParams();
    
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        if (Array.isArray(value)) {
          value.forEach(item => searchParams.append(key, item.toString()));
        } else {
          searchParams.append(key, value.toString());
        }
      }
    });
    
    return searchParams.toString();
  },

  // Handle API errors
  handleError: (error: any): string => {
    if (error.response) {
      // Server responded with error status
      return error.response.data?.message || `HTTP ${error.response.status}: ${error.response.statusText}`;
    } else if (error.request) {
      // Request was made but no response received
      return 'Network error: Unable to connect to server';
    } else {
      // Something else happened
      return error.message || 'An unexpected error occurred';
    }
  },
};

// Memory utilities
export const memoryUtils = {
  // Filter memories by criteria
  filterMemories: (memories: any[], filters: any): any[] => {
    return memories.filter(memory => {
      // Type filter
      if (filters.type?.length && !filters.type.includes(memory.type)) {
        return false;
      }

      // Tags filter
      if (filters.tags?.length) {
        const hasTag = filters.tags.some((tag: string) =>
          memory.tags.includes(tag)
        );
        if (!hasTag) return false;
      }

      // Date range filter
      if (filters.dateRange) {
        const memoryDate = new Date(memory.createdAt);
        const start = new Date(filters.dateRange.start);
        const end = new Date(filters.dateRange.end);
        
        if (memoryDate < start || memoryDate > end) {
          return false;
        }
      }

      // Importance filter
      if (filters.importance) {
        const importance = memory.metadata.importance;
        if (importance < filters.importance.min || importance > filters.importance.max) {
          return false;
        }
      }

      return true;
    });
  },

  // Sort memories by criteria
  sortMemories: (memories: any[], sortBy: 'date' | 'importance' | 'relevance', order: 'asc' | 'desc' = 'desc'): any[] => {
    return [...memories].sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case 'date':
          comparison = new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
          break;
        case 'importance':
          comparison = a.metadata.importance - b.metadata.importance;
          break;
        case 'relevance':
          comparison = (b.similarity || 0) - (a.similarity || 0);
          break;
        default:
          return 0;
      }

      return order === 'asc' ? comparison : -comparison;
    });
  },
};

// Validation utilities
export const validationUtils = {
  // Validate email
  isValidEmail: (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  },

  // Validate API key format
  isValidApiKey: (apiKey: string): boolean => {
    // Assuming API keys are alphanumeric and at least 16 characters
    return /^[a-zA-Z0-9]{16,}$/.test(apiKey);
  },

  // Validate memory content
  isValidMemoryContent: (content: string): boolean => {
    return content.trim().length >= 3 && content.length <= 10000;
  },
};

// Date utilities
export const dateUtils = {
  // Format date for display
  formatDate: (date: string | Date, format: 'short' | 'medium' | 'long' = 'medium'): string => {
    const d = typeof date === 'string' ? new Date(date) : date;
    
    const options: Intl.DateTimeFormatOptions = {
      short: { month: 'short', day: 'numeric' },
      medium: { month: 'short', day: 'numeric', year: 'numeric' },
      long: { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' },
    }[format];

    return d.toLocaleDateString('en-US', options);
  },

  // Get relative time string
  getRelativeTime: (date: string | Date): string => {
    const d = typeof date === 'string' ? new Date(date) : date;
    const now = new Date();
    const diffInSeconds = Math.floor((now.getTime() - d.getTime()) / 1000);

    if (diffInSeconds < 60) return 'just now';
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`;
    if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)} days ago`;
    
    return dateUtils.formatDate(d, 'medium');
  },
};
