// Updated store types to integrate with unified services

// Re-export service types for compatibility
export * from "../../services/types";

// Store-specific state interfaces
export interface AuthState {
  user: import("../../services/types").User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  token: string | null;
  apiKey: string | null;
}

export interface MemoryState {
  memories: import("../../services/types").Memory[];
  searchResults: import("../../services/types").PaginatedResponse<import("../../services/types").Memory> | null;
  recentMemories: import("../../services/types").Memory[];
  memoryStats: import("../../services/memory").MemoryStats | null;
  memoryTypes: string[];
  memoryTags: string[];
  isLoading: boolean;
  isSearching: boolean;
  error: string | null;
  selectedMemory: import("../../services/types").Memory | null;
  currentSearchQuery: string;
  recentQueries: string[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

export interface AIState {
  featuresSummary: import("../../services/ai").AIFeatureSummary | null;
  sessionContinuity: import("../../services/ai").SessionContinuity | null;
  mistakeLearning: import("../../services/ai").MistakeLearning | null;
  proactiveAssistance: import("../../services/ai").ProactiveAssistance | null;
  decisionReasoning: import("../../services/ai").DecisionReasoning | null;
  codeEvolution: import("../../services/ai").CodeEvolution | null;
  patternRecognition: import("../../services/ai").PatternRecognition | null;
  performanceMetrics: import("../../services/types").PerformanceMetrics | null;
  isLoading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

export interface AppState {
  isInitialized: boolean;
  isOnline: boolean;
  theme: "light" | "dark" | "auto";
  sidebarOpen: boolean;
  notifications: Notification[];
  loading: {
    global: boolean;
    components: Record<string, boolean>;
  };
}

export interface UIState {
  theme: "light" | "dark" | "auto";
  sidebarCollapsed: boolean;
  selectedProject: string | null;
  breadcrumbs: BreadcrumbItem[];
  notifications: UINotification[];
  modals: {
    [key: string]: {
      open: boolean;
      data?: any;
    };
  };
  layout: {
    headerHeight: number;
    sidebarWidth: number;
    contentPadding: number;
  };
  performance: {
    renderTime: number;
    apiLatency: number;
    lastUpdate: string;
  };
}

// UI-specific types
export interface BreadcrumbItem {
  label: string;
  href?: string;
  isActive?: boolean;
}

export interface UINotification {
  id: string;
  type: "success" | "error" | "warning" | "info";
  title: string;
  message: string;
  timestamp: string;
  actions?: NotificationAction[];
  persistent?: boolean;
  autoHide?: boolean;
  duration?: number;
}

export interface NotificationAction {
  label: string;
  action: () => void;
  style?: "primary" | "secondary";
}

export interface Notification {
  id: string;
  type: "success" | "error" | "warning" | "info";
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  actions?: {
    label: string;
    action: () => void;
  }[];
}

// Root state interface
export interface RootState {
  auth: AuthState;
  memory: MemoryState;
  ai: AIState;
  app: AppState;
  ui: UIState;
}

// Legacy compatibility - re-export commonly used types
export type { User, UserPreferences } from "../../services/types";

// Store slices interfaces (for type inference)
export interface AuthSlice extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  setApiKey: (apiKey: string) => void;
  updatePreferences: (preferences: Partial<import("../../services/types").UserPreferences>) => Promise<void>;
  checkAuth: () => Promise<void>;
  refreshToken: () => Promise<void>;
  updateProfile: (updates: Partial<import("../../services/types").User>) => Promise<void>;
  isLoggedIn: boolean;
  isTokenExpired: boolean;
  isTokenExpiringSoon: boolean;
}

export interface MemorySlice extends MemoryState {
  fetchMemories: (params?: { page?: number; limit?: number; type?: string; tags?: string[] }) => Promise<void>;
  searchMemories: (query: import("../../services/types").MemorySearchQuery) => Promise<void>;
  createMemory: (memory: import("../../services/memory").CreateMemoryRequest) => Promise<void>;
  updateMemory: (id: string, memory: import("../../services/memory").UpdateMemoryRequest) => Promise<void>;
  deleteMemory: (memoryId: string) => Promise<void>;
  setSelectedMemory: (memory: import("../../services/types").Memory | null) => void;
  setSearchQuery: (query: string) => void;
  addRecentQuery: (query: string) => void;
  fetchMemoryStats: () => Promise<void>;
  fetchRecentMemories: (limit?: number) => Promise<void>;
  bulkDeleteMemories: (ids: string[]) => Promise<void>;
  clearSearch: () => void;
  clearError: () => void;
}

export interface AISlice extends AIState {
  fetchFeaturesSummary: () => Promise<void>;
  fetchSessionContinuity: () => Promise<void>;
  fetchMistakeLearning: () => Promise<void>;
  fetchProactiveAssistance: () => Promise<void>;
  fetchDecisionReasoning: () => Promise<void>;
  fetchCodeEvolution: () => Promise<void>;
  fetchPatternRecognition: () => Promise<void>;
  fetchPerformanceMetrics: () => Promise<void>;
  reportMistake: (mistake: any) => Promise<void>;
  acceptSuggestion: (suggestionId: string) => Promise<void>;
  declineSuggestion: (suggestionId: string, reason?: string) => Promise<void>;
  clearError: () => void;
}

export interface AppSlice extends AppState {
  initialize: () => Promise<void>;
  setTheme: (theme: "light" | "dark" | "auto") => void;
  setSidebarOpen: (open: boolean) => void;
  addNotification: (notification: Omit<Notification, "id" | "timestamp" | "read">) => void;
  removeNotification: (id: string) => void;
  markNotificationRead: (id: string) => void;
  setGlobalLoading: (loading: boolean) => void;
  setComponentLoading: (component: string, loading: boolean) => void;
}

export interface UISlice extends UIState {
  setTheme: (theme: "light" | "dark" | "auto") => void;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  setSelectedProject: (projectId: string | null) => void;
  setBreadcrumbs: (breadcrumbs: BreadcrumbItem[]) => void;
  addNotification: (notification: Omit<UINotification, "id" | "timestamp">) => void;
  removeNotification: (id: string) => void;
  openModal: (modalId: string, data?: any) => void;
  closeModal: (modalId: string) => void;
  updatePerformance: (metrics: Partial<UIState["performance"]>) => void;
}
