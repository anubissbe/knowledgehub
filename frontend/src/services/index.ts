// Unified Services - Main export file for all services

// Core infrastructure
export { apiClient, ApiClient } from "./api/client";
export { wsManager, WebSocketManager } from "./websocket/manager";
export { environment, isProduction, isDev, isLAN } from "./config/environment";

// Error handling
export {
  ErrorHandler,
  ApiClientError,
  NetworkError,
  TimeoutError,
  AuthError,
  ValidationError,
  ServerError,
  parseErrorResponse,
  withRetry,
} from "./errors";

// Types
export * from "./types";

// Services
export { authService, AuthService } from "./auth";
export { memoryService, MemoryService } from "./memory";
export { aiService, AIService } from "./ai";
export { systemService, SystemService } from "./system";

// Enhanced RAG Services
export { hybridRAGService, HybridRAGService } from "./hybridRAGService";
export { agentWorkflowService, AgentWorkflowService } from "./agentWorkflowService";
export { webIngestionService, WebIngestionService } from "./webIngestionService";

// Service-specific types
export type {
  CreateMemoryRequest,
  UpdateMemoryRequest,
  MemoryStats,
} from "./memory";

export type {
  AIFeatureSummary,
  SessionContinuity,
  MistakeLearning,
  ProactiveAssistance,
  DecisionReasoning,
  CodeEvolution,
  PatternRecognition,
} from "./ai";

export type {
  ServiceStatus,
  SystemOverview,
  ResourceUsage,
  LogEntry,
  AlertRule,
} from "./system";

// Enhanced RAG types
export type {
  RetrievalMode,
  HybridRAGRequest,
  HybridRAGResponse,
  RAGAnalytics,
  ComparisonResult,
} from "./hybridRAGService";

export type {
  WorkflowType,
  AgentRole,
  WorkflowStatus,
  WorkflowExecution,
  WorkflowAnalytics,
  AgentCoordinationMetrics,
} from "./agentWorkflowService";

export type {
  IngestionStatus,
  IngestionMode,
  IngestionJob,
  IngestionResult,
  IngestionAnalytics,
  WebSource,
} from "./webIngestionService";

// Initialize services on import
// This ensures all services are ready when the module is loaded
if (typeof window !== "undefined") {
  // Only initialize in browser environment

  // Auto-connect WebSocket in non-production environments  
  if (import.meta.env.DEV || window.location.hostname.startsWith("192.168.1.")) {
    // Import and connect WebSocket dynamically to avoid circular imports
    import("./websocket/manager").then(({ wsManager }) => {
      wsManager.connect().catch((error) => {
        console.warn("WebSocket connection failed:", error);
      });
    });
  }
}
