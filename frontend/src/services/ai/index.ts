// AIService for AI intelligence features, real-time metrics, and insights

import { ApiResponse, AIFeatureStatus, AIInsight, PerformanceMetrics } from "../types";
import { apiClient } from "../api/client";
import { ErrorHandler } from "../errors";
import { wsManager } from "../websocket/manager";

export interface AIFeatureSummary {
  total_features: number;
  active_features: number;
  features: Record<string, AIFeatureStatus>;
  performance: {
    accuracy: number;
    speed: number;
    reliability: number;
  };
}

export interface SessionContinuity {
  session_id: string;
  start_time: string;
  status: string;
  memories_count: number;
  context_size: number;
}

export interface MistakeLearning {
  total_mistakes: number;
  resolved_mistakes: number;
  pending_mistakes: number;
  recent_lessons: Array<{
    id: string;
    error_type: string;
    solution: string;
    confidence: number;
    created_at: string;
  }>;
}

export interface ProactiveAssistance {
  suggestions_made: number;
  accepted_suggestions: number;
  pending_suggestions: Array<{
    id: string;
    type: string;
    content: string;
    confidence: number;
    created_at: string;
  }>;
}

export interface DecisionReasoning {
  total_decisions: number;
  categories: Record<string, number>;
  recent_decisions: Array<{
    id: string;
    title: string;
    reasoning: string;
    alternatives: string[];
    confidence: number;
    created_at: string;
  }>;
}

export interface CodeEvolution {
  total_changes: number;
  languages: Record<string, number>;
  recent_changes: Array<{
    id: string;
    file_path: string;
    change_type: string;
    description: string;
    created_at: string;
  }>;
}

export interface PatternRecognition {
  total_patterns: number;
  pattern_types: Record<string, number>;
  recent_patterns: Array<{
    id: string;
    pattern_type: string;
    description: string;
    confidence: number;
    created_at: string;
  }>;
}

export class AIService {
  constructor() {
    this.setupRealtimeListeners();
  }

  // AI Features Overview
  async getFeaturesSummary(): Promise<AIFeatureSummary> {
    try {
      const response = await apiClient.get<AIFeatureSummary>("/api/ai-features/summary", {
        cacheTTL: 30000, // Cache for 30 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get AI features summary");
    }
  }

  // Session Continuity
  async getSessionContinuity(): Promise<SessionContinuity> {
    try {
      const response = await apiClient.get<SessionContinuity>("/api/claude-auto/session/current", {
        cacheTTL: 15000, // Cache for 15 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get session continuity");
    }
  }

  async createSession(sessionData: { name?: string; description?: string }): Promise<SessionContinuity> {
    try {
      const response = await apiClient.post<SessionContinuity>(
        "/api/claude-auto/session/create",
        sessionData,
        { cache: false }
      );
      
      // Clear session cache
      apiClient.clearCacheByPattern("GET:/api/claude-auto/session");
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Create session");
    }
  }

  // Mistake Learning
  async getMistakeLearning(): Promise<MistakeLearning> {
    try {
      const response = await apiClient.get<MistakeLearning>("/api/mistake-learning/summary", {
        cacheTTL: 30000, // Cache for 30 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get mistake learning");
    }
  }

  async getLessons(params: { type?: string; limit?: number } = {}): Promise<any[]> {
    try {
      const response = await apiClient.get("/api/mistake-learning/lessons", {
        params: {
          type: params.type,
          limit: params.limit || 10,
        },
        cacheTTL: 60000, // Cache for 1 minute
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get lessons");
    }
  }

  async reportMistake(mistake: {
    type: string;
    description: string;
    solution?: string;
    context?: Record<string, any>;
  }): Promise<void> {
    try {
      await apiClient.post("/api/mistake-learning/report", mistake, {
        cache: false,
      });
      
      // Clear related cache
      apiClient.clearCacheByPattern("GET:/api/mistake-learning");
    } catch (error) {
      throw ErrorHandler.handle(error, "Report mistake");
    }
  }

  // Proactive Assistance
  async getProactiveAssistance(): Promise<ProactiveAssistance> {
    try {
      const response = await apiClient.get<ProactiveAssistance>("/api/proactive/summary", {
        cacheTTL: 30000, // Cache for 30 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get proactive assistance");
    }
  }

  async getSuggestions(params: { type?: string; status?: string; limit?: number } = {}): Promise<any[]> {
    try {
      const response = await apiClient.get("/api/proactive/suggestions", {
        params: {
          type: params.type,
          status: params.status,
          limit: params.limit || 10,
        },
        cacheTTL: 30000, // Cache for 30 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get suggestions");
    }
  }

  async acceptSuggestion(suggestionId: string): Promise<void> {
    try {
      await apiClient.post(`/api/proactive/suggestions/${suggestionId}/accept`, {}, {
        cache: false,
      });
      
      // Clear suggestions cache
      apiClient.clearCacheByPattern("GET:/api/proactive");
    } catch (error) {
      throw ErrorHandler.handle(error, "Accept suggestion");
    }
  }

  async declineSuggestion(suggestionId: string, reason?: string): Promise<void> {
    try {
      await apiClient.post(
        `/api/proactive/suggestions/${suggestionId}/decline`,
        { reason },
        { cache: false }
      );
      
      // Clear suggestions cache
      apiClient.clearCacheByPattern("GET:/api/proactive");
    } catch (error) {
      throw ErrorHandler.handle(error, "Decline suggestion");
    }
  }

  // Decision Reasoning
  async getDecisionReasoning(): Promise<DecisionReasoning> {
    try {
      const response = await apiClient.get<DecisionReasoning>("/api/decisions/summary", {
        cacheTTL: 30000, // Cache for 30 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get decision reasoning");
    }
  }

  async getDecisions(params: {
    category?: string;
    dateFrom?: string;
    dateTo?: string;
    limit?: number;
  } = {}): Promise<any[]> {
    try {
      const response = await apiClient.post("/api/decisions/search", params, {
        cacheTTL: 60000, // Cache for 1 minute
        cacheKey: JSON.stringify(params),
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get decisions");
    }
  }

  async getDecisionCategories(): Promise<string[]> {
    try {
      const response = await apiClient.get<string[]>("/api/decisions/categories", {
        cacheTTL: 300000, // Cache for 5 minutes
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get decision categories");
    }
  }

  // Code Evolution
  async getCodeEvolution(): Promise<CodeEvolution> {
    try {
      const response = await apiClient.get<CodeEvolution>("/api/code-evolution/summary", {
        cacheTTL: 30000, // Cache for 30 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get code evolution");
    }
  }

  async getCodeChanges(params: {
    language?: string;
    dateFrom?: string;
    dateTo?: string;
    limit?: number;
  } = {}): Promise<any[]> {
    try {
      const response = await apiClient.get("/api/code-evolution/history", {
        params: {
          language: params.language,
          date_from: params.dateFrom,
          date_to: params.dateTo,
          limit: params.limit || 20,
        },
        cacheTTL: 60000, // Cache for 1 minute
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get code changes");
    }
  }

  // Pattern Recognition
  async getPatternRecognition(): Promise<PatternRecognition> {
    try {
      const response = await apiClient.get<PatternRecognition>("/api/patterns/summary", {
        cacheTTL: 30000, // Cache for 30 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get pattern recognition");
    }
  }

  async getPatterns(params: {
    type?: string;
    confidence?: number;
    limit?: number;
  } = {}): Promise<any[]> {
    try {
      const response = await apiClient.get("/api/patterns/recent", {
        params: {
          type: params.type,
          min_confidence: params.confidence,
          limit: params.limit || 10,
        },
        cacheTTL: 60000, // Cache for 1 minute
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get patterns");
    }
  }

  async recognizePattern(data: {
    content: string;
    context?: Record<string, any>;
    type?: string;
  }): Promise<any> {
    try {
      const response = await apiClient.post("/api/patterns/recognize", data, {
        cache: false,
      });
      
      // Clear patterns cache
      apiClient.clearCacheByPattern("GET:/api/patterns");
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Recognize pattern");
    }
  }

  // Performance Metrics
  async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    try {
      const response = await apiClient.get<PerformanceMetrics>("/api/performance/stats", {
        cacheTTL: 15000, // Cache for 15 seconds
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get performance metrics");
    }
  }

  async getPerformanceReport(params: {
    dateFrom?: string;
    dateTo?: string;
    granularity?: string;
  } = {}): Promise<any> {
    try {
      const response = await apiClient.get("/api/performance/report", {
        params: {
          date_from: params.dateFrom,
          date_to: params.dateTo,
          granularity: params.granularity || "hourly",
        },
        cacheTTL: 60000, // Cache for 1 minute
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get performance report");
    }
  }

  // Real-time updates
  private setupRealtimeListeners(): void {
    // Listen for AI feature updates
    wsManager.on("ai_feature_update", (data: { feature: string; status: AIFeatureStatus }) => {
      // Clear related cache
      apiClient.clearCacheByPattern("GET:/api/ai-features");
    });

    // Listen for new insights
    wsManager.on("ai_insight", (data: AIInsight) => {
      // Could trigger UI notifications here
    });

    // Listen for performance alerts
    wsManager.on("performance_alert", (data: { metric: string; value: number; threshold: number }) => {
      // Clear performance cache to get fresh data
      apiClient.clearCacheByPattern("GET:/api/performance");
    });
  }

  // Utility methods
  async generateInsight(params: {
    type: string;
    context: Record<string, any>;
    options?: Record<string, any>;
  }): Promise<AIInsight> {
    try {
      const response = await apiClient.post<AIInsight>("/api/ai/generate-insight", params, {
        cache: false,
      });
      
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Generate insight");
    }
  }

  // Cache management
  clearCache(): void {
    apiClient.clearCacheByPattern("GET:/api/ai-features");
    apiClient.clearCacheByPattern("GET:/api/claude-auto");
    apiClient.clearCacheByPattern("GET:/api/mistake-learning");
    apiClient.clearCacheByPattern("GET:/api/proactive");
    apiClient.clearCacheByPattern("GET:/api/decisions");
    apiClient.clearCacheByPattern("GET:/api/code-evolution");
    apiClient.clearCacheByPattern("GET:/api/patterns");
    apiClient.clearCacheByPattern("GET:/api/performance");
  }
}

// Export singleton instance
export const aiService = new AIService();
