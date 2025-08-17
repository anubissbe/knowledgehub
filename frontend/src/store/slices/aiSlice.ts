import { StateCreator } from "zustand";
import { RootState, AIState } from "../types";
import { aiService } from "../../services";
import { AIFeatureSummary, SessionContinuity, MistakeLearning, ProactiveAssistance, DecisionReasoning, CodeEvolution, PatternRecognition } from "../../services/types";

export interface AISlice extends AIState {
  // Actions
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

const initialState: AIState = {
  featuresSummary: null,
  sessionContinuity: null,
  mistakeLearning: null,
  proactiveAssistance: null,
  decisionReasoning: null,
  codeEvolution: null,
  patternRecognition: null,
  performanceMetrics: null,
  isLoading: false,
  error: null,
  lastUpdated: null,
};

export const createAISlice: StateCreator<
  RootState,
  [],
  [],
  AISlice
> = (set, get) => ({
  ...initialState,

  fetchFeaturesSummary: async () => {
    set((state) => ({
      ai: { ...state.ai, isLoading: true, error: null }
    }));

    try {
      const featuresSummary = await aiService.getFeaturesSummary();
      set((state) => ({
        ai: {
          ...state.ai,
          featuresSummary,
          isLoading: false,
          lastUpdated: new Date(),
        }
      }));
    } catch (error: any) {
      set((state) => ({
        ai: {
          ...state.ai,
          isLoading: false,
          error: error.message || "Failed to fetch AI features summary",
        }
      }));
    }
  },

  fetchSessionContinuity: async () => {
    try {
      const sessionContinuity = await aiService.getSessionContinuity();
      set((state) => ({
        ai: { ...state.ai, sessionContinuity }
      }));
    } catch (error: any) {
    }
  },

  fetchMistakeLearning: async () => {
    try {
      const mistakeLearning = await aiService.getMistakeLearning();
      set((state) => ({
        ai: { ...state.ai, mistakeLearning }
      }));
    } catch (error: any) {
    }
  },

  fetchProactiveAssistance: async () => {
    try {
      const proactiveAssistance = await aiService.getProactiveAssistance();
      set((state) => ({
        ai: { ...state.ai, proactiveAssistance }
      }));
    } catch (error: any) {
    }
  },

  fetchDecisionReasoning: async () => {
    try {
      const decisionReasoning = await aiService.getDecisionReasoning();
      set((state) => ({
        ai: { ...state.ai, decisionReasoning }
      }));
    } catch (error: any) {
    }
  },

  fetchCodeEvolution: async () => {
    try {
      const codeEvolution = await aiService.getCodeEvolution();
      set((state) => ({
        ai: { ...state.ai, codeEvolution }
      }));
    } catch (error: any) {
    }
  },

  fetchPatternRecognition: async () => {
    try {
      const patternRecognition = await aiService.getPatternRecognition();
      set((state) => ({
        ai: { ...state.ai, patternRecognition }
      }));
    } catch (error: any) {
    }
  },

  fetchPerformanceMetrics: async () => {
    try {
      const performanceMetrics = await aiService.getPerformanceMetrics();
      set((state) => ({
        ai: { ...state.ai, performanceMetrics }
      }));
    } catch (error: any) {
    }
  },

  reportMistake: async (mistake: any) => {
    try {
      await aiService.reportMistake(mistake);
      // Refresh mistake learning data
      get().ai.fetchMistakeLearning();
    } catch (error: any) {
      set((state) => ({
        ai: {
          ...state.ai,
          error: error.message || "Failed to report mistake",
        }
      }));
      throw error;
    }
  },

  acceptSuggestion: async (suggestionId: string) => {
    try {
      await aiService.acceptSuggestion(suggestionId);
      // Refresh proactive assistance data
      get().ai.fetchProactiveAssistance();
    } catch (error: any) {
      set((state) => ({
        ai: {
          ...state.ai,
          error: error.message || "Failed to accept suggestion",
        }
      }));
      throw error;
    }
  },

  declineSuggestion: async (suggestionId: string, reason?: string) => {
    try {
      await aiService.declineSuggestion(suggestionId, reason);
      // Refresh proactive assistance data
      get().ai.fetchProactiveAssistance();
    } catch (error: any) {
      set((state) => ({
        ai: {
          ...state.ai,
          error: error.message || "Failed to decline suggestion",
        }
      }));
      throw error;
    }
  },

  clearError: () => {
    set((state) => ({
      ai: { ...state.ai, error: null }
    }));
  },
});
