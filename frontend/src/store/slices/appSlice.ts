import { StateCreator } from 'zustand';
import { RootState, AppState } from '../types';

export interface AppSlice extends AppState {
  // Actions
  initialize: () => Promise<void>;
  updatePerformance: (metrics: Partial<AppState['performance']>) => void;
  toggleFeature: (feature: string, enabled: boolean) => void;
  trackRender: (renderTime: number) => void;
}

const initialState: AppState = {
  isInitialized: false,
  version: '1.0.0',
  environment: import.meta.env.PROD ? 'production' : 'development',
  features: {
    realTimeUpdates: true,
    advancedSearch: true,
    contextualMemory: true,
    aiInsights: true,
    knowledgeGraph: true,
    performanceMonitoring: import.meta.env.DEV,
  },
  performance: {
    lastRender: 0,
    renderCount: 0,
    averageRenderTime: 0,
  },
};

export const createAppSlice: StateCreator<
  RootState,
  [],
  [],
  AppSlice
> = (set, get) => ({
  ...initialState,

  initialize: async () => {
    if (get().app.isInitialized) return;

    try {
      // Initialize auth check
      const { checkAuth } = get().auth;
      await checkAuth();

      // Connect WebSocket for real-time updates
      if (get().app.features.realTimeUpdates) {
        const { connectWebSocket } = get().ai;
        connectWebSocket();
      }

      // Fetch initial data
      const { fetchFeatures } = get().ai;
      const { fetchMemories, fetchContexts } = get().memory;

      await Promise.allSettled([
        fetchFeatures(),
        fetchMemories(),
        fetchContexts(),
      ]);

      set((state) => ({
        app: { ...state.app, isInitialized: true }
      }));
    } catch (error) {
      // Don't block initialization for non-critical failures
      set((state) => ({
        app: { ...state.app, isInitialized: true }
      }));
    }
  },

  updatePerformance: (metrics: Partial<AppState['performance']>) => {
    set((state) => ({
      app: {
        ...state.app,
        performance: {
          ...state.app.performance,
          ...metrics,
        }
      }
    }));
  },

  toggleFeature: (feature: string, enabled: boolean) => {
    set((state) => ({
      app: {
        ...state.app,
        features: {
          ...state.app.features,
          [feature]: enabled,
        }
      }
    }));

    // Handle feature-specific logic
    if (feature === 'realTimeUpdates') {
      const { connectWebSocket, disconnectWebSocket } = get().ai;
      if (enabled) {
        connectWebSocket();
      } else {
        disconnectWebSocket();
      }
    }
  },

  trackRender: (renderTime: number) => {
    set((state) => {
      const { performance } = state.app;
      const newRenderCount = performance.renderCount + 1;
      const totalTime = performance.averageRenderTime * performance.renderCount + renderTime;
      const newAverageRenderTime = totalTime / newRenderCount;

      return {
        app: {
          ...state.app,
          performance: {
            lastRender: renderTime,
            renderCount: newRenderCount,
            averageRenderTime: newAverageRenderTime,
          }
        }
      };
    });
  },
});
