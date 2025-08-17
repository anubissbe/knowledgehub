import { create } from 'zustand';
import { devtools, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { RootState } from './types';
import { createAuthSlice, AuthSlice } from './slices/authSlice';
import { createAISlice, AISlice } from './slices/aiSlice';
import { createMemorySlice, MemorySlice } from './slices/memorySlice';
import { createUISlice, UISlice } from './slices/uiSlice';
import { createAppSlice, AppSlice } from './slices/appSlice';

// Combined store interface
export interface KnowledgeHubStore {
  auth: AuthSlice;
  ai: AISlice;
  memory: MemorySlice;
  ui: UISlice;
  app: AppSlice;
}

// Create the store with all slices combined
export const useStore = create<KnowledgeHubStore>()(
  devtools(
    subscribeWithSelector(
      immer((set, get, api) => ({
        auth: createAuthSlice(set, get, api),
        ai: createAISlice(set, get, api),
        memory: createMemorySlice(set, get, api),
        ui: createUISlice(set, get, api),
        app: createAppSlice(set, get, api),
      }))
    ),
    {
      name: 'knowledgehub-store',
      enabled: import.meta.env.DEV,
    }
  )
);

// Export selectors for better performance and type safety
export const useAuth = () => useStore((state) => state.auth);
export const useAI = () => useStore((state) => state.ai);
export const useMemory = () => useStore((state) => state.memory);
export const useUI = () => useStore((state) => state.ui);
export const useApp = () => useStore((state) => state.app);

// Specific selectors for commonly used data
export const useIsAuthenticated = () => useStore((state) => state.auth.isAuthenticated);
export const useCurrentUser = () => useStore((state) => state.auth.user);
export const useTheme = () => useStore((state) => state.ui.theme);
export const useNotifications = () => useStore((state) => state.ui.notifications);
export const useLoadingState = (key: string) => useStore((state) => state.ui.loading[key] || false);
export const useModalState = (key: string) => useStore((state) => state.ui.modals[key] || { open: false });
export const useAIMetrics = () => useStore((state) => state.ai.metrics);
export const useSearchResults = () => useStore((state) => state.memory.searchResults);
export const useSelectedMemory = () => useStore((state) => state.memory.selectedMemory);

// Performance tracking hook
export const usePerformanceTracker = () => {
  const trackRender = useStore((state) => state.app.trackRender);
  
  return (renderTime: number) => {
    if (import.meta.env.DEV) {
      trackRender(renderTime);
    }
  };
};

// Initialize the app
export const initializeApp = () => {
  const initialize = useStore.getState().app.initialize;
  return initialize();
};
