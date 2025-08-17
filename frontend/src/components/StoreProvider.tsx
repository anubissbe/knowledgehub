import React, { useEffect } from 'react';
import { useStore } from '../store';
import { performanceUtils } from '../store/utils';

interface StoreProviderProps {
  children: React.ReactNode;
}

/**
 * Provider component that initializes the KnowledgeHub store
 * and handles app-wide initialization logic
 */
export const StoreProvider: React.FC<StoreProviderProps> = ({ children }) => {
  const initialize = useStore((state) => state.app.initialize);
  const isInitialized = useStore((state) => state.app.isInitialized);
  
  useEffect(() => {
    // Initialize the app when the provider mounts
    const startTime = performance.now();
    
    initialize().then(() => {
      const endTime = performance.now();
    }).catch((error) => {
    });
  }, [initialize]);

  // Add performance monitoring in development
  useEffect(() => {
    if (import.meta.env.DEV && isInitialized) {
      const trackPerformance = performanceUtils.throttle(() => {
        const { performance: perf } = useStore.getState().app;
        if (perf.averageRenderTime > 20) {
        }
      }, 5000);

      // Subscribe to store changes for performance monitoring
      const unsubscribe = useStore.subscribe(
        (state) => state.app.performance,
        trackPerformance
      );

      return unsubscribe;
    }
  }, [isInitialized]);

  return <>{children}</>;
};
