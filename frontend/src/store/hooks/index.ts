import { useEffect, useCallback } from 'react';
import { useStore, useUI, useAuth, useAI } from '../index';

// Hook for managing notifications
export const useNotifications = () => {
  const { notifications, addNotification, removeNotification, clearNotifications } = useUI();

  const notify = useCallback(
    (type: 'success' | 'error' | 'warning' | 'info', title: string, message: string, duration?: number) => {
      addNotification({
        type,
        title,
        message,
        duration: duration ?? (type === 'error' ? 0 : 5000), // Errors persist, others auto-hide
      });
    },
    [addNotification]
  );

  const success = useCallback((title: string, message: string, duration?: number) => {
    notify('success', title, message, duration);
  }, [notify]);

  const error = useCallback((title: string, message: string, duration?: number) => {
    notify('error', title, message, duration);
  }, [notify]);

  const warning = useCallback((title: string, message: string, duration?: number) => {
    notify('warning', title, message, duration);
  }, [notify]);

  const info = useCallback((title: string, message: string, duration?: number) => {
    notify('info', title, message, duration);
  }, [notify]);

  return {
    notifications,
    notify,
    success,
    error,
    warning,
    info,
    remove: removeNotification,
    clear: clearNotifications,
  };
};

// Hook for managing modals
export const useModal = (key: string) => {
  const { modals, openModal, closeModal } = useUI();
  const modalState = modals[key] || { open: false };

  const open = useCallback((data?: any) => {
    openModal(key, data);
  }, [key, openModal]);

  const close = useCallback(() => {
    closeModal(key);
  }, [key, closeModal]);

  return {
    isOpen: modalState.open,
    data: modalState.data,
    open,
    close,
  };
};

// Hook for managing loading states
export const useLoading = (key: string) => {
  const { loading, setLoading } = useUI();
  const isLoading = loading[key] || false;

  const setLoadingState = useCallback((loading: boolean) => {
    setLoading(key, loading);
  }, [key, setLoading]);

  return [isLoading, setLoadingState] as const;
};

// Hook for async operations with automatic loading and error handling
export const useAsyncAction = (key: string) => {
  const [isLoading, setLoading] = useLoading(key);
  const { error: notifyError } = useNotifications();

  const execute = useCallback(
    async <T>(
      action: () => Promise<T>,
      options?: {
        onSuccess?: (result: T) => void;
        onError?: (error: any) => void;
        successMessage?: { title: string; message: string };
        errorTitle?: string;
      }
    ) => {
      setLoading(true);
      try {
        const result = await action();
        if (options?.onSuccess) {
          options.onSuccess(result);
        }
        if (options?.successMessage) {
          const { success } = useNotifications();
          success(options.successMessage.title, options.successMessage.message);
        }
        return result;
      } catch (error: any) {
        const errorMessage = error.response?.data?.message || error.message || 'An error occurred';
        if (options?.onError) {
          options.onError(error);
        } else {
          notifyError(
            options?.errorTitle || 'Operation Failed', 
            errorMessage
          );
        }
        throw error;
      } finally {
        setLoading(false);
      }
    },
    [setLoading, notifyError]
  );

  return { execute, isLoading };
};

// Hook for authentication operations
export const useAuthActions = () => {
  const { login, logout, setApiKey, updatePreferences } = useAuth();
  const { success, error: notifyError } = useNotifications();

  const handleLogin = useCallback(
    async (email: string, password: string) => {
      try {
        await login(email, password);
        success('Welcome!', 'You have successfully logged in.');
      } catch (error) {
        // Error notification is already handled in the login action
        throw error;
      }
    },
    [login, success]
  );

  const handleLogout = useCallback(() => {
    logout();
    success('Logged out', 'You have been successfully logged out.');
  }, [logout, success]);

  const handleApiKeyUpdate = useCallback(
    (apiKey: string) => {
      setApiKey(apiKey);
      success('API Key Updated', 'Your API key has been saved successfully.');
    },
    [setApiKey, success]
  );

  return {
    login: handleLogin,
    logout: handleLogout,
    setApiKey: handleApiKeyUpdate,
    updatePreferences,
  };
};

// Hook for real-time connection management
export const useRealTimeConnection = () => {
  const { isConnected, connectWebSocket, disconnectWebSocket } = useAI();
  const { features, toggleFeature } = useStore((state) => state.app);

  useEffect(() => {
    if (features.realTimeUpdates && !isConnected) {
      connectWebSocket();
    }

    return () => {
      if (isConnected) {
        disconnectWebSocket();
      }
    };
  }, [features.realTimeUpdates, isConnected, connectWebSocket, disconnectWebSocket]);

  const toggleRealTime = useCallback(
    (enabled: boolean) => {
      toggleFeature('realTimeUpdates', enabled);
    },
    [toggleFeature]
  );

  return {
    isConnected,
    toggle: toggleRealTime,
    enabled: features.realTimeUpdates,
  };
};

// Hook for page management (title, breadcrumbs)
export const usePage = () => {
  const { setPageTitle, setBreadcrumbs, pageTitle, breadcrumbs } = useUI();

  const setPage = useCallback(
    (title: string, breadcrumbs?: Array<{ label: string; path: string }>) => {
      setPageTitle(title);
      if (breadcrumbs) {
        setBreadcrumbs(breadcrumbs);
      }
    },
    [setPageTitle, setBreadcrumbs]
  );

  return {
    title: pageTitle,
    breadcrumbs,
    setPage,
    setTitle: setPageTitle,
    setBreadcrumbs,
  };
};

// Hook for memory search with debouncing
export const useMemorySearch = () => {
  const { searchMemories, searchResults, isSearching, addRecentQuery, recentQueries } = useStore(
    (state) => state.memory
  );

  const search = useCallback(
    async (query: string, filters?: any) => {
      if (!query.trim()) return;
      
      try {
        await searchMemories(query, filters);
      } catch (error) {
      }
    },
    [searchMemories]
  );

  return {
    search,
    results: searchResults,
    isSearching,
    recentQueries,
  };
};
