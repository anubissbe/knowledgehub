import { StateCreator } from 'zustand';
import { RootState, UIState, NotificationState, Breadcrumb } from '../types';

export interface UISlice extends UIState {
  // Actions
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  addNotification: (notification: Omit<NotificationState, 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  openModal: (key: string, data?: any) => void;
  closeModal: (key: string) => void;
  setLoading: (key: string, loading: boolean) => void;
  setBreadcrumbs: (breadcrumbs: Breadcrumb[]) => void;
  setPageTitle: (title: string) => void;
}

const getInitialTheme = (): 'light' | 'dark' | 'system' => {
  const saved = localStorage.getItem('knowledgehub_darkMode');
  if (saved === null) return 'system';
  return JSON.parse(saved) ? 'dark' : 'light';
};

const initialState: UIState = {
  theme: getInitialTheme(),
  sidebarOpen: window.innerWidth > 768, // Open on desktop by default
  sidebarCollapsed: false,
  notifications: [],
  modals: {},
  loading: {},
  breadcrumbs: [],
  pageTitle: 'KnowledgeHub',
};

export const createUISlice: StateCreator<
  RootState,
  [],
  [],
  UISlice
> = (set, get) => ({
  ...initialState,

  setTheme: (theme: 'light' | 'dark' | 'system') => {
    // Update localStorage for backward compatibility with existing theme system
    if (theme === 'system') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      localStorage.setItem('knowledgehub_darkMode', JSON.stringify(prefersDark));
    } else {
      localStorage.setItem('knowledgehub_darkMode', JSON.stringify(theme === 'dark'));
    }

    set((state) => ({
      ui: { ...state.ui, theme }
    }));

    // Dispatch custom event to notify theme context
    window.dispatchEvent(new CustomEvent('themeChange', { detail: theme }));
  },

  toggleSidebar: () => {
    set((state) => ({
      ui: { ...state.ui, sidebarOpen: !state.ui.sidebarOpen }
    }));
  },

  setSidebarCollapsed: (collapsed: boolean) => {
    set((state) => ({
      ui: { ...state.ui, sidebarCollapsed: collapsed }
    }));
  },

  addNotification: (notificationData: Omit<NotificationState, 'id' | 'timestamp'>) => {
    const notification: NotificationState = {
      ...notificationData,
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      timestamp: new Date().toISOString(),
    };

    set((state) => ({
      ui: {
        ...state.ui,
        notifications: [...state.ui.notifications, notification],
      }
    }));

    // Auto-remove notification after duration
    if (notification.duration && notification.duration > 0) {
      setTimeout(() => {
        get().ui.removeNotification(notification.id);
      }, notification.duration);
    }
  },

  removeNotification: (id: string) => {
    set((state) => ({
      ui: {
        ...state.ui,
        notifications: state.ui.notifications.filter(n => n.id !== id),
      }
    }));
  },

  clearNotifications: () => {
    set((state) => ({
      ui: { ...state.ui, notifications: [] }
    }));
  },

  openModal: (key: string, data?: any) => {
    set((state) => ({
      ui: {
        ...state.ui,
        modals: {
          ...state.ui.modals,
          [key]: { open: true, data },
        },
      }
    }));
  },

  closeModal: (key: string) => {
    set((state) => ({
      ui: {
        ...state.ui,
        modals: {
          ...state.ui.modals,
          [key]: { open: false },
        },
      }
    }));
  },

  setLoading: (key: string, loading: boolean) => {
    set((state) => ({
      ui: {
        ...state.ui,
        loading: {
          ...state.ui.loading,
          [key]: loading,
        },
      }
    }));
  },

  setBreadcrumbs: (breadcrumbs: Breadcrumb[]) => {
    set((state) => ({
      ui: { ...state.ui, breadcrumbs }
    }));
  },

  setPageTitle: (title: string) => {
    // Update document title
    document.title = `${title} - KnowledgeHub`;
    
    set((state) => ({
      ui: { ...state.ui, pageTitle: title }
    }));
  },
});
