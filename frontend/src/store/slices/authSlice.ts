import { StateCreator } from "zustand";
import { RootState, AuthState } from "../types";
import { authService } from "../../services";
import { User, UserPreferences } from "../../services/types";

export interface AuthSlice extends AuthState {
  // Actions
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  setApiKey: (apiKey: string) => void;
  updatePreferences: (preferences: Partial<UserPreferences>) => Promise<void>;
  checkAuth: () => Promise<void>;
  refreshToken: () => Promise<void>;
  updateProfile: (updates: Partial<User>) => Promise<void>;
  // Selectors
  isLoggedIn: boolean;
  isTokenExpired: boolean;
  isTokenExpiringSoon: boolean;
}

const initialState: AuthState = {
  user: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,
  token: null,
  apiKey: authService.getApiKey(),
};

export const createAuthSlice: StateCreator<
  RootState,
  [],
  [],
  AuthSlice
> = (set, get) => ({
  ...initialState,
  
  get isLoggedIn() {
    return get().auth.isAuthenticated && authService.isAuthenticated();
  },

  get isTokenExpired() {
    return authService.isTokenExpired();
  },

  get isTokenExpiringSoon() {
    return authService.isTokenExpiringSoon();
  },

  login: async (email: string, password: string) => {
    set((state) => ({ 
      auth: { ...state.auth, isLoading: true, error: null }
    }));

    try {
      const response = await authService.login({ email, password });
      const { user, tokens } = response;

      set((state) => ({
        auth: {
          ...state.auth,
          user,
          token: tokens.accessToken,
          isAuthenticated: true,
          isLoading: false,
          error: null,
        }
      }));
    } catch (error: any) {
      set((state) => ({
        auth: {
          ...state.auth,
          isLoading: false,
          error: error.message || "Login failed",
        }
      }));
      throw error;
    }
  },

  logout: async () => {
    set((state) => ({ 
      auth: { ...state.auth, isLoading: true }
    }));

    try {
      await authService.logout();
    } catch (error) {
    } finally {
      set((state) => ({
        auth: {
          ...initialState,
          apiKey: state.auth.apiKey, // Preserve API key
        }
      }));
    }
  },

  setApiKey: (apiKey: string) => {
    authService.setApiKey(apiKey);
    set((state) => ({
      auth: { ...state.auth, apiKey }
    }));
  },

  updatePreferences: async (newPreferences: Partial<UserPreferences>) => {
    set((state) => ({ 
      auth: { ...state.auth, isLoading: true, error: null }
    }));

    try {
      const preferences = await authService.updatePreferences(newPreferences);
      set((state) => {
        if (!state.auth.user) return state;
        
        const updatedUser = {
          ...state.auth.user,
          preferences,
        };

        return {
          auth: { 
            ...state.auth, 
            user: updatedUser,
            isLoading: false,
          }
        };
      });
    } catch (error: any) {
      set((state) => ({
        auth: {
          ...state.auth,
          isLoading: false,
          error: error.message || "Failed to update preferences",
        }
      }));
      throw error;
    }
  },

  updateProfile: async (updates: Partial<User>) => {
    set((state) => ({ 
      auth: { ...state.auth, isLoading: true, error: null }
    }));

    try {
      const user = await authService.updateProfile(updates);
      set((state) => ({
        auth: {
          ...state.auth,
          user,
          isLoading: false,
        }
      }));
    } catch (error: any) {
      set((state) => ({
        auth: {
          ...state.auth,
          isLoading: false,
          error: error.message || "Failed to update profile",
        }
      }));
      throw error;
    }
  },

  checkAuth: async () => {
    // Check if we have a valid token first
    if (!authService.isAuthenticated()) {
      set((state) => ({
        auth: { ...initialState, apiKey: state.auth.apiKey }
      }));
      return;
    }

    set((state) => ({ 
      auth: { ...state.auth, isLoading: true }
    }));

    try {
      const user = await authService.getCurrentUser();
      const tokenInfo = authService.getTokenInfo();

      set((state) => ({
        auth: {
          ...state.auth,
          user,
          token: tokenInfo ? "present" : null,
          isAuthenticated: true,
          isLoading: false,
          error: null,
        }
      }));
    } catch (error: any) {
      // Token is invalid, clear state
      set((state) => ({
        auth: { 
          ...initialState, 
          apiKey: state.auth.apiKey,
          error: error.message || "Authentication check failed",
        }
      }));
    }
  },

  refreshToken: async () => {
    try {
      const tokens = await authService.refreshToken();
      set((state) => ({
        auth: {
          ...state.auth,
          token: tokens.accessToken,
          error: null,
        }
      }));
    } catch (error: any) {
      // Token refresh failed, logout
      set((state) => ({
        auth: {
          ...initialState,
          apiKey: state.auth.apiKey,
          error: error.message || "Token refresh failed",
        }
      }));
    }
  },
});
