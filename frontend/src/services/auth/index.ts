// AuthService with token management, refresh logic, and user management

import { ApiResponse, User, AuthTokens, UserPreferences } from "../types";
import { apiClient } from "../api/client";
import { ErrorHandler, AuthError } from "../errors";

interface LoginRequest {
  email: string;
  password: string;
}

interface LoginResponse {
  user: User;
  tokens: AuthTokens;
  message?: string;
}

interface RefreshResponse {
  tokens: AuthTokens;
}

interface RegisterRequest {
  email: string;
  password: string;
  name: string;
}

export class AuthService {
  private tokenRefreshPromise: Promise<void> | null = null;
  private refreshTimer: NodeJS.Timeout | null = null;

  constructor() {
    this.setupTokenRefreshScheduler();
  }

  // Authentication methods
  async login(credentials: LoginRequest): Promise<LoginResponse> {
    try {
      const response = await apiClient.post<LoginResponse>("/api/auth/login", credentials, {
        cache: false,
      });

      if (response.data.tokens) {
        this.storeTokens(response.data.tokens);
        this.scheduleTokenRefresh(response.data.tokens);
      }

      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Authentication login");
    }
  }

  async logout(): Promise<void> {
    try {
      // Attempt to notify server of logout
      await apiClient.post("/api/auth/logout", {}, { cache: false });
    } catch (error) {
      // Continue with logout even if server call fails
    } finally {
      this.clearTokens();
      this.clearRefreshTimer();
    }
  }

  async refreshToken(): Promise<AuthTokens> {
    // Prevent multiple simultaneous refresh attempts
    if (this.tokenRefreshPromise) {
      await this.tokenRefreshPromise;
      return this.getStoredTokens()!;
    }

    this.tokenRefreshPromise = this.performTokenRefresh();

    try {
      await this.tokenRefreshPromise;
      return this.getStoredTokens()!;
    } finally {
      this.tokenRefreshPromise = null;
    }
  }

  private async performTokenRefresh(): Promise<void> {
    const tokens = this.getStoredTokens();
    if (!tokens?.refreshToken) {
      throw new AuthError("No refresh token available");
    }

    try {
      const response = await apiClient.post<RefreshResponse>(
        "/api/auth/refresh",
        { refreshToken: tokens.refreshToken },
        { cache: false }
      );

      this.storeTokens(response.data.tokens);
      this.scheduleTokenRefresh(response.data.tokens);
    } catch (error) {
      // If refresh fails, clear tokens and force re-login
      this.clearTokens();
      throw new AuthError("Token refresh failed. Please log in again.");
    }
  }

  // User management
  async getCurrentUser(): Promise<User> {
    try {
      const response = await apiClient.get<User>("/api/auth/me");
      return response.data;
    } catch (error) {
      throw ErrorHandler.handle(error, "Get current user");
    }
  }

  // Token management
  private storeTokens(tokens: AuthTokens): void {
    localStorage.setItem("knowledgehub_token", tokens.accessToken);
    localStorage.setItem("knowledgehub_refresh_token", tokens.refreshToken);
    localStorage.setItem("knowledgehub_token_expires", tokens.expiresAt.toString());
    
    // Also set on API client
    apiClient.setAuthToken(tokens.accessToken);
  }

  private getStoredTokens(): AuthTokens | null {
    const accessToken = localStorage.getItem("knowledgehub_token");
    const refreshToken = localStorage.getItem("knowledgehub_refresh_token");
    const expiresAt = localStorage.getItem("knowledgehub_token_expires");

    if (!accessToken || !refreshToken || !expiresAt) {
      return null;
    }

    return {
      accessToken,
      refreshToken,
      expiresAt: parseInt(expiresAt),
    };
  }

  private clearTokens(): void {
    localStorage.removeItem("knowledgehub_token");
    localStorage.removeItem("knowledgehub_refresh_token");
    localStorage.removeItem("knowledgehub_token_expires");
    
    // Also remove from API client
    apiClient.removeAuthToken();
  }

  private setupTokenRefreshScheduler(): void {
    // Check for existing token on service initialization
    const tokens = this.getStoredTokens();
    if (tokens) {
      this.scheduleTokenRefresh(tokens);
    }
  }

  private scheduleTokenRefresh(tokens: AuthTokens): void {
    this.clearRefreshTimer();

    const now = Date.now();
    const expiresAt = tokens.expiresAt;
    const refreshTime = expiresAt - (5 * 60 * 1000); // Refresh 5 minutes before expiry

    if (refreshTime <= now) {
      // Token is already expired or about to expire, refresh immediately
      this.refreshToken().catch((error) => {
      });
      return;
    }

    const timeUntilRefresh = refreshTime - now;
    this.refreshTimer = setTimeout(() => {
      this.refreshToken().catch((error) => {
      });
    }, timeUntilRefresh);
  }

  private clearRefreshTimer(): void {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
      this.refreshTimer = null;
    }
  }

  // State queries
  isAuthenticated(): boolean {
    const tokens = this.getStoredTokens();
    if (!tokens) return false;

    // Check if token is expired
    const now = Date.now();
    return tokens.expiresAt > now;
  }

  // API Key management (for backward compatibility)
  setApiKey(apiKey: string): void {
    const settings = JSON.parse(localStorage.getItem("knowledgehub_settings") || "{}");
    settings.apiKey = apiKey;
    localStorage.setItem("knowledgehub_settings", JSON.stringify(settings));
  }

  getApiKey(): string | null {
    const settings = JSON.parse(localStorage.getItem("knowledgehub_settings") || "{}");
    return settings.apiKey || null;
  }

  // Cleanup
  destroy(): void {
    this.clearRefreshTimer();
    this.tokenRefreshPromise = null;
  }
}

// Export singleton instance
export const authService = new AuthService();
