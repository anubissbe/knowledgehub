
import axios from 'axios';
import { API_CONFIG, AUTH_CONFIG } from '../apiConfig';

export interface AuthUser {
  id: string;
  email: string;
  name: string;
  roles: string[];
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  user: AuthUser;
}

class AuthService {
  private token: string | null = null;
  private refreshToken: string | null = null;

  constructor() {
    this.loadTokensFromStorage();
    this.setupAxiosInterceptors();
  }

  private loadTokensFromStorage() {
    this.token = localStorage.getItem(AUTH_CONFIG.TOKEN_KEY);
    this.refreshToken = localStorage.getItem(AUTH_CONFIG.TOKEN_REFRESH_KEY);
  }

  private saveTokensToStorage(accessToken: string, refreshToken: string) {
    localStorage.setItem(AUTH_CONFIG.TOKEN_KEY, accessToken);
    localStorage.setItem(AUTH_CONFIG.TOKEN_REFRESH_KEY, refreshToken);
    this.token = accessToken;
    this.refreshToken = refreshToken;
  }

  private clearTokensFromStorage() {
    localStorage.removeItem(AUTH_CONFIG.TOKEN_KEY);
    localStorage.removeItem(AUTH_CONFIG.TOKEN_REFRESH_KEY);
    this.token = null;
    this.refreshToken = null;
  }

  private setupAxiosInterceptors() {
    // Add authentication token to requests
    axios.interceptors.request.use((config) => {
      if (this.token) {
        config.headers.Authorization = `Bearer ${this.token}`;
      }
      return config;
    });

    // Handle token refresh on 401 responses
    axios.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (error.response?.status === 401 && this.refreshToken) {
          try {
            const response = await this.refreshAccessToken();
            // Retry the original request
            error.config.headers.Authorization = `Bearer ${response.access_token}`;
            return axios.request(error.config);
          } catch (refreshError) {
            this.logout();
            window.location.href = '/login';
          }
        }
        return Promise.reject(error);
      }
    );
  }

  async login(email: string, password: string): Promise<AuthResponse> {
    try {
      const response = await axios.post(`${API_CONFIG.BASE_URL}${AUTH_CONFIG.LOGIN_ENDPOINT}`, {
        email,
        password
      });

      const authData = response.data;
      this.saveTokensToStorage(authData.access_token, authData.refresh_token);
      
      return authData;
    } catch (error) {
      throw new Error('Login failed');
    }
  }

  async refreshAccessToken(): Promise<AuthResponse> {
    if (!this.refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await axios.post(`${API_CONFIG.BASE_URL}${AUTH_CONFIG.REFRESH_ENDPOINT}`, {
      refresh_token: this.refreshToken
    });

    const authData = response.data;
    this.saveTokensToStorage(authData.access_token, authData.refresh_token);
    
    return authData;
  }

  logout() {
    this.clearTokensFromStorage();
    // Optionally call logout endpoint
    if (this.token) {
      axios.post(`${API_CONFIG.BASE_URL}${AUTH_CONFIG.LOGOUT_ENDPOINT}`).catch(() => {
        // Ignore logout endpoint errors
      });
    }
  }

  isAuthenticated(): boolean {
    return !!this.token;
  }

  getToken(): string | null {
    return this.token;
  }
}

export const authService = new AuthService();
export default authService;
