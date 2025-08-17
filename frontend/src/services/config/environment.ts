import { Environment } from "../types";
import { getBaseUrlForPath } from "../apiConfig";

// Environment detection
const hostname = window.location.hostname;
const isDevelopment = import.meta.env.DEV;

// Get base API URL
const getApiBaseUrl = (): string => {
  // Check environment variable first
  const envApiUrl = import.meta.env.VITE_API_URL;
  if (envApiUrl) {
    return envApiUrl;
  }

  // Use existing config logic
  return getBaseUrlForPath("/", hostname);
};

// Get WebSocket URL
const getWsBaseUrl = (): string => {
  const apiBaseUrl = getApiBaseUrl();
  if (!apiBaseUrl) {
    // Development mode - use relative path
    return `ws://${hostname}:${window.location.port}`;
  }
  
  // Replace http with ws
  return apiBaseUrl.replace(/^http/, "ws");
};

// Environment configuration
export const environment: Environment = {
  API_BASE_URL: getApiBaseUrl(),
  WS_BASE_URL: getWsBaseUrl(),
  TIMEOUT: parseInt(import.meta.env.VITE_API_TIMEOUT || "30000"),
  RETRY_ATTEMPTS: parseInt(import.meta.env.VITE_RETRY_ATTEMPTS || "3"),
  RETRY_DELAY: parseInt(import.meta.env.VITE_RETRY_DELAY || "1000"),
  CACHE_ENABLED: import.meta.env.VITE_CACHE_ENABLED !== "false",
  CACHE_TTL: parseInt(import.meta.env.VITE_CACHE_TTL || "300000"), // 5 minutes
  LOG_LEVEL: (import.meta.env.VITE_LOG_LEVEL as any) || (isDevelopment ? "info" : "error"),
};

// Export utility functions
export const isProduction = import.meta.env.PROD;
export const isDev = isDevelopment;
export const isLAN = hostname.startsWith("192.168.1.");

// Export all environment info
export default {
  ...environment,
  isDevelopment,
  isProduction,
  isLAN,
  hostname,
};
