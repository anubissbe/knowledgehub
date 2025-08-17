// Error handling utilities and custom error classes

import { ApiError } from "../types";

// Custom error classes
export class ApiClientError extends Error {
  public code?: string | number;
  public details?: any;
  public timestamp: string;
  public requestId?: string;

  constructor(message: string, code?: string | number, details?: any, requestId?: string) {
    super(message);
    this.name = "ApiClientError";
    this.code = code;
    this.details = details;
    this.timestamp = new Date().toISOString();
    this.requestId = requestId;
  }

  static fromApiError(error: ApiError): ApiClientError {
    return new ApiClientError(
      error.message,
      error.code,
      error.details,
      error.requestId
    );
  }

  toApiError(): ApiError {
    return {
      message: this.message,
      code: this.code,
      details: this.details,
      timestamp: this.timestamp,
      requestId: this.requestId,
    };
  }
}

export class NetworkError extends ApiClientError {
  constructor(message = "Network error", details?: any) {
    super(message, "NETWORK_ERROR", details);
    this.name = "NetworkError";
  }
}

export class TimeoutError extends ApiClientError {
  constructor(message = "Request timeout", timeout?: number) {
    super(message, "TIMEOUT_ERROR", { timeout });
    this.name = "TimeoutError";
  }
}

export class AuthError extends ApiClientError {
  constructor(message = "Authentication failed", code?: string | number) {
    super(message, code || "AUTH_ERROR");
    this.name = "AuthError";
  }
}

export class ValidationError extends ApiClientError {
  constructor(message = "Validation failed", fields?: Record<string, string[]>) {
    super(message, "VALIDATION_ERROR", { fields });
    this.name = "ValidationError";
  }
}

export class ServerError extends ApiClientError {
  constructor(message = "Server error", code?: number) {
    super(message, code || "SERVER_ERROR");
    this.name = "ServerError";
  }
}

// Error utilities
export class ErrorHandler {
  static handle(error: unknown, context?: string): ApiClientError {
    if (error instanceof ApiClientError) {
      return error;
    }

    if (error instanceof Error) {
      // Map common error types
      if (error.name === "TypeError" && error.message.includes("fetch")) {
        return new NetworkError("Network connection failed", { originalError: error.message });
      }

      if (error.name === "AbortError") {
        return new TimeoutError("Request was aborted");
      }

      return new ApiClientError(error.message, "UNKNOWN_ERROR", { originalError: error });
    }

    // Handle non-Error objects
    if (typeof error === "string") {
      return new ApiClientError(error);
    }

    return new ApiClientError("An unknown error occurred", "UNKNOWN_ERROR", { error });
  }

  static isRetryable(error: ApiClientError): boolean {
    // Network errors are retryable
    if (error instanceof NetworkError || error instanceof TimeoutError) {
      return true;
    }

    // 5xx server errors are retryable
    if (typeof error.code === "number" && error.code >= 500) {
      return true;
    }

    // Specific error codes that are retryable
    const retryableCodes = ["NETWORK_ERROR", "TIMEOUT_ERROR", "RATE_LIMIT"];
    return retryableCodes.includes(error.code as string);
  }

  static getRetryDelay(attempt: number, baseDelay = 1000): number {
    // Exponential backoff with jitter
    const delay = Math.min(baseDelay * Math.pow(2, attempt), 10000);
    const jitter = Math.random() * 0.1 * delay;
    return delay + jitter;
  }

  static getUserFriendlyMessage(error: ApiClientError): string {
    switch (error.code) {
      case "NETWORK_ERROR":
        return "Unable to connect to the server. Please check your internet connection.";
      case "TIMEOUT_ERROR":
        return "The request took too long to complete. Please try again.";
      case "AUTH_ERROR":
        return "You need to log in to access this feature.";
      case "VALIDATION_ERROR":
        return "Please check your input and try again.";
      case "RATE_LIMIT":
        return "Too many requests. Please wait a moment before trying again.";
      case 404:
        return "The requested resource could not be found.";
      case 500:
        return "Server error. Please try again later.";
      default:
        return error.message || "An unexpected error occurred.";
    }
  }
}

// Error response parser
export function parseErrorResponse(response: any): ApiError {
  if (response?.data?.message) {
    return {
      message: response.data.message,
      code: response.status || response.data.code,
      details: response.data.details,
      timestamp: new Date().toISOString(),
    };
  }

  if (response?.statusText) {
    return {
      message: response.statusText,
      code: response.status,
      timestamp: new Date().toISOString(),
    };
  }

  return {
    message: "An error occurred",
    code: "UNKNOWN_ERROR",
    timestamp: new Date().toISOString(),
  };
}

// Retry utility
export async function withRetry<T>(
  operation: () => Promise<T>,
  maxAttempts = 3,
  baseDelay = 1000,
  isRetryable: (error: ApiClientError) => boolean = ErrorHandler.isRetryable
): Promise<T> {
  let lastError: ApiClientError;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = ErrorHandler.handle(error, `Retry attempt ${attempt + 1}`);

      if (attempt === maxAttempts - 1 || !isRetryable(lastError)) {
        throw lastError;
      }

      const delay = ErrorHandler.getRetryDelay(attempt, baseDelay);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError!;
}
