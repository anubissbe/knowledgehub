// WebSocket manager with auto-reconnection, event routing, and health monitoring

import { WebSocketMessage, WebSocketError, ConnectionState } from "../types";
import { environment } from "../config/environment";
import { ErrorHandler } from "../errors";

// Event handler type
type EventHandler<T = any> = (payload: T) => void;

// WebSocket manager options
interface WebSocketManagerOptions {
  url?: string;
  protocols?: string | string[];
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  enableHeartbeat?: boolean;
}

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private protocols?: string | string[];
  private reconnectInterval: number;
  private maxReconnectAttempts: number;
  private heartbeatInterval: number;
  private enableHeartbeat: boolean;

  // State management
  private state: ConnectionState = {
    isConnected: false,
    isConnecting: false,
    reconnectAttempts: 0,
  };

  // Event handlers
  private eventHandlers = new Map<string, Set<EventHandler>>();
  private globalHandlers = new Set<EventHandler<WebSocketMessage>>();

  // Timers
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;

  // Message queue for offline messages
  private messageQueue: WebSocketMessage[] = [];
  private maxQueueSize = 100;

  constructor(options: WebSocketManagerOptions = {}) {
    this.url = options.url || this.getWebSocketUrl();
    this.protocols = options.protocols;
    this.reconnectInterval = options.reconnectInterval || 5000;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 10;
    this.heartbeatInterval = options.heartbeatInterval || 30000;
    this.enableHeartbeat = options.enableHeartbeat ?? true;
  }

  private getWebSocketUrl(): string {
    const baseUrl = environment.WS_BASE_URL || environment.API_BASE_URL.replace(/^http/, "ws");
    return `${baseUrl}/ws`;
  }

  // Connection management
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.state.isConnected || this.state.isConnecting) {
        resolve();
        return;
      }

      this.setState({ isConnecting: true });

      try {
        this.ws = new WebSocket(this.url, this.protocols);
        this.setupEventListeners(resolve, reject);
      } catch (error) {
        this.setState({ isConnecting: false, error: this.createError(error) });
        reject(ErrorHandler.handle(error, "WebSocket connection"));
      }
    });
  }

  disconnect(): void {
    this.clearTimers();
    this.setState({ isConnecting: false });

    if (this.ws) {
      this.ws.close(1000, "Manual disconnect");
      this.ws = null;
    }

    this.setState({ isConnected: false });
  }

  reconnect(): void {
    if (this.state.reconnectAttempts >= this.maxReconnectAttempts) {
      return;
    }

    this.disconnect();
    this.setState({ reconnectAttempts: this.state.reconnectAttempts + 1 });

    this.reconnectTimer = setTimeout(() => {
      this.connect().catch((error) => {
        this.scheduleReconnect();
      });
    }, this.getReconnectDelay());
  }

  private scheduleReconnect(): void {
    if (this.state.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectTimer = setTimeout(() => {
        this.reconnect();
      }, this.getReconnectDelay());
    }
  }

  private getReconnectDelay(): number {
    // Exponential backoff with jitter
    const baseDelay = this.reconnectInterval;
    const exponentialDelay = Math.min(baseDelay * Math.pow(2, this.state.reconnectAttempts), 30000);
    const jitter = Math.random() * 1000;
    return exponentialDelay + jitter;
  }

  // Event setup
  private setupEventListeners(
    resolve: () => void,
    reject: (error: Error) => void
  ): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      this.setState({
        isConnected: true,
        isConnecting: false,
        reconnectAttempts: 0,
        lastConnectedAt: new Date(),
        error: undefined,
      });

      this.startHeartbeat();
      this.processQueuedMessages();
      resolve();
    };

    this.ws.onclose = (event) => {
      this.setState({ isConnected: false, isConnecting: false });
      this.stopHeartbeat();

      // Auto-reconnect if not manual disconnect
      if (event.code !== 1000) {
        this.scheduleReconnect();
      }
    };

    this.ws.onerror = (event) => {
      const error = this.createError("WebSocket connection error");
      this.setState({ error, isConnecting: false });
      reject(ErrorHandler.handle(error, "WebSocket"));
    };

    this.ws.onmessage = (event) => {
      this.handleMessage(event.data);
    };
  }

  // Message handling
  private handleMessage(data: string): void {
    try {
      const message: WebSocketMessage = JSON.parse(data);
      
      // Handle system messages
      if (message.type === "pong") {
        // Heartbeat response - no action needed
        return;
      }

      // Emit to specific event handlers
      const handlers = this.eventHandlers.get(message.type);
      if (handlers) {
        handlers.forEach(handler => {
          try {
            handler(message.payload);
          } catch (error) {
          }
        });
      }

      // Emit to global handlers
      this.globalHandlers.forEach(handler => {
        try {
          handler(message);
        } catch (error) {
        }
      });
    } catch (error) {
    }
  }

  // Message sending
  send<T = any>(type: string, payload: T): void {
    const message: WebSocketMessage<T> = {
      type,
      payload,
      timestamp: new Date().toISOString(),
      id: this.generateMessageId(),
    };

    if (this.state.isConnected && this.ws?.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(message));
      } catch (error) {
        this.queueMessage(message);
      }
    } else {
      this.queueMessage(message);
    }
  }

  private queueMessage(message: WebSocketMessage): void {
    if (this.messageQueue.length >= this.maxQueueSize) {
      this.messageQueue.shift(); // Remove oldest message
    }
    this.messageQueue.push(message);
  }

  private processQueuedMessages(): void {
    while (this.messageQueue.length > 0 && this.state.isConnected) {
      const message = this.messageQueue.shift()!;
      try {
        this.ws?.send(JSON.stringify(message));
      } catch (error) {
        // Re-queue the message
        this.messageQueue.unshift(message);
        break;
      }
    }
  }

  // Event handlers
  on<T = any>(event: string, handler: EventHandler<T>): () => void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }
    this.eventHandlers.get(event)!.add(handler);

    // Return unsubscribe function
    return () => {
      const handlers = this.eventHandlers.get(event);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          this.eventHandlers.delete(event);
        }
      }
    };
  }

  off(event: string, handler?: EventHandler): void {
    if (!handler) {
      this.eventHandlers.delete(event);
      return;
    }

    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.delete(handler);
      if (handlers.size === 0) {
        this.eventHandlers.delete(event);
      }
    }
  }

  onAny(handler: EventHandler<WebSocketMessage>): () => void {
    this.globalHandlers.add(handler);
    return () => this.globalHandlers.delete(handler);
  }

  offAny(handler?: EventHandler<WebSocketMessage>): void {
    if (handler) {
      this.globalHandlers.delete(handler);
    } else {
      this.globalHandlers.clear();
    }
  }

  // Heartbeat
  private startHeartbeat(): void {
    if (!this.enableHeartbeat) return;

    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.state.isConnected) {
        this.send("ping", { timestamp: Date.now() });
      }
    }, this.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  // Utility methods
  private setState(updates: Partial<ConnectionState>): void {
    this.state = { ...this.state, ...updates };
  }

  private createError(error: unknown): WebSocketError {
    return {
      code: -1,
      message: typeof error === "string" ? error : "WebSocket error",
      timestamp: new Date().toISOString(),
    };
  }

  private generateMessageId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private clearTimers(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.stopHeartbeat();
  }

  // Public getters
  get isConnected(): boolean {
    return this.state.isConnected;
  }

  get isConnecting(): boolean {
    return this.state.isConnecting;
  }

  get connectionState(): ConnectionState {
    return { ...this.state };
  }

  get queuedMessageCount(): number {
    return this.messageQueue.length;
  }

  // Health monitoring
  getHealthStatus() {
    return {
      isConnected: this.state.isConnected,
      isConnecting: this.state.isConnecting,
      reconnectAttempts: this.state.reconnectAttempts,
      lastConnectedAt: this.state.lastConnectedAt,
      queuedMessages: this.messageQueue.length,
      error: this.state.error,
      readyState: this.ws?.readyState,
      url: this.url,
    };
  }
}

// Export singleton instance
export const wsManager = new WebSocketManager();

// Export class for testing and custom instances
