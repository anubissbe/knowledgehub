/**
 * WebSocket Service for Real-time Communication
 * 
 * This service provides:
 * - WebSocket connection management
 * - Automatic reconnection with exponential backoff
 * - Event subscription system
 * - Message queue for offline scenarios
 * - Real-time metrics and alerts
 */

interface WebSocketConfig {
  url?: string;
  maxReconnectAttempts?: number;
  reconnectInterval?: number;
  maxReconnectInterval?: number;
  heartbeatInterval?: number;
  messageQueueLimit?: number;
  autoReconnect?: boolean;
}

interface WebSocketMessage {
  type: string;
  data: any;
  timestamp?: string;
  message_id?: string;
  source?: string;
}

interface EventSubscription {
  channel: string;
  eventTypes?: string[];
  filters?: Record<string, any>;
  callback: (data: any) => void;
}

interface ConnectionInfo {
  id: string;
  connected: boolean;
  authenticated: boolean;
  lastHeartbeat: Date;
  reconnectAttempts: number;
}

type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'authenticated' | 'error';

type EventType = 
  | 'metric_update'
  | 'alert_triggered'
  | 'alert_resolved'
  | 'system_status'
  | 'memory_created'
  | 'memory_updated'
  | 'session_started'
  | 'session_ended'
  | 'error_occurred'
  | 'workflow_completed'
  | 'dashboard_update'
  | 'user_activity';

export class WebSocketService {
  private ws: WebSocket | null = null;
  private config: Required<WebSocketConfig>;
  private connectionState: ConnectionState = 'disconnected';
  private subscriptions: Map<string, EventSubscription> = new Map();
  private messageQueue: WebSocketMessage[] = [];
  private reconnectTimer: number | null = null;
  private heartbeatTimer: number | null = null;
  private connectionInfo: ConnectionInfo | null = null;
  
  // Event handlers
  private onConnectHandlers: (() => void)[] = [];
  private onDisconnectHandlers: (() => void)[] = [];
  private onAuthenticatedHandlers: (() => void)[] = [];
  private onErrorHandlers: ((error: Error) => void)[] = [];
  private onStateChangeHandlers: ((state: ConnectionState) => void)[] = [];
  
  // Authentication
  private authToken: string | null = null;
  private authPromise: Promise<boolean> | null = null;

  constructor(config: WebSocketConfig = {}) {
    this.config = {
      url: config.url || this.getWebSocketUrl(),
      maxReconnectAttempts: config.maxReconnectAttempts || 10,
      reconnectInterval: config.reconnectInterval || 1000,
      maxReconnectInterval: config.maxReconnectInterval || 30000,
      heartbeatInterval: config.heartbeatInterval || 30000,
      messageQueueLimit: config.messageQueueLimit || 100,
      autoReconnect: config.autoReconnect !== false,
    };
  }

  /**
   * Connect to the WebSocket server
   */
  async connect(token?: string): Promise<boolean> {
    if (this.connectionState === 'connecting' || this.connectionState === 'connected') {
      return this.connectionState === 'connected';
    }

    if (token) {
      this.authToken = token;
    }

    return new Promise((resolve, reject) => {
      try {
        this.setState('connecting');
        this.ws = new WebSocket(this.config.url);

        const connectTimeout = setTimeout(() => {
          if (this.ws?.readyState === WebSocket.CONNECTING) {
            this.ws.close();
            reject(new Error('Connection timeout'));
          }
        }, 10000);

        this.ws.onopen = () => {
          clearTimeout(connectTimeout);
          this.setState('connected');
          this.onConnected();
          
          // Authenticate if token is available
          if (this.authToken) {
            this.authenticate(this.authToken).then(resolve).catch(reject);
          } else {
            resolve(true);
          }
        };

        this.ws.onclose = (event) => {
          clearTimeout(connectTimeout);
          this.onDisconnected(event);
          resolve(false);
        };

        this.ws.onerror = (error) => {
          clearTimeout(connectTimeout);
          this.onError(new Error('WebSocket connection error'));
          reject(error);
        };

        this.ws.onmessage = (event) => {
          this.onMessage(event);
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect(): void {
    this.config.autoReconnect = false;
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }

    this.setState('disconnected');
  }

  /**
   * Authenticate the WebSocket connection
   */
  async authenticate(token: string): Promise<boolean> {
    if (this.authPromise) {
      return this.authPromise;
    }

    this.authToken = token;

    this.authPromise = new Promise((resolve, reject) => {
      if (this.connectionState !== 'connected') {
        reject(new Error('Not connected'));
        return;
      }

      const authMessage: WebSocketMessage = {
        type: 'auth',
        data: { token }
      };

      // Set up one-time listener for auth response
      const originalOnMessage = this.onMessage.bind(this);
      const authTimeout = setTimeout(() => {
        reject(new Error('Authentication timeout'));
      }, 5000);

      this.onMessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.type === 'auth') {
            clearTimeout(authTimeout);
            this.onMessage = originalOnMessage;
            
            if (message.data?.status === 'authenticated') {
              this.setState('authenticated');
              this.onAuthenticatedHandlers.forEach(handler => handler());
              resolve(true);
            } else {
              reject(new Error('Authentication failed'));
            }
          } else {
            // Pass other messages to original handler
            originalOnMessage(event);
          }
        } catch (error) {
          originalOnMessage(event);
        }
      };

      this.send(authMessage);
    });

    try {
      const result = await this.authPromise;
      return result;
    } finally {
      this.authPromise = null;
    }
  }

  /**
   * Subscribe to events on a channel
   */
  subscribe(
    subscriptionId: string,
    channel: string,
    callback: (data: any) => void,
    options: {
      eventTypes?: EventType[];
      filters?: Record<string, any>;
    } = {}
  ): () => void {
    const subscription: EventSubscription = {
      channel,
      eventTypes: options.eventTypes,
      filters: options.filters,
      callback
    };

    this.subscriptions.set(subscriptionId, subscription);

    // Send subscription message if connected
    if (this.connectionState === 'connected' || this.connectionState === 'authenticated') {
      const subscribeMessage: WebSocketMessage = {
        type: 'subscribe',
        data: {
          channel,
          event_types: options.eventTypes,
          filters: options.filters
        }
      };
      this.send(subscribeMessage);
    }

    // Return unsubscribe function
    return () => {
      this.unsubscribe(subscriptionId);
    };
  }

  /**
   * Unsubscribe from events
   */
  unsubscribe(subscriptionId: string): void {
    const subscription = this.subscriptions.get(subscriptionId);
    if (!subscription) return;

    this.subscriptions.delete(subscriptionId);

    // Send unsubscribe message if connected
    if (this.connectionState === 'connected' || this.connectionState === 'authenticated') {
      const unsubscribeMessage: WebSocketMessage = {
        type: 'unsubscribe',
        data: {
          channel: subscription.channel
        }
      };
      this.send(unsubscribeMessage);
    }
  }

  /**
   * Subscribe to real-time metrics
   */
  subscribeToMetrics(callback: (metric: any) => void): () => void {
    return this.subscribe('metrics', 'metrics', callback, {
      eventTypes: ['metric_update', 'metric_threshold']
    });
  }

  /**
   * Subscribe to alerts
   */
  subscribeToAlerts(callback: (alert: any) => void): () => void {
    return this.subscribe('alerts', 'alerts', callback, {
      eventTypes: ['alert_triggered', 'alert_resolved']
    });
  }

  /**
   * Subscribe to system status updates
   */
  subscribeToSystemStatus(callback: (status: any) => void): () => void {
    return this.subscribe('system', 'system', callback, {
      eventTypes: ['system_status', 'service_status']
    });
  }

  /**
   * Subscribe to memory system events
   */
  subscribeToMemoryEvents(callback: (event: any) => void): () => void {
    return this.subscribe('memory', 'memory', callback, {
      eventTypes: ['memory_created', 'memory_updated', 'memory_deleted']
    });
  }

  /**
   * Subscribe to session events
   */
  subscribeToSessionEvents(callback: (event: any) => void): () => void {
    return this.subscribe('sessions', 'sessions', callback, {
      eventTypes: ['session_started', 'session_ended', 'session_updated']
    });
  }

  /**
   * Subscribe to dashboard updates
   */
  subscribeToDashboard(callback: (data: any) => void): () => void {
    return this.subscribe('dashboard', 'analytics', callback, {
      eventTypes: ['dashboard_update', 'trend_change']
    });
  }

  /**
   * Subscribe to user-specific events
   */
  subscribeToUserEvents(userId: string, callback: (event: any) => void): () => void {
    return this.subscribe(`user_${userId}`, `user.${userId}`, callback);
  }

  /**
   * Subscribe to project-specific events
   */
  subscribeToProjectEvents(projectId: string, callback: (event: any) => void): () => void {
    return this.subscribe(`project_${projectId}`, `project.${projectId}`, callback);
  }

  /**
   * Send a message to the server
   */
  send(message: WebSocketMessage): boolean {
    if (this.connectionState === 'connected' || this.connectionState === 'authenticated') {
      try {
        this.ws?.send(JSON.stringify(message));
        return true;
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        return false;
      }
    } else {
      // Queue message for later
      if (this.messageQueue.length < this.config.messageQueueLimit) {
        this.messageQueue.push(message);
      }
      return false;
    }
  }

  /**
   * Get connection information
   */
  getConnectionInfo(): ConnectionInfo | null {
    return this.connectionInfo;
  }

  /**
   * Get current connection state
   */
  getState(): ConnectionState {
    return this.connectionState;
  }

  /**
   * Check if connected and authenticated
   */
  isReady(): boolean {
    return this.connectionState === 'authenticated';
  }

  /**
   * Event handlers
   */
  onConnect(handler: () => void): () => void {
    this.onConnectHandlers.push(handler);
    return () => {
      this.onConnectHandlers = this.onConnectHandlers.filter(h => h !== handler);
    };
  }

  onDisconnect(handler: () => void): () => void {
    this.onDisconnectHandlers.push(handler);
    return () => {
      this.onDisconnectHandlers = this.onDisconnectHandlers.filter(h => h !== handler);
    };
  }

  onAuthenticated(handler: () => void): () => void {
    this.onAuthenticatedHandlers.push(handler);
    return () => {
      this.onAuthenticatedHandlers = this.onAuthenticatedHandlers.filter(h => h !== handler);
    };
  }

  onError(handler: (error: Error) => void): () => void {
    this.onErrorHandlers.push(handler);
    return () => {
      this.onErrorHandlers = this.onErrorHandlers.filter(h => h !== handler);
    };
  }

  onStateChange(handler: (state: ConnectionState) => void): () => void {
    this.onStateChangeHandlers.push(handler);
    return () => {
      this.onStateChangeHandlers = this.onStateChangeHandlers.filter(h => h !== handler);
    };
  }

  // Private methods

  private getWebSocketUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/ws`;
  }

  private setState(state: ConnectionState): void {
    if (this.connectionState !== state) {
      this.connectionState = state;
      this.onStateChangeHandlers.forEach(handler => handler(state));
    }
  }

  private onConnected(): void {
    console.log('WebSocket connected');
    
    this.connectionInfo = {
      id: this.generateConnectionId(),
      connected: true,
      authenticated: false,
      lastHeartbeat: new Date(),
      reconnectAttempts: 0
    };

    // Start heartbeat
    this.startHeartbeat();

    // Process queued messages
    this.processMessageQueue();

    // Resubscribe to channels
    this.resubscribeAll();

    // Notify handlers
    this.onConnectHandlers.forEach(handler => handler());
  }

  private onDisconnected(event: CloseEvent): void {
    console.log('WebSocket disconnected:', event.code, event.reason);
    
    this.setState('disconnected');
    
    if (this.connectionInfo) {
      this.connectionInfo.connected = false;
      this.connectionInfo.authenticated = false;
    }

    // Stop heartbeat
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    // Notify handlers
    this.onDisconnectHandlers.forEach(handler => handler());

    // Auto-reconnect if enabled
    if (this.config.autoReconnect && event.code !== 1000) {
      this.scheduleReconnect();
    }
  }

  private onError(error: Error): void {
    console.error('WebSocket error:', error);
    this.setState('error');
    this.onErrorHandlers.forEach(handler => handler(error));
  }

  private onMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      
      // Update heartbeat
      if (this.connectionInfo) {
        this.connectionInfo.lastHeartbeat = new Date();
      }

      // Handle different message types
      switch (message.type) {
        case 'pong':
        case 'heartbeat':
          // Heartbeat responses
          break;
          
        case 'error':
          console.error('WebSocket server error:', message.data);
          break;
          
        case 'data':
        case 'notification':
          this.handleDataMessage(message);
          break;
          
        default:
          console.log('Unknown message type:', message.type);
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }

  private handleDataMessage(message: WebSocketMessage): void {
    const { data } = message;
    
    if (!data.event_type) return;

    // Route to appropriate subscribers
    this.subscriptions.forEach((subscription, id) => {
      try {
        // Check if subscription matches
        if (this.matchesSubscription(subscription, data)) {
          subscription.callback(data);
        }
      } catch (error) {
        console.error(`Error in subscription callback ${id}:`, error);
      }
    });
  }

  private matchesSubscription(subscription: EventSubscription, data: any): boolean {
    // Check event types filter
    if (subscription.eventTypes && subscription.eventTypes.length > 0) {
      if (!subscription.eventTypes.includes(data.event_type)) {
        return false;
      }
    }

    // Check custom filters
    if (subscription.filters) {
      for (const [key, value] of Object.entries(subscription.filters)) {
        if (data[key] !== value) {
          return false;
        }
      }
    }

    return true;
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = window.setInterval(() => {
      if (this.connectionState === 'connected' || this.connectionState === 'authenticated') {
        this.send({ type: 'ping', data: {} });
      }
    }, this.config.heartbeatInterval);
  }

  private processMessageQueue(): void {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message) {
        this.send(message);
      }
    }
  }

  private resubscribeAll(): void {
    this.subscriptions.forEach((subscription, id) => {
      const subscribeMessage: WebSocketMessage = {
        type: 'subscribe',
        data: {
          channel: subscription.channel,
          event_types: subscription.eventTypes,
          filters: subscription.filters
        }
      };
      this.send(subscribeMessage);
    });
  }

  private scheduleReconnect(): void {
    if (!this.connectionInfo) return;

    const attempt = this.connectionInfo.reconnectAttempts;
    if (attempt >= this.config.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    // Exponential backoff
    const delay = Math.min(
      this.config.reconnectInterval * Math.pow(2, attempt),
      this.config.maxReconnectInterval
    );

    console.log(`Reconnecting in ${delay}ms (attempt ${attempt + 1})`);

    this.reconnectTimer = window.setTimeout(async () => {
      if (this.connectionInfo) {
        this.connectionInfo.reconnectAttempts++;
      }
      
      try {
        await this.connect();
      } catch (error) {
        console.error('Reconnection failed:', error);
        this.scheduleReconnect();
      }
    }, delay);
  }

  private generateConnectionId(): string {
    return `conn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

// Global WebSocket service instance
export const websocketService = new WebSocketService();

// Enhanced real-time service that uses WebSocket
export class EnhancedRealTimeService {
  private wsService: WebSocketService;
  private callbacks: ((data: any) => void)[] = [];
  private lastData: any = {};
  private fallbackPolling: boolean = false;
  private pollInterval: number | null = null;

  constructor() {
    this.wsService = websocketService;
    this.setupWebSocketListeners();
  }

  private setupWebSocketListeners(): void {
    // Handle connection state changes
    this.wsService.onStateChange((state) => {
      if (state === 'authenticated') {
        this.fallbackPolling = false;
        this.stopPolling();
        this.subscribeToRealTimeData();
      } else if (state === 'disconnected' || state === 'error') {
        if (!this.fallbackPolling) {
          this.fallbackPolling = true;
          this.startPolling();
        }
      }
    });
  }

  async connect(token?: string): Promise<boolean> {
    try {
      const connected = await this.wsService.connect(token);
      if (connected && token) {
        await this.wsService.authenticate(token);
      }
      return connected;
    } catch (error) {
      console.error('Failed to connect to WebSocket:', error);
      this.fallbackPolling = true;
      this.startPolling();
      return false;
    }
  }

  subscribe(callback: (data: any) => void): () => void {
    this.callbacks.push(callback);

    // Send last known data immediately
    if (Object.keys(this.lastData).length > 0) {
      callback(this.lastData);
    }

    // Start appropriate data source
    if (this.wsService.isReady()) {
      this.subscribeToRealTimeData();
    } else if (!this.fallbackPolling) {
      this.fallbackPolling = true;
      this.startPolling();
    }

    return () => {
      this.callbacks = this.callbacks.filter(cb => cb !== callback);
      if (this.callbacks.length === 0) {
        this.cleanup();
      }
    };
  }

  private subscribeToRealTimeData(): void {
    // Subscribe to various real-time events
    this.wsService.subscribeToMetrics(this.handleMetricUpdate.bind(this));
    this.wsService.subscribeToDashboard(this.handleDashboardUpdate.bind(this));
    this.wsService.subscribeToSystemStatus(this.handleSystemUpdate.bind(this));
    this.wsService.subscribeToMemoryEvents(this.handleMemoryUpdate.bind(this));
    this.wsService.subscribeToSessionEvents(this.handleSessionUpdate.bind(this));
  }

  private handleMetricUpdate(data: any): void {
    if (!this.lastData.metrics) this.lastData.metrics = {};
    
    // Update specific metric
    this.lastData.metrics[data.metric_name] = {
      value: data.value,
      timestamp: data.timestamp,
      tags: data.tags
    };
    
    this.notifyCallbacks(this.lastData);
  }

  private handleDashboardUpdate(data: any): void {
    // Merge dashboard data
    this.lastData = { ...this.lastData, ...data.data };
    this.notifyCallbacks(this.lastData);
  }

  private handleSystemUpdate(data: any): void {
    if (!this.lastData.system) this.lastData.system = {};
    this.lastData.system = { ...this.lastData.system, ...data.status };
    this.notifyCallbacks(this.lastData);
  }

  private handleMemoryUpdate(data: any): void {
    if (!this.lastData.memory_events) this.lastData.memory_events = [];
    this.lastData.memory_events.unshift(data);
    // Keep only last 10 events
    this.lastData.memory_events = this.lastData.memory_events.slice(0, 10);
    this.notifyCallbacks(this.lastData);
  }

  private handleSessionUpdate(data: any): void {
    if (!this.lastData.session_events) this.lastData.session_events = [];
    this.lastData.session_events.unshift(data);
    this.lastData.session_events = this.lastData.session_events.slice(0, 10);
    this.notifyCallbacks(this.lastData);
  }

  private startPolling(): void {
    if (this.pollInterval) return;

    // Import the existing polling logic
    this.pollInterval = window.setInterval(async () => {
      await this.fetchPollingData();
    }, 5000);

    // Initial fetch
    this.fetchPollingData();
  }

  private stopPolling(): void {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }

  private async fetchPollingData(): Promise<void> {
    // This would use the existing API polling logic
    // For now, just a placeholder
    try {
      const response = await fetch('/api/realtime/dashboard');
      if (response.ok) {
        const data = await response.json();
        this.lastData = data;
        this.notifyCallbacks(this.lastData);
      }
    } catch (error) {
      console.error('Polling failed:', error);
    }
  }

  private notifyCallbacks(data: any): void {
    this.callbacks.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error('Error in real-time callback:', error);
      }
    });
  }

  private cleanup(): void {
    this.stopPolling();
    // WebSocket subscriptions are handled by the service itself
  }

  disconnect(): void {
    this.cleanup();
    this.wsService.disconnect();
  }
}

// Export enhanced service as default
export const enhancedRealTimeService = new EnhancedRealTimeService();

// Re-export for compatibility
export { enhancedRealTimeService as realtimeService };