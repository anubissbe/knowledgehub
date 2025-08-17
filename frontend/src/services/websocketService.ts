// Advanced WebSocket Service for Real-Time Updates
// Supports reconnection, message queuing, and event handling

import { io, Socket } from 'socket.io-client'

export interface WebSocketMessage {
  type: string
  data: any
  timestamp: number
  id?: string
}

export interface WebSocketOptions {
  autoReconnect?: boolean
  reconnectInterval?: number
  maxReconnectAttempts?: number
  heartbeatInterval?: number
  timeout?: number
}

export interface RealtimeMetrics {
  memory: {
    total: number
    used: number
    available: number
    growth_rate: number
  }
  performance: {
    response_time: number
    throughput: number
    error_rate: number
    cpu_usage: number
  }
  ai: {
    requests_per_minute: number
    pattern_detections: number
    learning_accuracy: number
    model_updates: number
  }
  system: {
    active_sessions: number
    uptime: number
    health_score: number
    alerts: number
  }
}

export interface ActivityEvent {
  id: string
  type: 'memory_created' | 'search_performed' | 'ai_analysis' | 'error_detected' | 'system_update'
  user?: string
  message: string
  metadata?: Record<string, any>
  timestamp: string
  severity?: 'info' | 'warning' | 'error' | 'success'
}

type EventCallback<T = any> = (data: T) => void
type ConnectionCallback = () => void

class WebSocketService {
  private socket: Socket | null = null
  private eventListeners: Map<string, Set<EventCallback>> = new Map()
  private connectionListeners: Set<ConnectionCallback> = new Set()
  private disconnectionListeners: Set<ConnectionCallback> = new Set()
  
  private options: Required<WebSocketOptions>
  private reconnectAttempts = 0
  private isReconnecting = false
  private messageQueue: WebSocketMessage[] = []
  private heartbeatTimer: NodeJS.Timeout | null = null
  private lastPingTime = 0

  private baseUrl: string
  private isConnected = false

  constructor(options: WebSocketOptions = {}) {
    this.options = {
      autoReconnect: true,
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      timeout: 10000,
      ...options,
    }

    // Determine WebSocket URL based on current location
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const hostname = window.location.hostname
    
    // For LAN setup (192.168.1.x), connect to KnowledgeHub server
    if (hostname.startsWith('192.168.1.')) {
      this.baseUrl = `${protocol}//192.168.1.25:3000`
    } else if (hostname === 'localhost' || hostname === '127.0.0.1') {
      this.baseUrl = `${protocol}//localhost:3000`
    } else {
      this.baseUrl = `${protocol}//${hostname}:3000`
    }
  }

  // Connection Management
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {

        this.socket = io(this.baseUrl, {
          transports: ['websocket', 'polling'],
          timeout: this.options.timeout,
          forceNew: true,
          autoConnect: true,
        })

        this.socket.on('connect', () => {
          this.isConnected = true
          this.reconnectAttempts = 0
          this.isReconnecting = false
          
          // Process queued messages
          this.processMessageQueue()
          
          // Start heartbeat
          this.startHeartbeat()
          
          // Notify connection listeners
          this.connectionListeners.forEach(callback => callback())
          
          resolve()
        })

        this.socket.on('disconnect', (reason) => {
          this.isConnected = false
          this.stopHeartbeat()
          
          // Notify disconnection listeners
          this.disconnectionListeners.forEach(callback => callback())

          if (this.options.autoReconnect && reason !== 'io client disconnect') {
            this.scheduleReconnect()
          }
        })

        this.socket.on('connect_error', (error) => {
          this.isConnected = false
          
          if (this.options.autoReconnect) {
            this.scheduleReconnect()
          } else {
            reject(new Error(`WebSocket connection failed: ${error.message}`))
          }
        })

        // Real-time event handlers
        this.setupEventHandlers()

        // Connection timeout
        setTimeout(() => {
          if (!this.isConnected) {
            reject(new Error('WebSocket connection timeout'))
          }
        }, this.options.timeout)

      } catch (error) {
        reject(error)
      }
    })
  }

  disconnect(): void {
    
    this.stopHeartbeat()
    
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
    }
    
    this.isConnected = false
    this.eventListeners.clear()
    this.connectionListeners.clear()
    this.disconnectionListeners.clear()
  }

  private scheduleReconnect(): void {
    if (this.isReconnecting || this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      return
    }

    this.isReconnecting = true
    this.reconnectAttempts++

    const delay = Math.min(
      this.options.reconnectInterval * Math.pow(1.5, this.reconnectAttempts - 1),
      30000
    )


    setTimeout(() => {
      if (this.isReconnecting) {
        this.connect().catch((error) => {
        })
      }
    }, delay)
  }

  private setupEventHandlers(): void {
    if (!this.socket) return

    // Real-time metrics
    this.socket.on('metrics_update', (data: RealtimeMetrics) => {
      this.emit('metrics_update', data)
    })

    // Activity events
    this.socket.on('activity_event', (data: ActivityEvent) => {
      this.emit('activity_event', data)
    })

    // System alerts
    this.socket.on('system_alert', (data: any) => {
      this.emit('system_alert', data)
    })

    // AI insights
    this.socket.on('ai_insight', (data: any) => {
      this.emit('ai_insight', data)
    })

    // Memory updates
    this.socket.on('memory_update', (data: any) => {
      this.emit('memory_update', data)
    })

    // Performance updates
    this.socket.on('performance_update', (data: any) => {
      this.emit('performance_update', data)
    })

    // Heartbeat response
    this.socket.on('pong', () => {
      const latency = Date.now() - this.lastPingTime
      this.emit('latency_update', { latency })
    })

    // Generic message handler
    this.socket.on('message', (message: WebSocketMessage) => {
      this.emit(message.type, message.data)
    })
  }

  // Heartbeat System
  private startHeartbeat(): void {
    this.stopHeartbeat()
    
    this.heartbeatTimer = setInterval(() => {
      if (this.socket?.connected) {
        this.lastPingTime = Date.now()
        this.socket.emit('ping')
      }
    }, this.options.heartbeatInterval)
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  // Message Management
  send(type: string, data: any): void {
    const message: WebSocketMessage = {
      type,
      data,
      timestamp: Date.now(),
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    }

    if (this.isConnected && this.socket) {
      this.socket.emit('message', message)
    } else {
      // Queue message for later delivery
      this.messageQueue.push(message)
    }
  }

  private processMessageQueue(): void {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift()
      if (message && this.socket) {
        this.socket.emit('message', message)
      }
    }
  }

  // Event System
  on<T = any>(eventType: string, callback: EventCallback<T>): () => void {
    if (!this.eventListeners.has(eventType)) {
      this.eventListeners.set(eventType, new Set())
    }
    
    const listeners = this.eventListeners.get(eventType)!
    listeners.add(callback)

    // Return unsubscribe function
    return () => {
      listeners.delete(callback)
      if (listeners.size === 0) {
        this.eventListeners.delete(eventType)
      }
    }
  }

  off(eventType: string, callback?: EventCallback): void {
    if (!callback) {
      this.eventListeners.delete(eventType)
      return
    }

    const listeners = this.eventListeners.get(eventType)
    if (listeners) {
      listeners.delete(callback)
      if (listeners.size === 0) {
        this.eventListeners.delete(eventType)
      }
    }
  }

  private emit<T = any>(eventType: string, data: T): void {
    const listeners = this.eventListeners.get(eventType)
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(data)
        } catch (error) {
        }
      })
    }
  }

  // Connection Lifecycle
  onConnect(callback: ConnectionCallback): () => void {
    this.connectionListeners.add(callback)
    
    // Call immediately if already connected
    if (this.isConnected) {
      callback()
    }

    return () => {
      this.connectionListeners.delete(callback)
    }
  }

  onDisconnect(callback: ConnectionCallback): () => void {
    this.disconnectionListeners.add(callback)
    return () => {
      this.disconnectionListeners.delete(callback)
    }
  }

  // Utility Methods
  getConnectionState(): {
    connected: boolean
    reconnecting: boolean
    reconnectAttempts: number
    queuedMessages: number
  } {
    return {
      connected: this.isConnected,
      reconnecting: this.isReconnecting,
      reconnectAttempts: this.reconnectAttempts,
      queuedMessages: this.messageQueue.length,
    }
  }

  // Specialized Methods for KnowledgeHub
  subscribeToMetrics(): () => void {
    this.send('subscribe', { type: 'metrics' })
    return () => {
      this.send('unsubscribe', { type: 'metrics' })
    }
  }

  subscribeToActivities(): () => void {
    this.send('subscribe', { type: 'activities' })
    return () => {
      this.send('unsubscribe', { type: 'activities' })
    }
  }

  subscribeToAIInsights(): () => void {
    this.send('subscribe', { type: 'ai_insights' })
    return () => {
      this.send('unsubscribe', { type: 'ai_insights' })
    }
  }

  requestMetricsSnapshot(): void {
    this.send('request_snapshot', { type: 'metrics' })
  }

  requestActivitiesHistory(limit = 50): void {
    this.send('request_history', { 
      type: 'activities', 
      limit 
    })
  }
}

// Create singleton instance
export const webSocketService = new WebSocketService({
  autoReconnect: true,
  reconnectInterval: 5000,
  maxReconnectAttempts: 5,
  heartbeatInterval: 30000,
  timeout: 10000,
})

export default webSocketService