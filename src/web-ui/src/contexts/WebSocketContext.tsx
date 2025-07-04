import { createContext, useContext, useEffect, useRef, ReactNode } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { api } from '@/services/api'

interface WebSocketContextType {
  isConnected: boolean
}

const WebSocketContext = createContext<WebSocketContextType>({ isConnected: false })

export const useWebSocket = () => useContext(WebSocketContext)

interface WebSocketProviderProps {
  children: ReactNode
}

export function WebSocketProvider({ children }: WebSocketProviderProps) {
  const wsRef = useRef<WebSocket | null>(null)
  const queryClient = useQueryClient()
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()

  const connect = () => {
    try {
      const ws = api.connectWebSocket()
      wsRef.current = ws

      ws.onopen = () => {
        console.log('WebSocket connected')
        // Clear any reconnect timeout
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current)
        }
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          console.log('WebSocket message:', data)

          // Handle different message types
          switch (data.type) {
            case 'job_completed':
            case 'job_failed':
            case 'job_cancelled':
              // Invalidate queries when job status changes
              queryClient.invalidateQueries({ queryKey: ['jobs'] })
              queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] })
              queryClient.invalidateQueries({ queryKey: ['sources'] })
              break

            case 'source_updated':
              // Invalidate source-related queries
              queryClient.invalidateQueries({ queryKey: ['sources'] })
              queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] })
              break

            case 'stats_updated':
              // Invalidate dashboard stats
              queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] })
              break

            default:
              console.log('Unknown WebSocket message type:', data.type)
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        wsRef.current = null
        
        // Attempt to reconnect after 5 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Attempting to reconnect WebSocket...')
          connect()
        }, 5000)
      }
    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
      
      // Retry connection after 5 seconds
      reconnectTimeoutRef.current = setTimeout(() => {
        connect()
      }, 5000)
    }
  }

  useEffect(() => {
    connect()

    // Cleanup on unmount
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  return (
    <WebSocketContext.Provider value={{ isConnected: !!wsRef.current }}>
      {children}
    </WebSocketContext.Provider>
  )
}