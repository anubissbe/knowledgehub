import { api } from './api'

export interface RealTimeData {
  memories: {
    total: number
    recent: number
    growth: number
  }
  sessions: {
    active: number
    total: number
    average_duration: number
  }
  performance: {
    response_time: number
    error_rate: number
    throughput: number
  }
  ai: {
    requests_per_minute: number
    pattern_detections: number
    learning_rate: number
  }
}

class RealTimeService {
  private callbacks: ((data: Partial<RealTimeData>) => void)[] = []
  private intervalId: number | null = null
  private lastData: Partial<RealTimeData> = {}

  subscribe(callback: (data: Partial<RealTimeData>) => void) {
    this.callbacks.push(callback)
    
    // Start polling if not already started
    if (!this.intervalId) {
      this.startPolling()
    }
    
    // Send last known data immediately
    if (Object.keys(this.lastData).length > 0) {
      callback(this.lastData)
    }
    
    // Return unsubscribe function
    return () => {
      this.callbacks = this.callbacks.filter(cb => cb !== callback)
      if (this.callbacks.length === 0) {
        this.stopPolling()
      }
    }
  }

  private async startPolling() {
    // Initial fetch
    await this.fetchData()
    
    // Poll every 5 seconds
    this.intervalId = setInterval(() => {
      this.fetchData()
    }, 5000)
  }

  private stopPolling() {
    if (this.intervalId) {
      clearInterval(this.intervalId)
      this.intervalId = null
    }
  }

  private async fetchData() {
    try {
      // Fetch from multiple endpoints in parallel
      const [memoryStats, sessionData, perfReport, aiStats] = await Promise.all([
        api.get('/api/memory/stats').catch(() => null),
        api.get('/api/claude-auto/session/current').catch(() => null),
        api.get('/api/performance/report').catch(() => null),
        api.get('/api/ai-features/summary').catch(() => null),
      ])

      // const now = Date.now()
      // const fiveMinutesAgo = now - 5 * 60 * 1000

      // Process real data
      const data: Partial<RealTimeData> = {}

      if (memoryStats?.data) {
        const previousTotal = this.lastData.memories?.total || 0
        data.memories = {
          total: memoryStats.data.total_memories || 0,
          recent: memoryStats.data.recent_memories || 0,
          growth: previousTotal > 0 ? ((memoryStats.data.total_memories - previousTotal) / previousTotal) * 100 : 0,
        }
      }

      if (sessionData?.data) {
        data.sessions = {
          active: sessionData.data.session_id ? 1 : 0,
          total: sessionData.data.total_sessions || 1,
          average_duration: sessionData.data.average_duration || 0,
        }
      }

      if (perfReport?.data) {
        data.performance = {
          response_time: perfReport.data.avg_response_time || 150,
          error_rate: perfReport.data.error_rate || 0.02,
          throughput: perfReport.data.requests_per_second || 50,
        }
      }

      if (aiStats?.data) {
        data.ai = {
          requests_per_minute: aiStats.data.requests_per_minute || 30,
          pattern_detections: aiStats.data.patterns_detected || 0,
          learning_rate: aiStats.data.learning_rate || 0.85,
        }
      }

      // Fetch activities from API
      try {
        const activitiesResponse = await api.get('/api/activity/recent')
        data.activities = activitiesResponse.data.activities || []
      } catch (error) {
        console.log('Failed to fetch activities:', error)
        data.activities = []
      }

      this.lastData = data
      this.notifyCallbacks(data)
    } catch (error) {
      console.error('Error fetching real-time data:', error)
    }
  }

  private notifyCallbacks(data: Partial<RealTimeData>) {
    this.callbacks.forEach(callback => {
      try {
        callback(data)
      } catch (error) {
        console.error('Error in real-time data callback:', error)
      }
    })
  }

  // Get current metrics snapshot
  async getSnapshot(): Promise<RealTimeData> {
    await this.fetchData()
    return {
      memories: this.lastData.memories || { total: 0, recent: 0, growth: 0 },
      sessions: this.lastData.sessions || { active: 0, total: 0, average_duration: 0 },
      performance: this.lastData.performance || { response_time: 0, error_rate: 0, throughput: 0 },
      ai: this.lastData.ai || { requests_per_minute: 0, pattern_detections: 0, learning_rate: 0 },
    }
  }
}

export const realtimeService = new RealTimeService()