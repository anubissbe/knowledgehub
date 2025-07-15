import { useEffect, useState } from 'react'
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CircularProgress,
  Chip,
  LinearProgress,
} from '@mui/material'
import {
  TrendingUp,
  Memory,
  Psychology,
  Storage,
  Speed,
  CheckCircle,
  Warning,
  Error,
} from '@mui/icons-material'
import { api } from '../services/api'

interface SystemHealth {
  status: string
  services: {
    database: string
    redis: string
    ai_service: string
    weaviate?: string
    neo4j?: string
  }
}

interface SystemMetrics {
  total_memories: number
  total_sessions: number
  active_users: number
  ai_requests_today: number
  average_response_time: number
  cache_hit_rate: number
}

const StatusIcon = ({ status }: { status: string }) => {
  switch (status) {
    case 'operational':
      return <CheckCircle sx={{ color: 'success.main' }} />
    case 'degraded':
      return <Warning sx={{ color: 'warning.main' }} />
    default:
      return <Error sx={{ color: 'error.main' }} />
  }
}

export default function Dashboard() {
  const [health, setHealth] = useState<SystemHealth | null>(null)
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [healthRes, metricsRes] = await Promise.all([
          api.get('/health'),
          api.get('/api/metrics/dashboard'),
        ])
        setHealth(healthRes.data)
        setMetrics(metricsRes.data)
      } catch (error) {
        console.error('Error fetching dashboard data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="80vh"
      >
        <CircularProgress />
      </Box>
    )
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Dashboard
      </Typography>
      
      {/* System Health */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          System Health
        </Typography>
        <Grid container spacing={2}>
          {health?.services && Object.entries(health.services).map(([service, status]) => (
            <Grid item xs={12} sm={6} md={4} key={service}>
              <Box display="flex" alignItems="center" gap={1}>
                <StatusIcon status={status} />
                <Typography variant="body1" sx={{ textTransform: 'capitalize' }}>
                  {service.replace('_', ' ')}
                </Typography>
                <Chip
                  label={status}
                  size="small"
                  color={status === 'operational' ? 'success' : 'error'}
                />
              </Box>
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Metrics Cards */}
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={2}>
                <Memory color="primary" />
                <Typography color="text.secondary" variant="body2">
                  Total Memories
                </Typography>
              </Box>
              <Typography variant="h4">
                {metrics?.total_memories?.toLocaleString() || '0'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={2}>
                <Psychology color="primary" />
                <Typography color="text.secondary" variant="body2">
                  AI Requests Today
                </Typography>
              </Box>
              <Typography variant="h4">
                {metrics?.ai_requests_today?.toLocaleString() || '0'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={2}>
                <TrendingUp color="primary" />
                <Typography color="text.secondary" variant="body2">
                  Active Sessions
                </Typography>
              </Box>
              <Typography variant="h4">
                {metrics?.total_sessions?.toLocaleString() || '0'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={2}>
                <Speed color="primary" />
                <Typography color="text.secondary" variant="body2">
                  Avg Response Time
                </Typography>
              </Box>
              <Typography variant="h4">
                {metrics?.average_response_time?.toFixed(0) || '0'}ms
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={2}>
                <Storage color="primary" />
                <Typography color="text.secondary" variant="body2">
                  Cache Hit Rate
                </Typography>
              </Box>
              <Typography variant="h4">
                {((metrics?.cache_hit_rate || 0) * 100).toFixed(1)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={(metrics?.cache_hit_rate || 0) * 100}
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}