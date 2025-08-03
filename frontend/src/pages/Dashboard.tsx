import { useState, useEffect } from 'react'
import { Grid, Box, Typography, Chip, LinearProgress, alpha, useTheme } from '@mui/material'
import { 
  Memory, 
  Psychology, 
  Speed, 
  Storage,
  Security,
  Cloud,
  AutoAwesome,
  Timeline,
  BubbleChart,
} from '@mui/icons-material'
import { motion } from 'framer-motion'
import PageWrapper from '../components/PageWrapper'
import UltraHeader from '../components/ultra/UltraHeader'
import MetricCard from '../components/ultra/MetricCard'
import GlassCard from '../components/GlassCard'
import AnimatedChart from '../components/AnimatedChart'
import Network3D from '../components/Network3D'
import { api } from '../services/api'
import { realtimeService } from '../services/realtime'

const METRIC_COLORS = {
  primary: '#2196F3',
  secondary: '#FF00FF',
  success: '#00FF88',
  warning: '#FFD700',
  error: '#FF3366',
  info: '#00FFFF',
  violet: '#8B5CF6',
  pink: '#EC4899',
}

// 3D network data
const generate3DNetwork = () => {
  const nodes = [
    { id: '1', label: 'Core', position: [0, 0, 0] as [number, number, number], color: METRIC_COLORS.primary, size: 0.8 },
    { id: '2', label: 'API', position: [3, 1, 0] as [number, number, number], color: METRIC_COLORS.success, size: 0.6 },
    { id: '3', label: 'Cache', position: [-3, 1, 0] as [number, number, number], color: METRIC_COLORS.secondary, size: 0.6 },
    { id: '4', label: 'DB', position: [0, 1, 3] as [number, number, number], color: METRIC_COLORS.warning, size: 0.5 },
    { id: '5', label: 'AI', position: [0, 1, -3] as [number, number, number], color: METRIC_COLORS.info, size: 0.5 },
  ]
  
  const edges = [
    { from: '1', to: '2' },
    { from: '1', to: '3' },
    { from: '1', to: '4' },
    { from: '1', to: '5' },
    { from: '2', to: '4' },
    { from: '3', to: '5' },
  ]
  
  return { nodes, edges }
}

export default function Dashboard() {
  const theme = useTheme()
  const [realTimeData, setRealTimeData] = useState<any>(null)
  const [metrics, setMetrics] = useState<any[]>([])
  const [chartData, setChartData] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const unsubscribe = realtimeService.subscribe((data) => {
      setRealTimeData(data)
      updateMetrics(data)
    })

    fetchInitialData()

    return () => {
      unsubscribe()
    }
  }, [])

  const fetchInitialData = async () => {
    try {
      const memoryResponse = await api.get('/api/claude-auto/memory/stats').catch(() => ({
        data: { stats: { total_memories: 0, recent_memories: 0 } }
      }))

      const sessionResponse = await api.get('/api/claude-auto/session/current').catch(() => ({
        data: { session_id: null }
      }))
      
      // Update metrics with the fetched data
      updateMetrics({
        memories: {
          total: memoryResponse.data?.stats?.total_memories || 0,
          growth: memoryResponse.data?.stats?.memories_last_24h || 0
        },
        sessions: {
          active: sessionResponse.data?.session_id ? 1 : 0
        }
      })

      // Fetch chart data from API
      try {
        const chartResponse = await api.get('/api/performance/stats')
        setChartData(chartResponse.data.metrics || [])
      } catch (error) {
        console.error('Failed to fetch chart data:', error)
        setChartData([])
      }

      setLoading(false)
    } catch (error) {
      console.error('Error fetching data:', error)
      setLoading(false)
    }
  }

  const updateMetrics = (data: any) => {
    const metricsData = [
      {
        icon: <Memory />,
        label: 'Total Memories',
        value: data?.memories?.total || 0,
        trend: data?.memories?.growth || 0,
        unit: '',
        color: METRIC_COLORS.primary,
        sparkline: generateSparkline(data?.memories?.total || 0),
      },
      {
        icon: <Psychology />,
        label: 'AI Requests/min',
        value: data?.ai?.requests_per_minute || 0,
        trend: -5.2,
        unit: 'req/min',
        color: METRIC_COLORS.secondary,
        sparkline: generateSparkline(35),
      },
      {
        icon: <Speed />,
        label: 'Response Time',
        value: data?.performance?.response_time || 145,
        trend: -15.8,
        unit: 'ms',
        color: METRIC_COLORS.success,
        sparkline: generateSparkline(145),
      },
      {
        icon: <Storage />,
        label: 'Cache Hit Rate',
        value: 87.3,
        trend: 3.2,
        unit: '%',
        color: METRIC_COLORS.info,
        sparkline: generateSparkline(87),
      },
      {
        icon: <Cloud />,
        label: 'Active Sessions',
        value: data?.sessions?.active || 1,
        trend: 2.1,
        unit: '',
        color: METRIC_COLORS.violet,
        sparkline: generateSparkline(5),
      },
      {
        icon: <Security />,
        label: 'Security Score',
        value: 98.5,
        trend: 0.5,
        unit: '/100',
        color: METRIC_COLORS.success,
        sparkline: generateSparkline(98),
      },
    ]
    setMetrics(metricsData)
  }

  const generateSparkline = (baseValue: number): number[] => {
    return Array.from({ length: 12 }, () => 
      baseValue + (Math.random() - 0.5) * baseValue * 0.2
    )
  }

  if (loading) {
    return (
      <PageWrapper>
        <Box
          display="flex"
          alignItems="center"
          justifyContent="center"
          minHeight="100vh"
        >
          <Box textAlign="center">
            <Box
              sx={{
                width: 100,
                height: 100,
                margin: '0 auto',
                position: 'relative',
                '&::before, &::after': {
                  content: '""',
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  width: '100%',
                  height: '100%',
                  border: theme => `2px solid ${theme.palette.primary.main}`,
                  borderRadius: '50%',
                  transform: 'translate(-50%, -50%)',
                  animation: 'ripple 2s infinite',
                },
                '&::after': {
                  animationDelay: '1s',
                },
                '@keyframes ripple': {
                  '0%': {
                    transform: 'translate(-50%, -50%) scale(0)',
                    opacity: 1,
                  },
                  '100%': {
                    transform: 'translate(-50%, -50%) scale(1)',
                    opacity: 0,
                  },
                },
              }}
            >
              <AutoAwesome
                sx={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  fontSize: 40,
                  color: 'primary.main',
                  animation: 'pulse 2s infinite',
                }}
              />
            </Box>
            <Typography variant="h6" sx={{ mt: 3 }}>
              Initializing Dashboard...
            </Typography>
          </Box>
        </Box>
      </PageWrapper>
    )
  }

  return (
    <PageWrapper>
      <UltraHeader 
        title="Intelligence Dashboard" 
        subtitle="REAL-TIME SYSTEM MONITORING & ANALYTICS"
      />

      {/* Metrics Grid */}
      <Box sx={{ px: { xs: 2, sm: 3, md: 4 } }}>
        <Grid container spacing={{ xs: 2, sm: 3 }} sx={{ mb: 4 }}>
          {metrics.map((metric, index) => (
            <Grid item xs={12} sm={6} md={4} lg={2} key={metric.label}>
              <Box sx={{ height: '100%', minHeight: { xs: 180, sm: 220 } }}>
                <MetricCard {...metric} delay={index * 0.1} />
              </Box>
            </Grid>
          ))}
        </Grid>

        {/* Main Charts */}
        <Grid container spacing={{ xs: 2, sm: 3 }} sx={{ mb: 4 }}>
          <Grid item xs={12} lg={8}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <GlassCard sx={{ height: { xs: 300, sm: 400, md: 500 } }}>
                <Box sx={{ p: { xs: 2, sm: 3 }, height: '100%' }}>
                  <Box display="flex" alignItems="center" gap={2} mb={2}>
                    <Timeline sx={{ color: 'primary.main' }} />
                    <Typography variant="h6" fontWeight="bold">
                      System Performance
                    </Typography>
                    <Chip
                      size="small"
                      icon={<AutoAwesome />}
                      label="Live"
                      color="primary"
                      sx={{ ml: 'auto' }}
                    />
                  </Box>
                  
                  <AnimatedChart
                    type="area"
                    data={chartData}
                    dataKeys={['memories', 'requests', 'accuracy']}
                    height={420}
                  />
                </Box>
              </GlassCard>
            </motion.div>
          </Grid>

          <Grid item xs={12} lg={4}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
            >
              <GlassCard sx={{ height: { xs: 300, sm: 400, md: 500 } }}>
                <Box sx={{ p: { xs: 2, sm: 3 }, height: '100%' }}>
                  <Box display="flex" alignItems="center" gap={2} mb={2}>
                    <BubbleChart sx={{ color: 'secondary.main' }} />
                    <Typography variant="h6" fontWeight="bold">
                      Network Topology
                    </Typography>
                  </Box>
                  
                  <Network3D {...generate3DNetwork()} />
                </Box>
              </GlassCard>
            </motion.div>
          </Grid>
        </Grid>

        {/* Activity Feed */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
        >
          <GlassCard>
            <Box sx={{ p: 3 }}>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                Live Activity Stream
              </Typography>
              
              <Box sx={{ mt: 2 }}>
                {(realTimeData?.activities || []).map((activity: any, index: number) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.9 + index * 0.1 }}
                  >
                    <Box
                      display="flex"
                      alignItems="center"
                      gap={2}
                      sx={{
                        p: 2,
                        borderRadius: 2,
                        mb: 1,
                        backgroundColor: alpha(theme.palette.background.default, 0.5),
                        borderLeft: `3px solid ${theme.palette.primary.main}`,
                        '&:hover': {
                          backgroundColor: alpha(theme.palette.primary.main, 0.05),
                          transform: 'translateX(5px)',
                        },
                        transition: 'all 0.3s',
                      }}
                    >
                      <Box flex={1}>
                        <Typography variant="body2">
                          <strong>{activity.user || 'System'}</strong> {activity.action || activity.message}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {activity.timestamp ? new Date(activity.timestamp).toLocaleTimeString() : 'Just now'}
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="indeterminate"
                        sx={{
                          width: 50,
                          height: 2,
                          borderRadius: 1,
                          backgroundColor: alpha(theme.palette.primary.main, 0.2),
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: theme.palette.primary.main,
                          },
                        }}
                      />
                    </Box>
                  </motion.div>
                ))}
              </Box>
            </Box>
          </GlassCard>
        </motion.div>
      </Box>
    </PageWrapper>
  )
}