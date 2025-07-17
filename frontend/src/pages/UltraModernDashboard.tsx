import { useState, useEffect } from 'react'
import {
  Box,
  Grid,
  Typography,
  useTheme,
  alpha,
  Card,
  Chip,
  LinearProgress,
  IconButton,
  Tooltip,
  Fab,
} from '@mui/material'
import {
  AutoAwesome,
  Bolt,
  Memory,
  Psychology,
  TrendingUp,
  Speed,
  Timeline,
  BubbleChart,
  Dashboard as DashboardIcon,
  ThreeSixty,
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'
import ParticlesBackground from '../components/ParticlesBackground'
import GlassCard from '../components/GlassCard'
import AnimatedChart from '../components/AnimatedChart'
import Network3D from '../components/Network3D'
import { api } from '../services/api'
import { realtimeService } from '../services/realtime'


export default function UltraModernDashboard() {
  const theme = useTheme()
  const [realTimeData, setRealTimeData] = useState<any>(null)
  const [aiMetrics, setAiMetrics] = useState<any[]>([])
  const [performanceData, setPerformanceData] = useState<any[]>([])
  const [view3D, setView3D] = useState(false)
  const [networkData, setNetworkData] = useState<any>({ nodes: [], edges: [] })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Subscribe to real-time updates
    const unsubscribe = realtimeService.subscribe((data) => {
      setRealTimeData(data)
    })

    // Fetch initial data
    fetchDashboardData()

    return () => {
      unsubscribe()
    }
  }, [])

  const fetchDashboardData = async () => {
    setLoading(true)
    try {
      // Fetch AI metrics
      await api.get('/api/ai-features/summary').catch(() => ({
        data: { features: {} }
      }))

      // Fetch time series data from API
      try {
        const metricsResponse = await api.get('/api/performance/metrics/hourly')
        setAiMetrics(metricsResponse.data.metrics || [])
      } catch (error) {
        console.error('Failed to fetch AI metrics:', error)
        setAiMetrics([])
      }

      // Fetch performance data from API
      try {
        const perfResponse = await api.get('/api/performance/radar')
        setPerformanceData(perfResponse.data.data || [])
      } catch (error) {
        console.error('Failed to fetch performance data:', error)
        setPerformanceData([])
      }

      // Fetch network topology from API
      try {
        const networkResponse = await api.get('/api/knowledge-graph/topology')
        setNetworkData(networkResponse.data || { nodes: [], edges: [] })
      } catch (error) {
        console.error('Failed to fetch network data:', error)
        setNetworkData({ nodes: [], edges: [] })
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
      },
    },
  }

  const itemVariants = {
    hidden: { y: 50, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: 'spring' as const,
        stiffness: 100,
        damping: 12,
      },
    },
  }

  const floatingVariants = {
    animate: {
      y: [0, -10, 0],
      transition: {
        duration: 3,
        repeat: Infinity,
        ease: 'easeInOut' as const,
      },
    },
  }

  return (
    <Box sx={{ position: 'relative', minHeight: '100vh', overflow: 'hidden' }}>
      <ParticlesBackground />
      
      {/* Animated gradient background */}
      <Box
        sx={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `radial-gradient(circle at 20% 50%, ${alpha(theme.palette.primary.main, 0.1)} 0%, transparent 50%),
                       radial-gradient(circle at 80% 80%, ${alpha(theme.palette.secondary.main, 0.1)} 0%, transparent 50%),
                       radial-gradient(circle at 40% 20%, ${alpha(theme.palette.info.main, 0.1)} 0%, transparent 50%)`,
          animation: 'gradient 15s ease infinite',
          '@keyframes gradient': {
            '0%, 100%': { transform: 'scale(1) rotate(0deg)' },
            '50%': { transform: 'scale(1.1) rotate(180deg)' },
          },
          zIndex: -1,
        }}
      />

      <motion.div
        initial="hidden"
        animate="visible"
        variants={containerVariants}
      >
        {/* Hero Section */}
        <Box sx={{ textAlign: 'center', py: 6 }}>
          <motion.div variants={itemVariants}>
            <Typography
              variant="h1"
              sx={{
                fontSize: { xs: '3rem', md: '5rem' },
                fontWeight: 900,
                background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main} 100%)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mb: 2,
                textShadow: `0 0 80px ${alpha(theme.palette.primary.main, 0.5)}`,
              }}
            >
              AI Intelligence Hub
            </Typography>
            <Typography
              variant="h5"
              color="text.secondary"
              sx={{ mb: 4, letterSpacing: 2 }}
            >
              NEXT-GENERATION COGNITIVE INFRASTRUCTURE
            </Typography>
          </motion.div>

          {/* Floating stats */}
          <motion.div variants={floatingVariants} animate="animate">
            <Box display="flex" justifyContent="center" gap={4} flexWrap="wrap">
              {[
                { icon: <Memory />, label: 'Memories', value: realTimeData?.memories?.total || '1,247' },
                { icon: <Psychology />, label: 'AI Requests', value: realTimeData?.ai?.requests_per_minute || '75/min' },
                { icon: <Speed />, label: 'Response Time', value: `${realTimeData?.performance?.response_time || 120}ms` },
                { icon: <TrendingUp />, label: 'Learning Rate', value: `${((realTimeData?.ai?.learning_rate || 0.85) * 100).toFixed(0)}%` },
              ].map((stat) => (
                <motion.div key={stat.label} variants={itemVariants}>
                  <GlassCard
                    gradient
                    sx={{
                      p: 3,
                      minWidth: 200,
                      position: 'relative',
                      overflow: 'hidden',
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: -2,
                        left: -2,
                        right: -2,
                        bottom: -2,
                        background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                        borderRadius: 'inherit',
                        opacity: 0,
                        transition: 'opacity 0.3s',
                        zIndex: -1,
                      },
                      '&:hover::before': {
                        opacity: 0.2,
                      },
                    }}
                  >
                    <Box display="flex" alignItems="center" gap={2}>
                      <Box
                        sx={{
                          p: 1.5,
                          borderRadius: 2,
                          background: alpha(theme.palette.primary.main, 0.1),
                          color: theme.palette.primary.main,
                        }}
                      >
                        {stat.icon}
                      </Box>
                      <Box>
                        <Typography variant="h4" fontWeight="bold">
                          {stat.value}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {stat.label}
                        </Typography>
                      </Box>
                    </Box>
                  </GlassCard>
                </motion.div>
              ))}
            </Box>
          </motion.div>
        </Box>

        {/* Main Content Grid */}
        <Grid container spacing={4} sx={{ px: 3, pb: 6 }}>
          {/* 3D Network Visualization */}
          <Grid item xs={12} lg={8}>
            <motion.div variants={itemVariants}>
              <GlassCard sx={{ height: 500, position: 'relative', overflow: 'hidden' }}>
                <Box sx={{ p: 3, height: '100%' }}>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Box display="flex" alignItems="center" gap={2}>
                      <BubbleChart sx={{ color: theme.palette.primary.main }} />
                      <Typography variant="h5" fontWeight="bold">
                        System Architecture
                      </Typography>
                      <Chip
                        icon={<AutoAwesome />}
                        label="LIVE 3D"
                        color="primary"
                        size="small"
                        sx={{ animation: 'pulse 2s infinite' }}
                      />
                    </Box>
                    <Tooltip title="Toggle 3D View">
                      <IconButton onClick={() => setView3D(!view3D)}>
                        <ThreeSixty />
                      </IconButton>
                    </Tooltip>
                  </Box>
                  
                  <AnimatePresence mode="wait">
                    {view3D ? (
                      <motion.div
                        key="3d"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        style={{ height: 'calc(100% - 60px)' }}
                      >
                        <Network3D {...networkData} />
                      </motion.div>
                    ) : (
                      <motion.div
                        key="chart"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        style={{ height: 'calc(100% - 60px)' }}
                      >
                        <AnimatedChart
                          type="area"
                          data={aiMetrics}
                          dataKeys={['requests', 'accuracy', 'patterns']}
                          height={380}
                        />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </Box>
              </GlassCard>
            </motion.div>
          </Grid>

          {/* Performance Radar */}
          <Grid item xs={12} lg={4}>
            <motion.div variants={itemVariants}>
              <GlassCard sx={{ height: 500 }}>
                <Box sx={{ p: 3, height: '100%' }}>
                  <Box display="flex" alignItems="center" gap={2} mb={2}>
                    <Timeline sx={{ color: theme.palette.secondary.main }} />
                    <Typography variant="h5" fontWeight="bold">
                      Performance Matrix
                    </Typography>
                  </Box>
                  
                  <AnimatedChart
                    type="radar"
                    data={performanceData}
                    dataKeys={['value']}
                    colors={[theme.palette.secondary.main]}
                    height={400}
                  />
                </Box>
              </GlassCard>
            </motion.div>
          </Grid>

          {/* AI Activity Stream */}
          <Grid item xs={12}>
            <motion.div variants={itemVariants}>
              <GlassCard>
                <Box sx={{ p: 3 }}>
                  <Box display="flex" alignItems="center" gap={2} mb={3}>
                    <Bolt sx={{ color: theme.palette.warning.main }} />
                    <Typography variant="h5" fontWeight="bold">
                      AI Intelligence Stream
                    </Typography>
                    <Box sx={{ ml: 'auto' }}>
                      <LinearProgress
                        variant="indeterminate"
                        sx={{
                          width: 100,
                          height: 4,
                          borderRadius: 2,
                          backgroundColor: alpha(theme.palette.primary.main, 0.1),
                          '& .MuiLinearProgress-bar': {
                            background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                          },
                        }}
                      />
                    </Box>
                  </Box>

                  <Grid container spacing={2}>
                    {[
                      { type: 'memory', message: 'New memory pattern detected and indexed', time: 'Just now', severity: 'success' },
                      { type: 'ai', message: 'Model accuracy improved by 2.3%', time: '2 min ago', severity: 'info' },
                      { type: 'security', message: 'Security scan completed - All systems secure', time: '5 min ago', severity: 'success' },
                      { type: 'performance', message: 'Cache optimization reduced latency by 15ms', time: '8 min ago', severity: 'warning' },
                    ].map((activity, index) => (
                      <Grid item xs={12} md={6} key={index}>
                        <motion.div
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.1 }}
                        >
                          <Card
                            sx={{
                              p: 2,
                              background: alpha(theme.palette.background.paper, 0.5),
                              backdropFilter: 'blur(10px)',
                              border: `1px solid ${alpha(theme.palette[activity.severity as 'success' | 'info' | 'warning'].main, 0.3)}`,
                              transition: 'all 0.3s',
                              '&:hover': {
                                transform: 'translateX(10px)',
                                borderColor: theme.palette[activity.severity as 'success' | 'info' | 'warning'].main,
                              },
                            }}
                          >
                            <Box display="flex" alignItems="center" gap={2}>
                              <Box
                                sx={{
                                  width: 8,
                                  height: 40,
                                  borderRadius: 1,
                                  background: theme.palette[activity.severity as 'success' | 'info' | 'warning'].main,
                                  boxShadow: `0 0 20px ${theme.palette[activity.severity as 'success' | 'info' | 'warning'].main}`,
                                }}
                              />
                              <Box flex={1}>
                                <Typography variant="body2">{activity.message}</Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {activity.type.toUpperCase()} • {activity.time}
                                </Typography>
                              </Box>
                            </Box>
                          </Card>
                        </motion.div>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              </GlassCard>
            </motion.div>
          </Grid>
        </Grid>
      </motion.div>

      {/* Floating Action Button */}
      <Fab
        color="primary"
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
          boxShadow: `0 4px 20px ${alpha(theme.palette.primary.main, 0.4)}`,
          '&:hover': {
            transform: 'scale(1.1)',
          },
        }}
      >
        <DashboardIcon />
      </Fab>

      {/* Loading overlay */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: alpha(theme.palette.background.default, 0.9),
              backdropFilter: 'blur(10px)',
              zIndex: 9999,
            }}
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
                    border: `2px solid ${theme.palette.primary.main}`,
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
                    color: theme.palette.primary.main,
                    animation: 'pulse 2s infinite',
                  }}
                />
              </Box>
              <Typography variant="h6" sx={{ mt: 3 }}>
                Initializing AI Systems...
              </Typography>
            </Box>
          </motion.div>
        )}
      </AnimatePresence>
    </Box>
  )
}