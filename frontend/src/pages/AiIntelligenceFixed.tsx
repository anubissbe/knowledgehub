import { useState, useEffect } from 'react'
import { Grid, Box, Typography, Chip, Tab, Tabs, alpha, IconButton, CircularProgress } from '@mui/material'
import {
  AutoAwesome,
  Psychology,
  Timeline,
  Speed,
  BubbleChart,
  Code,
  TrendingUp,
  School,
  Extension,
  Insights,
  AccountTree,
  CloudSync,
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'
import PageContainer from '../components/ultra/PageContainer'
import UltraHeader from '../components/ultra/UltraHeader'
import FeatureCard from '../components/ultra/FeatureCard'
import GlassCard from '../components/GlassCard'
import AnimatedChart from '../components/AnimatedChart'
import { api } from '../services/api'

const AI_FEATURES_STATIC = [
  {
    id: 'session-continuity',
    icon: <CloudSync fontSize="large" />,
    title: 'Session Continuity',
    description: 'Seamless context preservation across sessions',
    color: '#2196F3',
    progress: 95,
    status: 'active',
    stats: [
      { label: 'Sessions', value: '128' },
      { label: 'Memories', value: '4.3K' },
    ],
  },
  {
    id: 'mistake-learning',
    icon: <School fontSize="large" />,
    title: 'Mistake Learning',
    description: 'Learn from errors to prevent repetition',
    color: '#FF00FF',
    progress: 88,
    status: 'active',
    stats: [
      { label: 'Patterns', value: '156' },
      { label: 'Lessons', value: '89' },
    ],
  },
  {
    id: 'proactive-assistant',
    icon: <Psychology fontSize="large" />,
    title: 'Proactive Assistant',
    description: 'Anticipate needs and suggest next actions',
    color: '#00FF88',
    progress: 92,
    status: 'active',
    stats: [
      { label: 'Predictions', value: '342' },
      { label: 'Accuracy', value: '87%' },
    ],
  },
  {
    id: 'decision-reasoning',
    icon: <AccountTree fontSize="large" />,
    title: 'Decision Reasoning',
    description: 'Track and explain all technical decisions',
    color: '#FFD700',
    progress: 85,
    status: 'active',
    stats: [
      { label: 'Decisions', value: '234' },
      { label: 'Categories', value: '12' },
    ],
  },
  {
    id: 'code-evolution',
    icon: <Code fontSize="large" />,
    title: 'Code Evolution',
    description: 'Track code changes and refactoring patterns',
    color: '#00FFFF',
    progress: 90,
    status: 'active',
    stats: [
      { label: 'Changes', value: '567' },
      { label: 'Patterns', value: '45' },
    ],
  },
  {
    id: 'performance-optimization',
    icon: <Speed fontSize="large" />,
    title: 'Performance Optimization',
    description: 'Continuous performance monitoring and tuning',
    color: '#8B5CF6',
    progress: 93,
    status: 'active',
    stats: [
      { label: 'Metrics', value: '1.2K' },
      { label: 'Optimized', value: '89%' },
    ],
  },
  {
    id: 'workflow-integration',
    icon: <Extension fontSize="large" />,
    title: 'Workflow Integration',
    description: 'Seamless integration with development workflows',
    color: '#EC4899',
    progress: 87,
    status: 'active',
    stats: [
      { label: 'Workflows', value: '23' },
      { label: 'Automated', value: '85%' },
    ],
  },
  {
    id: 'pattern-recognition',
    icon: <Insights fontSize="large" />,
    title: 'Pattern Recognition',
    description: 'Identify and apply coding patterns',
    color: '#FF3366',
    progress: 91,
    status: 'active',
    stats: [
      { label: 'Patterns', value: '456' },
      { label: 'Applied', value: '234' },
    ],
  },
]

export default function AiIntelligenceFixed() {
  const [selectedTab, setSelectedTab] = useState(0)
  const [features, setFeatures] = useState(AI_FEATURES_STATIC)
  const [chartData, setChartData] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [featureDetails, setFeatureDetails] = useState<any>(null)

  useEffect(() => {
    // Generate chart data
    const now = new Date()
    const data = Array.from({ length: 30 }, (_, i) => {
      const date = new Date(now.getTime() - (29 - i) * 24 * 60 * 60 * 1000)
      return {
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        accuracy: 85 + Math.random() * 10,
        predictions: Math.floor(Math.random() * 100) + 200,
        errors: Math.floor(Math.random() * 20) + 5,
        performance: 90 + Math.random() * 8,
      }
    })
    setChartData(data)
  }, [])

  const handleFeatureClick = async (feature: any) => {
    console.log('Feature clicked:', feature.id)
    setFeatureDetails(feature)
  }

  const tabCategories = [
    { label: 'All Features', features: features },
    { label: 'Learning & Adaptation', features: features.filter(f => ['mistake-learning', 'pattern-recognition', 'code-evolution'].includes(f.id)) },
    { label: 'Automation', features: features.filter(f => ['proactive-assistant', 'workflow-integration', 'performance-optimization'].includes(f.id)) },
    { label: 'Intelligence', features: features.filter(f => ['session-continuity', 'decision-reasoning'].includes(f.id)) },
  ]

  return (
    <PageContainer>
      <UltraHeader 
        title="AI Intelligence" 
        subtitle="ADVANCED COGNITIVE CAPABILITIES"
      />

      <Box sx={{ px: 3 }}>
        {/* Category Tabs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <GlassCard sx={{ mb: 4 }}>
            <Tabs
              value={selectedTab}
              onChange={(_, value) => setSelectedTab(value)}
              variant="fullWidth"
              sx={{
                '& .MuiTab-root': {
                  py: 2,
                  fontSize: '1rem',
                  fontWeight: 600,
                },
              }}
            >
              {tabCategories.map((category, index) => (
                <Tab
                  key={index}
                  label={
                    <Box display="flex" alignItems="center" gap={1}>
                      {category.label}
                      <Chip
                        label={category.features.length}
                        size="small"
                        sx={{
                          height: 20,
                          fontSize: '0.75rem',
                          backgroundColor: theme => 
                            selectedTab === index 
                              ? theme.palette.primary.main 
                              : alpha(theme.palette.text.primary, 0.1),
                        }}
                      />
                    </Box>
                  }
                />
              ))}
            </Tabs>
          </GlassCard>
        </motion.div>

        {/* Features Grid */}
        <AnimatePresence mode="wait">
          <motion.div
            key={selectedTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Grid container spacing={3} sx={{ mb: 4 }}>
              {tabCategories[selectedTab].features.map((feature, index) => (
                <Grid item xs={12} sm={6} lg={4} key={feature.id}>
                  <FeatureCard
                    {...feature}
                    delay={index * 0.1}
                    onClick={() => handleFeatureClick(feature)}
                  />
                </Grid>
              ))}
            </Grid>
          </motion.div>
        </AnimatePresence>

        {/* Analytics Section */}
        <Grid container spacing={3}>
          <Grid item xs={12} lg={8}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <GlassCard sx={{ height: 400 }}>
                <Box sx={{ p: 3, height: '100%' }}>
                  <Box display="flex" alignItems="center" gap={2} mb={2}>
                    <Timeline sx={{ color: 'primary.main' }} />
                    <Typography variant="h6" fontWeight="bold">
                      AI Performance Metrics
                    </Typography>
                    <Chip
                      icon={<AutoAwesome />}
                      label="30 Days"
                      size="small"
                      color="primary"
                      sx={{ ml: 'auto' }}
                    />
                  </Box>
                  
                  <AnimatedChart
                    type="line"
                    data={chartData}
                    dataKeys={['accuracy', 'predictions', 'performance']}
                    colors={['#2196F3', '#00FF88', '#FF00FF']}
                    height={320}
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
              <GlassCard sx={{ height: 400 }}>
                <Box sx={{ p: 3 }}>
                  <Box display="flex" alignItems="center" gap={2} mb={3}>
                    <BubbleChart sx={{ color: 'secondary.main' }} />
                    <Typography variant="h6" fontWeight="bold">
                      AI Insights
                    </Typography>
                  </Box>
                  
                  <Box display="flex" flexDirection="column" gap={2}>
                    {[
                      { label: 'Learning Rate', value: '94%', trend: '+2.3%', color: '#00FF88' },
                      { label: 'Prediction Accuracy', value: '87%', trend: '+5.1%', color: '#2196F3' },
                      { label: 'Error Prevention', value: '92%', trend: '+3.7%', color: '#FF00FF' },
                      { label: 'Automation Level', value: '85%', trend: '+8.2%', color: '#FFD700' },
                    ].map((insight, index) => (
                      <motion.div
                        key={insight.label}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.8 + index * 0.1 }}
                      >
                        <Box
                          sx={{
                            p: 2,
                            borderRadius: 2,
                            backgroundColor: theme => alpha(insight.color, 0.1),
                            border: `1px solid ${alpha(insight.color, 0.3)}`,
                          }}
                        >
                          <Box display="flex" justifyContent="space-between" alignItems="center">
                            <Typography variant="body2" color="text.secondary">
                              {insight.label}
                            </Typography>
                            <Chip
                              label={insight.trend}
                              size="small"
                              sx={{
                                backgroundColor: alpha(insight.color, 0.2),
                                color: insight.color,
                                fontWeight: 'bold',
                                fontSize: '0.75rem',
                              }}
                            />
                          </Box>
                          <Typography variant="h5" fontWeight="bold" color={insight.color}>
                            {insight.value}
                          </Typography>
                        </Box>
                      </motion.div>
                    ))}
                  </Box>
                </Box>
              </GlassCard>
            </motion.div>
          </Grid>
        </Grid>
      </Box>
    </PageContainer>
  )
}