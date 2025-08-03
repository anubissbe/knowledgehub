import { useState, useEffect } from 'react'
import { Grid, Box, Typography, Card, CardContent, Paper } from '@mui/material'
import {
  AutoAwesome,
  Psychology,
  School,
  AccountTree,
  Code,
  Speed,
  Extension,
  Insights,
} from '@mui/icons-material'

const AI_FEATURES = [
  {
    id: 'session-continuity',
    icon: <AutoAwesome />,
    title: 'Session Continuity',
    description: 'Seamless context preservation across sessions',
    color: '#2196F3',
  },
  {
    id: 'mistake-learning',
    icon: <School />,
    title: 'Mistake Learning',
    description: 'Learn from errors to prevent repetition',
    color: '#FF00FF',
  },
  {
    id: 'proactive-assistant',
    icon: <Psychology />,
    title: 'Proactive Assistant',
    description: 'Anticipate needs and suggest next actions',
    color: '#00FF88',
  },
  {
    id: 'decision-reasoning',
    icon: <AccountTree />,
    title: 'Decision Reasoning',
    description: 'Track and explain all technical decisions',
    color: '#FFD700',
  },
  {
    id: 'code-evolution',
    icon: <Code />,
    title: 'Code Evolution',
    description: 'Track code changes and refactoring patterns',
    color: '#00FFFF',
  },
  {
    id: 'performance-optimization',
    icon: <Speed />,
    title: 'Performance Optimization',
    description: 'Continuous performance monitoring and tuning',
    color: '#8B5CF6',
  },
  {
    id: 'workflow-integration',
    icon: <Extension />,
    title: 'Workflow Integration',
    description: 'Seamless integration with development workflows',
    color: '#EC4899',
  },
  {
    id: 'pattern-recognition',
    icon: <Insights />,
    title: 'Pattern Recognition',
    description: 'Identify and apply coding patterns',
    color: '#FF3366',
  },
]

export default function AiIntelligenceSimple() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    console.log('AiIntelligenceSimple mounted')
    console.log('Features:', AI_FEATURES)
  }, [])

  if (!mounted) {
    return <div>Loading...</div>
  }

  return (
    <Box sx={{ p: 4 }}>
      <Typography variant="h3" gutterBottom>
        AI Intelligence Dashboard
      </Typography>
      
      <Typography variant="h6" color="text.secondary" gutterBottom>
        8 Advanced AI Features
      </Typography>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        {AI_FEATURES.map((feature) => (
          <Grid item xs={12} sm={6} md={4} lg={3} key={feature.id}>
            <Card
              sx={{
                height: '100%',
                backgroundColor: feature.color + '10',
                border: `2px solid ${feature.color}`,
                cursor: 'pointer',
                '&:hover': {
                  transform: 'scale(1.02)',
                  transition: 'transform 0.2s',
                },
              }}
            >
              <CardContent>
                <Box sx={{ color: feature.color, mb: 2 }}>
                  {feature.icon}
                </Box>
                <Typography variant="h6" gutterBottom>
                  {feature.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  )
}