import { useState } from 'react'
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  CardHeader,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  IconButton,
  Tooltip,
  LinearProgress,
  Alert,
} from '@mui/material'
import {
  AutoFixHigh,
  BugReport,
  Timeline,
  Lightbulb,
  Code,
  Speed,
  TrendingUp,
  Info,
} from '@mui/icons-material'

const aiFeatures = [
  {
    title: 'Session Continuity',
    icon: <Timeline />,
    description: 'Seamless context preservation across sessions',
    endpoint: '/api/claude-auto/session',
    status: 'active',
  },
  {
    title: 'Mistake Learning',
    icon: <BugReport />,
    description: 'Learn from errors and improve over time',
    endpoint: '/api/mistake-learning',
    status: 'active',
  },
  {
    title: 'Proactive Assistant',
    icon: <Lightbulb />,
    description: 'Anticipate needs and suggest next actions',
    endpoint: '/api/proactive',
    status: 'active',
  },
  {
    title: 'Code Evolution',
    icon: <Code />,
    description: 'Track code changes and improvements',
    endpoint: '/api/code-evolution',
    status: 'active',
  },
  {
    title: 'Performance Metrics',
    icon: <Speed />,
    description: 'Monitor and optimize system performance',
    endpoint: '/api/performance',
    status: 'active',
  },
  {
    title: 'Decision Reasoning',
    icon: <AutoFixHigh />,
    description: 'Document decisions with rationale',
    endpoint: '/api/decisions',
    status: 'active',
  },
]

const recentActivity = [
  { time: '2 min ago', action: 'Session restored', type: 'session' },
  { time: '5 min ago', action: 'Error pattern learned', type: 'learning' },
  { time: '10 min ago', action: 'Performance optimization applied', type: 'performance' },
  { time: '15 min ago', action: 'Code improvement suggested', type: 'suggestion' },
  { time: '20 min ago', action: 'Decision recorded', type: 'decision' },
]

export default function AiIntelligence() {
  const [selectedFeature, setSelectedFeature] = useState<number | null>(null)

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        AI Intelligence System
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          The AI Intelligence System provides 8 core features that enhance Claude's coding assistance capabilities
          through learning, adaptation, and proactive support.
        </Typography>
      </Alert>

      {/* AI Features Grid */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {aiFeatures.map((feature, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Card
              sx={{
                cursor: 'pointer',
                transition: 'all 0.3s',
                border: selectedFeature === index ? 2 : 0,
                borderColor: 'primary.main',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: 4,
                },
              }}
              onClick={() => setSelectedFeature(index)}
            >
              <CardHeader
                avatar={feature.icon}
                title={feature.title}
                action={
                  <Chip
                    label={feature.status}
                    color="success"
                    size="small"
                  />
                }
              />
              <CardContent>
                <Typography variant="body2" color="text.secondary">
                  {feature.description}
                </Typography>
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                  Endpoint: {feature.endpoint}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Activity and Stats */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Recent AI Activity
            </Typography>
            <List>
              {recentActivity.map((activity, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    <TrendingUp color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary={activity.action}
                    secondary={activity.time}
                  />
                  <Chip
                    label={activity.type}
                    size="small"
                    variant="outlined"
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Learning Progress
            </Typography>
            <Box sx={{ mt: 2 }}>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2">Error Patterns Learned</Typography>
                <Typography variant="body2">156</Typography>
              </Box>
              <LinearProgress variant="determinate" value={78} sx={{ mb: 3 }} />
              
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2">Code Patterns Recognized</Typography>
                <Typography variant="body2">89</Typography>
              </Box>
              <LinearProgress variant="determinate" value={65} sx={{ mb: 3 }} />
              
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2">Performance Optimizations</Typography>
                <Typography variant="body2">42</Typography>
              </Box>
              <LinearProgress variant="determinate" value={42} />
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  )
}