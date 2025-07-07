import React from 'react'
import {
  Card,
  CardContent,
  Typography,
  Grid,
  LinearProgress,
  Box,
  Chip,
  Divider
} from '@mui/material'
import {
  TrendingUp,
  Speed,
  Memory,
  Storage
} from '@mui/icons-material'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/services/api'

interface MetricCardProps {
  title: string
  value: number
  unit: string
  max?: number
  color: 'primary' | 'secondary' | 'success' | 'warning' | 'error'
  trend?: number
  icon: React.ReactNode
}

function MetricCard({ title, value, unit, max = 100, color, trend, icon }: MetricCardProps) {
  const percentage = max > 0 ? (value / max) * 100 : 0
  
  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Box sx={{ color: `${color}.main`, mr: 1 }}>
            {icon}
          </Box>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            {title}
          </Typography>
          {trend !== undefined && (
            <Chip
              size="small"
              icon={<TrendingUp />}
              label={`${trend > 0 ? '+' : ''}${trend}%`}
              color={trend > 0 ? 'success' : trend < 0 ? 'error' : 'default'}
              variant="outlined"
            />
          )}
        </Box>
        
        <Typography variant="h4" component="div" sx={{ mb: 1 }}>
          {value.toLocaleString()} {unit}
        </Typography>
        
        <LinearProgress
          variant="determinate"
          value={Math.min(percentage, 100)}
          color={color}
          sx={{ height: 8, borderRadius: 4 }}
        />
        
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          {percentage.toFixed(1)}% of capacity
        </Typography>
      </CardContent>
    </Card>
  )
}

export default function PerformanceMetrics() {
  const { data: metrics, isLoading } = useQuery({
    queryKey: ['performance-metrics'],
    queryFn: api.getPerformanceMetrics,
    refetchInterval: 5000, // Refresh every 5 seconds
  })

  if (isLoading || !metrics) {
    return (
      <Card>
        <CardContent>
          <Typography>Loading performance metrics...</Typography>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Performance Metrics
        </Typography>
        
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <MetricCard
              title="Memory Usage"
              value={metrics.memory_used_mb || 0}
              unit="MB"
              max={metrics.memory_total_mb || 8192}
              color="warning"
              trend={metrics.memory_trend}
              icon={<Memory />}
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <MetricCard
              title="Storage Usage"
              value={metrics.storage_used_gb || 0}
              unit="GB"
              max={metrics.storage_total_gb || 100}
              color="primary"
              trend={metrics.storage_trend}
              icon={<Storage />}
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <MetricCard
              title="Avg Response Time"
              value={metrics.avg_response_time_ms || 0}
              unit="ms"
              max={1000}
              color="success"
              trend={metrics.response_time_trend}
              icon={<Speed />}
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <MetricCard
              title="Requests/Hour"
              value={metrics.requests_per_hour || 0}
              unit="req/h"
              max={10000}
              color="secondary"
              trend={metrics.requests_trend}
              icon={<TrendingUp />}
            />
          </Grid>
        </Grid>

        <Divider sx={{ my: 2 }} />
        
        <Typography variant="subtitle2" gutterBottom>
          System Status
        </Typography>
        
        <Grid container spacing={1}>
          <Grid item>
            <Chip
              label="API Gateway"
              color={metrics.api_status === 'healthy' ? 'success' : 'error'}
              variant="outlined"
              size="small"
            />
          </Grid>
          <Grid item>
            <Chip
              label="Database"
              color={metrics.database_status === 'healthy' ? 'success' : 'error'}
              variant="outlined"
              size="small"
            />
          </Grid>
          <Grid item>
            <Chip
              label="Vector Store"
              color={metrics.weaviate_status === 'healthy' ? 'success' : 'error'}
              variant="outlined"
              size="small"
            />
          </Grid>
          <Grid item>
            <Chip
              label="Cache"
              color={metrics.redis_status === 'healthy' ? 'success' : 'error'}
              variant="outlined"
              size="small"
            />
          </Grid>
          <Grid item>
            <Chip
              label="AI Service"
              color={metrics.ai_service_status === 'healthy' ? 'success' : 'error'}
              variant="outlined"
              size="small"
            />
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  )
}