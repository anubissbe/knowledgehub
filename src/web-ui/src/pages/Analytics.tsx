import React, { useState, useEffect } from 'react'
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  Alert,
  IconButton,
  Tooltip,
  Stack,
  Divider
} from '@mui/material'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  Legend
} from 'recharts'
import { 
  Analytics as AnalyticsIcon, 
  Search as SearchIcon,
  Speed as SpeedIcon,
  TrendingUp as TrendingUpIcon,
  QueryStats as QueryStatsIcon,
  AccessTime as AccessTimeIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material'
import { api } from '@/services/api'

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8']

const Analytics: React.FC = () => {
  const [searchAnalytics, setSearchAnalytics] = useState<any>(null)
  const [realtimeMetrics, setRealtimeMetrics] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = async () => {
    try {
      setLoading(true)
      setError(null)
      
      const [analyticsData, realtimeData] = await Promise.all([
        api.getSearchAnalytics(),
        api.getRealtimeSearchMetrics()
      ])
      
      setSearchAnalytics(analyticsData)
      setRealtimeMetrics(realtimeData)
    } catch (err) {
      setError('Failed to load analytics data')
      console.error('Analytics fetch error:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
    
    // Set up automatic refresh for realtime metrics
    const interval = setInterval(async () => {
      try {
        const realtimeData = await api.getRealtimeSearchMetrics()
        setRealtimeMetrics(realtimeData)
      } catch (err) {
        console.error('Realtime metrics refresh error:', err)
      }
    }, 30000) // Refresh every 30 seconds

    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          <AnalyticsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Search Analytics
        </Typography>
        <LinearProgress />
      </Box>
    )
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
        <Typography variant="h4" gutterBottom>
          <AnalyticsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Search Analytics
        </Typography>
      </Box>
    )
  }

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          <AnalyticsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Search Analytics
        </Typography>
        <Tooltip title="Refresh data">
          <IconButton onClick={fetchData} size="small">
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Overview Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <SearchIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Today's Searches</Typography>
              </Box>
              <Typography variant="h4" color="primary">
                {searchAnalytics?.search_volume?.today || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Week: {searchAnalytics?.search_volume?.week || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <SpeedIcon color="success" sx={{ mr: 1 }} />
                <Typography variant="h6">Avg Response Time</Typography>
              </Box>
              <Typography variant="h4" color="success.main">
                {searchAnalytics?.performance?.avg_response_time_ms || 0}ms
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Real-time: {realtimeMetrics?.avg_response_time || 0}ms
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <TrendingUpIcon color="warning" sx={{ mr: 1 }} />
                <Typography variant="h6">Success Rate</Typography>
              </Box>
              <Typography variant="h4" color="warning.main">
                {searchAnalytics?.performance?.success_rate_pct || 0}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Queries with results
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <AccessTimeIcon color="info" sx={{ mr: 1 }} />
                <Typography variant="h6">Hourly Searches</Typography>
              </Box>
              <Typography variant="h4" color="info.main">
                {realtimeMetrics?.hourly_searches || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Last hour
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts Section */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Daily Performance Chart */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Daily Search Performance
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={searchAnalytics?.daily_performance || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <RechartsTooltip />
                  <Legend />
                  <Bar yAxisId="left" dataKey="searches" fill="#8884d8" name="Searches" />
                  <Line yAxisId="right" type="monotone" dataKey="avg_response_time" stroke="#82ca9d" name="Avg Response Time (ms)" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Search Types Distribution */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Search Types
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={searchAnalytics?.search_types || []}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ type, count }) => `${type}: ${count}`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="count"
                    nameKey="type"
                  >
                    {(searchAnalytics?.search_types || []).map((entry: any, index: number) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <RechartsTooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tables Section */}
      <Grid container spacing={3}>
        {/* Popular Queries */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <QueryStatsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Popular Queries
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Query</TableCell>
                      <TableCell align="right">Frequency</TableCell>
                      <TableCell align="right">Avg Results</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {(searchAnalytics?.popular_queries || []).map((query: any, index: number) => (
                      <TableRow key={index}>
                        <TableCell>
                          <Typography variant="body2" title={query.query}>
                            {query.query}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Chip label={query.frequency} size="small" color="primary" />
                        </TableCell>
                        <TableCell align="right">
                          {query.avg_results}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Queries */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <AccessTimeIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Recent Queries
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Query</TableCell>
                      <TableCell align="right">Response Time</TableCell>
                      <TableCell align="right">Results</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {(realtimeMetrics?.recent_queries || []).map((query: any, index: number) => (
                      <TableRow key={index}>
                        <TableCell>
                          <Typography variant="body2" title={query.query}>
                            {query.query}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Chip 
                            label={`${query.response_time}ms`} 
                            size="small" 
                            color={query.response_time > 200 ? 'error' : query.response_time > 100 ? 'warning' : 'success'}
                          />
                        </TableCell>
                        <TableCell align="right">
                          {query.results}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

export default Analytics