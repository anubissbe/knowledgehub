/**
 * Retrieval Analytics Page
 * Deep analytics and insights for RAG retrieval performance
 */

import React, { useState, useEffect } from 'react'
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Tab,
  Tabs,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Chip,
  Alert,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material'
import {
  Analytics as AnalyticsIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  Search as SearchIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material'
import { motion } from 'framer-motion'

import PageWrapper from '../components/PageWrapper'
import GlassCard from '../components/GlassCard'
import AnimatedChart from '../components/AnimatedChart'
import { hybridRAGService, RAGAnalytics } from '../services/hybridRAGService'

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analytics-tabpanel-${index}`}
      aria-labelledby={`analytics-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  )
}

const RetrievalAnalytics: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0)
  const [timeframe, setTimeframe] = useState('24h')
  const [analytics, setAnalytics] = useState<RAGAnalytics | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadAnalytics()
  }, [timeframe])

  const loadAnalytics = async () => {
    setLoading(true)
    try {
      const data = await hybridRAGService.getAnalytics(timeframe)
      setAnalytics(data)
    } catch (error) {
      console.error('Failed to load analytics:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue)
  }

  return (
    <PageWrapper>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
          <Typography variant="h4" sx={{ fontWeight: 600 }}>
            📊 Retrieval Analytics
          </Typography>
          
          <Box display="flex" gap={2} alignItems="center">
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Timeframe</InputLabel>
              <Select
                value={timeframe}
                label="Timeframe"
                onChange={(e) => setTimeframe(e.target.value)}
              >
                <MenuItem value="1h">Last Hour</MenuItem>
                <MenuItem value="24h">Last 24h</MenuItem>
                <MenuItem value="7d">Last 7 Days</MenuItem>
                <MenuItem value="30d">Last 30 Days</MenuItem>
              </Select>
            </FormControl>
            
            <Button
              variant="outlined"
              onClick={loadAnalytics}
              startIcon={<RefreshIcon />}
              disabled={loading}
            >
              Refresh
            </Button>
          </Box>
        </Box>

        {loading && <LinearProgress sx={{ mb: 3 }} />}

        {analytics ? (
          <>
            {/* Key Metrics Overview */}
            <Grid container spacing={3} sx={{ mb: 3 }}>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      <SearchIcon color="primary" sx={{ mr: 1 }} />
                      <Typography variant="h6">Total Queries</Typography>
                    </Box>
                    <Typography variant="h3" color="primary.main">
                      {analytics.total_queries.toLocaleString()}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      <SpeedIcon color="success" sx={{ mr: 1 }} />
                      <Typography variant="h6">Avg Response</Typography>
                    </Box>
                    <Typography variant="h3" color="success.main">
                      {analytics.avg_response_time.toFixed(0)}ms
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      <TrendingUpIcon color="info" sx={{ mr: 1 }} />
                      <Typography variant="h6">Relevance Score</Typography>
                    </Box>
                    <Typography variant="h3" color="info.main">
                      {(analytics.quality_metrics.relevance_score * 100).toFixed(1)}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      <AnalyticsIcon color="warning" sx={{ mr: 1 }} />
                      <Typography variant="h6">MRR Score</Typography>
                    </Box>
                    <Typography variant="h3" color="warning.main">
                      {analytics.quality_metrics.mrr.toFixed(3)}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            <GlassCard>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={activeTab} onChange={handleTabChange}>
                  <Tab label="Performance Metrics" />
                  <Tab label="Quality Analysis" />
                  <Tab label="Retrieval Methods" />
                  <Tab label="Detailed Breakdown" />
                </Tabs>
              </Box>

              <TabPanel value={activeTab} index={0}>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Response Time by Method
                        </Typography>
                        <Box display="flex" flexDirection="column" gap={2}>
                          <Box>
                            <Box display="flex" justifyContent="space-between" mb={1}>
                              <Typography variant="body2">Vector Search</Typography>
                              <Typography variant="body2">
                                {analytics.performance_metrics.avg_vector_time.toFixed(2)}ms
                              </Typography>
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={(analytics.performance_metrics.avg_vector_time / analytics.avg_response_time) * 100}
                              color="primary"
                            />
                          </Box>
                          
                          <Box>
                            <Box display="flex" justifyContent="space-between" mb={1}>
                              <Typography variant="body2">Sparse Search</Typography>
                              <Typography variant="body2">
                                {analytics.performance_metrics.avg_sparse_time.toFixed(2)}ms
                              </Typography>
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={(analytics.performance_metrics.avg_sparse_time / analytics.avg_response_time) * 100}
                              color="secondary"
                            />
                          </Box>
                          
                          <Box>
                            <Box display="flex" justifyContent="space-between" mb={1}>
                              <Typography variant="body2">Graph Search</Typography>
                              <Typography variant="body2">
                                {analytics.performance_metrics.avg_graph_time.toFixed(2)}ms
                              </Typography>
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={(analytics.performance_metrics.avg_graph_time / analytics.avg_response_time) * 100}
                              color="info"
                            />
                          </Box>
                          
                          <Box>
                            <Box display="flex" justifyContent="space-between" mb={1}>
                              <Typography variant="body2">Reranking</Typography>
                              <Typography variant="body2">
                                {analytics.performance_metrics.avg_reranking_time.toFixed(2)}ms
                              </Typography>
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={(analytics.performance_metrics.avg_reranking_time / analytics.avg_response_time) * 100}
                              color="warning"
                            />
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Performance Trends
                        </Typography>
                        <Alert severity="info">
                          Performance trending data will be available with more historical data.
                          Continue using the system to build comprehensive analytics.
                        </Alert>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </TabPanel>

              <TabPanel value={activeTab} index={1}>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Quality Metrics
                        </Typography>
                        <Box display="flex" flexDirection="column" gap={3}>
                          <Box>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Relevance Score
                            </Typography>
                            <Box display="flex" alignItems="center" gap={2}>
                              <LinearProgress
                                variant="determinate"
                                value={analytics.quality_metrics.relevance_score * 100}
                                sx={{ flex: 1, height: 8 }}
                                color="success"
                              />
                              <Typography variant="h6" color="success.main">
                                {(analytics.quality_metrics.relevance_score * 100).toFixed(1)}%
                              </Typography>
                            </Box>
                          </Box>

                          <Box>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Precision@K
                            </Typography>
                            <Box display="flex" alignItems="center" gap={2}>
                              <LinearProgress
                                variant="determinate"
                                value={analytics.quality_metrics.precision_at_k * 100}
                                sx={{ flex: 1, height: 8 }}
                                color="info"
                              />
                              <Typography variant="h6" color="info.main">
                                {(analytics.quality_metrics.precision_at_k * 100).toFixed(1)}%
                              </Typography>
                            </Box>
                          </Box>

                          <Box>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Recall@K
                            </Typography>
                            <Box display="flex" alignItems="center" gap={2}>
                              <LinearProgress
                                variant="determinate"
                                value={analytics.quality_metrics.recall_at_k * 100}
                                sx={{ flex: 1, height: 8 }}
                                color="warning"
                              />
                              <Typography variant="h6" color="warning.main">
                                {(analytics.quality_metrics.recall_at_k * 100).toFixed(1)}%
                              </Typography>
                            </Box>
                          </Box>

                          <Box>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Mean Reciprocal Rank (MRR)
                            </Typography>
                            <Box display="flex" alignItems="center" gap={2}>
                              <LinearProgress
                                variant="determinate"
                                value={analytics.quality_metrics.mrr * 100}
                                sx={{ flex: 1, height: 8 }}
                                color="primary"
                              />
                              <Typography variant="h6" color="primary.main">
                                {analytics.quality_metrics.mrr.toFixed(3)}
                              </Typography>
                            </Box>
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Quality Insights
                        </Typography>
                        <Box display="flex" flexDirection="column" gap={2}>
                          {analytics.quality_metrics.relevance_score > 0.8 && (
                            <Alert severity="success">
                              Excellent relevance score! Your retrieval system is performing very well.
                            </Alert>
                          )}
                          
                          {analytics.quality_metrics.relevance_score < 0.6 && (
                            <Alert severity="warning">
                              Consider tuning your retrieval parameters or improving your knowledge base quality.
                            </Alert>
                          )}

                          {analytics.quality_metrics.mrr > 0.7 && (
                            <Alert severity="info">
                              High MRR score indicates that relevant results are consistently ranked high.
                            </Alert>
                          )}

                          <Typography variant="body2" color="text.secondary">
                            <strong>Tip:</strong> A good balance between precision and recall indicates
                            an optimal retrieval configuration. Consider hybrid approaches if one metric
                            is significantly lower than the other.
                          </Typography>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </TabPanel>

              <TabPanel value={activeTab} index={2}>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={8}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Retrieval Method Usage
                        </Typography>
                        <AnimatedChart
                          type="pie"
                          data={Object.entries(analytics.mode_distribution).map(([mode, count]) => ({
                            name: mode.replace('_', ' ').toUpperCase(),
                            value: count
                          }))}
                          height={400}
                        />
                      </CardContent>
                    </Card>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Method Performance
                        </Typography>
                        <Box display="flex" flexDirection="column" gap={2}>
                          {Object.entries(analytics.mode_distribution).map(([mode, count]) => {
                            const percentage = (count / analytics.total_queries) * 100
                            return (
                              <Box key={mode}>
                                <Box display="flex" justifyContent="space-between" mb={1}>
                                  <Typography variant="body2">
                                    {mode.replace('_', ' ').toUpperCase()}
                                  </Typography>
                                  <Typography variant="body2">
                                    {count} ({percentage.toFixed(1)}%)
                                  </Typography>
                                </Box>
                                <LinearProgress
                                  variant="determinate"
                                  value={percentage}
                                  sx={{ mb: 1 }}
                                />
                              </Box>
                            )
                          })}
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </TabPanel>

              <TabPanel value={activeTab} index={3}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Detailed Performance Breakdown
                    </Typography>
                    <TableContainer>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Metric</TableCell>
                            <TableCell align="right">Vector</TableCell>
                            <TableCell align="right">Sparse</TableCell>
                            <TableCell align="right">Graph</TableCell>
                            <TableCell align="right">Reranking</TableCell>
                            <TableCell align="right">Overall</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          <TableRow>
                            <TableCell>Average Time (ms)</TableCell>
                            <TableCell align="right">
                              {analytics.performance_metrics.avg_vector_time.toFixed(2)}
                            </TableCell>
                            <TableCell align="right">
                              {analytics.performance_metrics.avg_sparse_time.toFixed(2)}
                            </TableCell>
                            <TableCell align="right">
                              {analytics.performance_metrics.avg_graph_time.toFixed(2)}
                            </TableCell>
                            <TableCell align="right">
                              {analytics.performance_metrics.avg_reranking_time.toFixed(2)}
                            </TableCell>
                            <TableCell align="right">
                              <strong>{analytics.avg_response_time.toFixed(2)}</strong>
                            </TableCell>
                          </TableRow>
                          
                          <TableRow>
                            <TableCell>Performance Ratio</TableCell>
                            <TableCell align="right">
                              {((analytics.performance_metrics.avg_vector_time / analytics.avg_response_time) * 100).toFixed(1)}%
                            </TableCell>
                            <TableCell align="right">
                              {((analytics.performance_metrics.avg_sparse_time / analytics.avg_response_time) * 100).toFixed(1)}%
                            </TableCell>
                            <TableCell align="right">
                              {((analytics.performance_metrics.avg_graph_time / analytics.avg_response_time) * 100).toFixed(1)}%
                            </TableCell>
                            <TableCell align="right">
                              {((analytics.performance_metrics.avg_reranking_time / analytics.avg_response_time) * 100).toFixed(1)}%
                            </TableCell>
                            <TableCell align="right">
                              <strong>100%</strong>
                            </TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </TableContainer>

                    <Box mt={3}>
                      <Typography variant="h6" gutterBottom>
                        Quality Metrics Detail
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid item xs={12} sm={6} md={3}>
                          <Paper sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="h4" color="success.main">
                              {(analytics.quality_metrics.relevance_score * 100).toFixed(1)}%
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Relevance Score
                            </Typography>
                          </Paper>
                        </Grid>
                        
                        <Grid item xs={12} sm={6} md={3}>
                          <Paper sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="h4" color="info.main">
                              {(analytics.quality_metrics.precision_at_k * 100).toFixed(1)}%
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Precision@K
                            </Typography>
                          </Paper>
                        </Grid>
                        
                        <Grid item xs={12} sm={6} md={3}>
                          <Paper sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="h4" color="warning.main">
                              {(analytics.quality_metrics.recall_at_k * 100).toFixed(1)}%
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Recall@K
                            </Typography>
                          </Paper>
                        </Grid>
                        
                        <Grid item xs={12} sm={6} md={3}>
                          <Paper sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="h4" color="primary.main">
                              {analytics.quality_metrics.mrr.toFixed(3)}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              MRR
                            </Typography>
                          </Paper>
                        </Grid>
                      </Grid>
                    </Box>
                  </CardContent>
                </Card>
              </TabPanel>
            </GlassCard>
          </>
        ) : (
          <Box textAlign="center" py={8}>
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No analytics data available
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Start using the hybrid RAG system to generate analytics data
            </Typography>
            <Button
              variant="contained"
              onClick={loadAnalytics}
              startIcon={<AnalyticsIcon />}
            >
              Load Analytics
            </Button>
          </Box>
        )}
      </motion.div>
    </PageWrapper>
  )
}

export default RetrievalAnalytics