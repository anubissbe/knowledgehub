/**
 * Hybrid RAG Dashboard
 * Central interface for enhanced RAG capabilities with real-time monitoring
 */

import React, { useState, useEffect, useRef } from 'react'
import {
  Box,
  Grid,
  Paper,
  Typography,
  TextField,
  Button,
  Chip,
  Card,
  CardContent,
  CardActions,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tab,
  Tabs,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel,
  LinearProgress,
  Alert,
  IconButton,
  Tooltip,
  Fade,
  Collapse
} from '@mui/material'
import {
  ExpandMore as ExpandMoreIcon,
  Search as SearchIcon,
  Compare as CompareIcon,
  Analytics as AnalyticsIcon,
  Tune as TuneIcon,
  Speed as SpeedIcon,
  Psychology as PsychologyIcon,
  Feedback as FeedbackIcon,
  Refresh as RefreshIcon,
  History as HistoryIcon,
  TrendingUp as TrendingUpIcon
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'

import PageWrapper from '../components/PageWrapper'
import GlassCard from '../components/GlassCard'
import AnimatedChart from '../components/AnimatedChart'
import {
  hybridRAGService,
  RetrievalMode,
  HybridRAGRequest,
  HybridRAGResponse,
  RAGAnalytics,
  ComparisonResult
} from '../services/hybridRAGService'

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
      id={`rag-tabpanel-${index}`}
      aria-labelledby={`rag-tab-${index}`}
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

const HybridRAGDashboard: React.FC = () => {
  // State
  const [activeTab, setActiveTab] = useState(0)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<HybridRAGResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [analytics, setAnalytics] = useState<RAGAnalytics | null>(null)
  const [comparison, setComparison] = useState<ComparisonResult | null>(null)
  const [realtimeMetrics, setRealtimeMetrics] = useState<any>(null)
  
  // Configuration
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>(RetrievalMode.HYBRID_ALL)
  const [topK, setTopK] = useState(10)
  const [includeReasoning, setIncludeReasoning] = useState(true)
  const [enableReranking, setEnableReranking] = useState(true)
  
  // UI State
  const [expandedResults, setExpandedResults] = useState<string[]>([])
  const [showAnalytics, setShowAnalytics] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(false)
  
  const queryInputRef = useRef<HTMLInputElement>(null)
  const sessionId = useRef<string>(`session_${Date.now()}`)

  // Effects
  useEffect(() => {
    loadAnalytics()
    loadRealtimeMetrics()
  }, [])

  useEffect(() => {
    let interval: NodeJS.Timeout
    if (autoRefresh) {
      interval = setInterval(() => {
        loadRealtimeMetrics()
        if (showAnalytics) {
          loadAnalytics()
        }
      }, 5000)
    }
    return () => clearInterval(interval)
  }, [autoRefresh, showAnalytics])

  // API Calls
  const loadAnalytics = async () => {
    try {
      const data = await hybridRAGService.getAnalytics('24h')
      setAnalytics(data)
    } catch (error) {
      console.error('Failed to load analytics:', error)
    }
  }

  const loadRealtimeMetrics = async () => {
    try {
      const data = await hybridRAGService.getRealtimeMetrics()
      setRealtimeMetrics(data)
    } catch (error) {
      console.error('Failed to load realtime metrics:', error)
    }
  }

  const handleQuery = async () => {
    if (!query.trim()) return

    setLoading(true)
    try {
      const request: HybridRAGRequest = {
        query: query.trim(),
        retrieval_mode: retrievalMode,
        top_k: topK,
        include_reasoning: includeReasoning,
        enable_reranking: enableReranking,
        session_id: sessionId.current
      }

      const response = await hybridRAGService.query(request)
      setResults(response)
      setExpandedResults([])
    } catch (error) {
      console.error('Query failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleCompareRetrievalModes = async () => {
    if (!query.trim()) return

    setLoading(true)
    try {
      const modes = [
        RetrievalMode.VECTOR_ONLY,
        RetrievalMode.SPARSE_ONLY,
        RetrievalMode.GRAPH_ONLY,
        RetrievalMode.HYBRID_ALL
      ]
      
      const comparisonResult = await hybridRAGService.compareRetrievalModes(
        query.trim(),
        modes,
        topK
      )
      setComparison(comparisonResult)
      setActiveTab(2)
    } catch (error) {
      console.error('Comparison failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const toggleResultExpansion = (resultId: string) => {
    setExpandedResults(prev =>
      prev.includes(resultId)
        ? prev.filter(id => id !== resultId)
        : [...prev, resultId]
    )
  }

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue)
  }

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleQuery()
    }
  }

  return (
    <PageWrapper>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Typography variant="h4" gutterBottom sx={{ mb: 4, fontWeight: 600 }}>
          🔬 Hybrid RAG Dashboard
        </Typography>

        {/* Real-time Metrics Bar */}
        {realtimeMetrics && (
          <Fade in={true}>
            <Paper sx={{ p: 2, mb: 3, bgcolor: 'background.paper' }}>
              <Grid container spacing={3} alignItems="center">
                <Grid item xs={12} sm={6} md={2}>
                  <Box display="flex" alignItems="center">
                    <SpeedIcon color="primary" sx={{ mr: 1 }} />
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Avg Response Time
                      </Typography>
                      <Typography variant="h6">
                        {realtimeMetrics.avg_response_time?.toFixed(2)}ms
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={2}>
                  <Box display="flex" alignItems="center">
                    <TrendingUpIcon color="success" sx={{ mr: 1 }} />
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Cache Hit Rate
                      </Typography>
                      <Typography variant="h6">
                        {(realtimeMetrics.cache_hit_rate * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={2}>
                  <Box display="flex" alignItems="center">
                    <PsychologyIcon color="info" sx={{ mr: 1 }} />
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Active Queries
                      </Typography>
                      <Typography variant="h6">
                        {realtimeMetrics.active_queries}
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={autoRefresh}
                        onChange={(e) => setAutoRefresh(e.target.checked)}
                        size="small"
                      />
                    }
                    label="Auto Refresh"
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={<RefreshIcon />}
                    onClick={loadRealtimeMetrics}
                    disabled={loading}
                  >
                    Refresh
                  </Button>
                </Grid>
              </Grid>
            </Paper>
          </Fade>
        )}

        <Grid container spacing={3}>
          {/* Query Interface */}
          <Grid item xs={12}>
            <GlassCard>
              <Box sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  🎯 Hybrid RAG Query Interface
                </Typography>
                
                <Box sx={{ mb: 3 }}>
                  <TextField
                    ref={queryInputRef}
                    fullWidth
                    multiline
                    rows={3}
                    variant="outlined"
                    placeholder="Enter your query for hybrid retrieval..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={handleKeyPress}
                    disabled={loading}
                    sx={{ mb: 2 }}
                  />
                  
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs={12} sm={6} md={3}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Retrieval Mode</InputLabel>
                        <Select
                          value={retrievalMode}
                          label="Retrieval Mode"
                          onChange={(e) => setRetrievalMode(e.target.value as RetrievalMode)}
                          disabled={loading}
                        >
                          <MenuItem value={RetrievalMode.VECTOR_ONLY}>Vector Only</MenuItem>
                          <MenuItem value={RetrievalMode.SPARSE_ONLY}>Sparse Only</MenuItem>
                          <MenuItem value={RetrievalMode.GRAPH_ONLY}>Graph Only</MenuItem>
                          <MenuItem value={RetrievalMode.HYBRID_VECTOR_SPARSE}>Vector + Sparse</MenuItem>
                          <MenuItem value={RetrievalMode.HYBRID_VECTOR_GRAPH}>Vector + Graph</MenuItem>
                          <MenuItem value={RetrievalMode.HYBRID_ALL}>All Methods</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    
                    <Grid item xs={12} sm={6} md={2}>
                      <TextField
                        fullWidth
                        size="small"
                        type="number"
                        label="Top K"
                        value={topK}
                        onChange={(e) => setTopK(Math.max(1, Math.min(50, parseInt(e.target.value) || 10)))}
                        disabled={loading}
                        inputProps={{ min: 1, max: 50 }}
                      />
                    </Grid>
                    
                    <Grid item xs={12} sm={6} md={3}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={includeReasoning}
                            onChange={(e) => setIncludeReasoning(e.target.checked)}
                            disabled={loading}
                          />
                        }
                        label="Include Reasoning"
                      />
                    </Grid>
                    
                    <Grid item xs={12} sm={6} md={2}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={enableReranking}
                            onChange={(e) => setEnableReranking(e.target.checked)}
                            disabled={loading}
                          />
                        }
                        label="Reranking"
                      />
                    </Grid>
                    
                    <Grid item xs={12} md={2}>
                      <Box display="flex" gap={1}>
                        <Button
                          variant="contained"
                          onClick={handleQuery}
                          disabled={loading || !query.trim()}
                          startIcon={<SearchIcon />}
                          fullWidth
                        >
                          {loading ? 'Searching...' : 'Search'}
                        </Button>
                      </Box>
                    </Grid>
                  </Grid>
                  
                  {query.trim() && (
                    <Box sx={{ mt: 2 }}>
                      <Button
                        variant="outlined"
                        onClick={handleCompareRetrievalModes}
                        disabled={loading}
                        startIcon={<CompareIcon />}
                        size="small"
                      >
                        Compare Retrieval Modes
                      </Button>
                    </Box>
                  )}
                </Box>

                {loading && (
                  <LinearProgress sx={{ mb: 2 }} />
                )}
              </Box>
            </GlassCard>
          </Grid>

          {/* Results Section */}
          <Grid item xs={12}>
            <GlassCard>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={activeTab} onChange={handleTabChange} aria-label="RAG dashboard tabs">
                  <Tab label="Query Results" />
                  <Tab label="Analytics" icon={<AnalyticsIcon />} />
                  <Tab label="Comparison" icon={<CompareIcon />} />
                  <Tab label="Configuration" icon={<TuneIcon />} />
                </Tabs>
              </Box>

              {/* Query Results Tab */}
              <TabPanel value={activeTab} index={0}>
                {results ? (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Box sx={{ mb: 3 }}>
                      <Grid container spacing={2} alignItems="center">
                        <Grid item>
                          <Chip
                            label={`${results.total_results} results`}
                            color="primary"
                            variant="outlined"
                          />
                        </Grid>
                        <Grid item>
                          <Chip
                            label={`${results.processing_time.toFixed(2)}ms`}
                            color="success"
                            variant="outlined"
                          />
                        </Grid>
                        <Grid item>
                          <Chip
                            label={results.mode.replace('_', ' ').toUpperCase()}
                            color="info"
                            variant="outlined"
                          />
                        </Grid>
                        {results.metadata.reranking_applied && (
                          <Grid item>
                            <Chip
                              label="Reranked"
                              color="warning"
                              variant="outlined"
                            />
                          </Grid>
                        )}
                      </Grid>
                    </Box>

                    {results.reasoning_steps && includeReasoning && (
                      <Accordion sx={{ mb: 3 }}>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Box display="flex" alignItems="center">
                            <PsychologyIcon sx={{ mr: 1 }} />
                            <Typography>Reasoning Steps ({results.reasoning_steps.length})</Typography>
                          </Box>
                        </AccordionSummary>
                        <AccordionDetails>
                          {results.reasoning_steps.map((step, index) => (
                            <Card key={index} sx={{ mb: 2 }}>
                              <CardContent>
                                <Typography variant="h6" gutterBottom>
                                  Step {step.step}: {step.method}
                                </Typography>
                                <Typography variant="body2" color="text.secondary" gutterBottom>
                                  {step.description}
                                </Typography>
                                <Box display="flex" gap={2} mt={1}>
                                  <Chip
                                    label={`${step.results_count} results`}
                                    size="small"
                                    variant="outlined"
                                  />
                                  <Chip
                                    label={`${step.processing_time.toFixed(2)}ms`}
                                    size="small"
                                    variant="outlined"
                                  />
                                </Box>
                              </CardContent>
                            </Card>
                          ))}
                        </AccordionDetails>
                      </Accordion>
                    )}

                    <Grid container spacing={2}>
                      {results.results.map((result, index) => (
                        <Grid item xs={12} key={result.id}>
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.1 }}
                          >
                            <Card sx={{ mb: 2 }}>
                              <CardContent>
                                <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                                  <Typography variant="h6" gutterBottom>
                                    Result #{index + 1}
                                  </Typography>
                                  <Box display="flex" gap={1}>
                                    <Chip
                                      label={`Score: ${result.score.toFixed(3)}`}
                                      size="small"
                                      color="primary"
                                    />
                                    <Chip
                                      label={result.retrieval_method}
                                      size="small"
                                      variant="outlined"
                                    />
                                  </Box>
                                </Box>
                                
                                <Typography variant="body2" color="text.secondary" gutterBottom>
                                  Source: {result.source}
                                </Typography>
                                
                                <Collapse in={expandedResults.includes(result.id)} timeout={300}>
                                  <Typography variant="body1" sx={{ mt: 2 }}>
                                    {result.content}
                                  </Typography>
                                  
                                  {result.metadata && Object.keys(result.metadata).length > 0 && (
                                    <Box sx={{ mt: 2 }}>
                                      <Typography variant="subtitle2" gutterBottom>
                                        Metadata:
                                      </Typography>
                                      <Box display="flex" flexWrap="wrap" gap={1}>
                                        {Object.entries(result.metadata).map(([key, value]) => (
                                          <Chip
                                            key={key}
                                            label={`${key}: ${value}`}
                                            size="small"
                                            variant="outlined"
                                          />
                                        ))}
                                      </Box>
                                    </Box>
                                  )}
                                </Collapse>
                              </CardContent>
                              
                              <CardActions>
                                <Button
                                  size="small"
                                  onClick={() => toggleResultExpansion(result.id)}
                                >
                                  {expandedResults.includes(result.id) ? 'Show Less' : 'Show More'}
                                </Button>
                                <IconButton size="small">
                                  <FeedbackIcon />
                                </IconButton>
                              </CardActions>
                            </Card>
                          </motion.div>
                        </Grid>
                      ))}
                    </Grid>
                  </motion.div>
                ) : (
                  <Box textAlign="center" py={8}>
                    <Typography variant="h6" color="text.secondary">
                      Enter a query to see hybrid RAG results
                    </Typography>
                  </Box>
                )}
              </TabPanel>

              {/* Analytics Tab */}
              <TabPanel value={activeTab} index={1}>
                {analytics ? (
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            Performance Metrics
                          </Typography>
                          <Box display="flex" flexDirection="column" gap={2}>
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                Average Response Time
                              </Typography>
                              <Typography variant="h4">
                                {analytics.avg_response_time.toFixed(2)}ms
                              </Typography>
                            </Box>
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                Total Queries
                              </Typography>
                              <Typography variant="h4">
                                {analytics.total_queries.toLocaleString()}
                              </Typography>
                            </Box>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                    
                    <Grid item xs={12} md={6}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            Quality Metrics
                          </Typography>
                          <Box display="flex" flexDirection="column" gap={2}>
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                Relevance Score
                              </Typography>
                              <Typography variant="h4">
                                {(analytics.quality_metrics.relevance_score * 100).toFixed(1)}%
                              </Typography>
                            </Box>
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                Mean Reciprocal Rank
                              </Typography>
                              <Typography variant="h4">
                                {analytics.quality_metrics.mrr.toFixed(3)}
                              </Typography>
                            </Box>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                    
                    <Grid item xs={12}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            Retrieval Mode Distribution
                          </Typography>
                          <AnimatedChart
                            type="pie"
                            data={Object.entries(analytics.mode_distribution).map(([mode, count]) => ({
                              name: mode.replace('_', ' ').toUpperCase(),
                              value: count
                            }))}
                            height={300}
                          />
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                ) : (
                  <Box textAlign="center" py={4}>
                    <Button
                      variant="contained"
                      onClick={loadAnalytics}
                      startIcon={<AnalyticsIcon />}
                    >
                      Load Analytics
                    </Button>
                  </Box>
                )}
              </TabPanel>

              {/* Comparison Tab */}
              <TabPanel value={activeTab} index={2}>
                {comparison ? (
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Retrieval Mode Comparison: "{comparison.query}"
                    </Typography>
                    
                    <Grid container spacing={3}>
                      {comparison.modes.map((mode) => {
                        const results = comparison.results[mode] || []
                        const uniqueResults = comparison.analysis.unique_results[mode] || 0
                        const qualityScore = comparison.analysis.quality_comparison[mode] || 0
                        
                        return (
                          <Grid item xs={12} md={6} key={mode}>
                            <Card>
                              <CardContent>
                                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                                  <Typography variant="h6">
                                    {mode.replace('_', ' ').toUpperCase()}
                                  </Typography>
                                  <Box display="flex" gap={1}>
                                    <Chip
                                      label={`${results.length} results`}
                                      size="small"
                                      color="primary"
                                    />
                                    <Chip
                                      label={`${uniqueResults} unique`}
                                      size="small"
                                      color="secondary"
                                    />
                                  </Box>
                                </Box>
                                
                                <Typography variant="body2" color="text.secondary" gutterBottom>
                                  Quality Score: {(qualityScore * 100).toFixed(1)}%
                                </Typography>
                                
                                <LinearProgress
                                  variant="determinate"
                                  value={qualityScore * 100}
                                  sx={{ mb: 2 }}
                                />
                                
                                {results.slice(0, 3).map((result, index) => (
                                  <Box key={result.id} sx={{ mb: 1, p: 1, bgcolor: 'background.default', borderRadius: 1 }}>
                                    <Typography variant="body2" fontWeight="medium">
                                      #{index + 1} (Score: {result.score.toFixed(3)})
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary">
                                      {result.content.substring(0, 100)}...
                                    </Typography>
                                  </Box>
                                ))}
                              </CardContent>
                            </Card>
                          </Grid>
                        )
                      })}
                    </Grid>
                    
                    <Card sx={{ mt: 3 }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Analysis Summary
                        </Typography>
                        <Typography variant="body1">
                          Overall overlap: {(comparison.analysis.overlap_percentage * 100).toFixed(1)}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                          This comparison shows how different retrieval methods perform for your query.
                          Higher overlap indicates consistent results across methods, while unique results
                          show the value of hybrid approaches.
                        </Typography>
                      </CardContent>
                    </Card>
                  </Box>
                ) : (
                  <Box textAlign="center" py={8}>
                    <Typography variant="h6" color="text.secondary">
                      Run a comparison query to see results
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      Use the "Compare Retrieval Modes" button in the query interface
                    </Typography>
                  </Box>
                )}
              </TabPanel>

              {/* Configuration Tab */}
              <TabPanel value={activeTab} index={3}>
                <Typography variant="h6" gutterBottom>
                  Configuration & Settings
                </Typography>
                <Alert severity="info" sx={{ mb: 3 }}>
                  Advanced configuration options will be available in the next version.
                  Current settings are applied per-query basis.
                </Alert>
              </TabPanel>
            </GlassCard>
          </Grid>
        </Grid>
      </motion.div>
    </PageWrapper>
  )
}

export default HybridRAGDashboard