/**
 * Web Ingestion Monitor
 * Real-time monitoring and management of Firecrawl jobs and web content ingestion
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
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material'
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  CloudDownload as CloudDownloadIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
  Schedule as ScheduleIcon,
  Visibility as VisibilityIcon,
  Settings as SettingsIcon,
  ExpandMore as ExpandMoreIcon,
  Language as LanguageIcon,
  Link as LinkIcon,
  Image as ImageIcon,
  Analytics as AnalyticsIcon
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'

import PageWrapper from '../components/PageWrapper'
import GlassCard from '../components/GlassCard'
import AnimatedChart from '../components/AnimatedChart'
import {
  webIngestionService,
  IngestionStatus,
  IngestionMode,
  IngestionJob,
  IngestionResult,
  IngestionAnalytics,
  WebSource
} from '../services/webIngestionService'

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
      id={`ingestion-tabpanel-${index}`}
      aria-labelledby={`ingestion-tab-${index}`}
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

const getStatusIcon = (status: IngestionStatus) => {
  switch (status) {
    case IngestionStatus.COMPLETED:
      return <CheckCircleIcon color="success" />
    case IngestionStatus.FAILED:
      return <ErrorIcon color="error" />
    case IngestionStatus.RUNNING:
      return <PlayIcon color="primary" />
    case IngestionStatus.PAUSED:
      return <PauseIcon color="warning" />
    default:
      return <ScheduleIcon color="disabled" />
  }
}

const getStatusColor = (status: IngestionStatus) => {
  switch (status) {
    case IngestionStatus.COMPLETED:
      return 'success'
    case IngestionStatus.FAILED:
      return 'error'
    case IngestionStatus.RUNNING:
      return 'primary'
    case IngestionStatus.PAUSED:
      return 'warning'
    default:
      return 'default'
  }
}

const WebIngestionMonitor: React.FC = () => {
  // State
  const [activeTab, setActiveTab] = useState(0)
  const [jobs, setJobs] = useState<IngestionJob[]>([])
  const [selectedJob, setSelectedJob] = useState<IngestionJob | null>(null)
  const [jobResults, setJobResults] = useState<IngestionResult[]>([])
  const [sources, setSources] = useState<WebSource[]>([])
  const [analytics, setAnalytics] = useState<IngestionAnalytics | null>(null)
  const [systemStatus, setSystemStatus] = useState<any>(null)
  
  // Create Job State
  const [createJobDialogOpen, setCreateJobDialogOpen] = useState(false)
  const [newJobName, setNewJobName] = useState('')
  const [newJobMode, setNewJobMode] = useState<IngestionMode>(IngestionMode.SINGLE_PAGE)
  const [newJobUrls, setNewJobUrls] = useState('')
  const [newJobConfig, setNewJobConfig] = useState({
    max_pages: 100,
    depth_limit: 3,
    rate_limit: 10,
    respect_robots_txt: true
  })
  
  // UI State
  const [loading, setLoading] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [jobDetailsDialogOpen, setJobDetailsDialogOpen] = useState(false)
  const [resultsDialogOpen, setResultsDialogOpen] = useState(false)
  
  const refreshInterval = useRef<NodeJS.Timeout>()

  // Effects
  useEffect(() => {
    loadJobs()
    loadSources()
    loadAnalytics()
    loadSystemStatus()
  }, [])

  useEffect(() => {
    if (autoRefresh) {
      refreshInterval.current = setInterval(() => {
        loadJobs()
        loadSystemStatus()
        if (selectedJob && selectedJob.status === IngestionStatus.RUNNING) {
          loadJobDetails(selectedJob.job_id)
        }
      }, 3000)
    } else {
      if (refreshInterval.current) {
        clearInterval(refreshInterval.current)
      }
    }

    return () => {
      if (refreshInterval.current) {
        clearInterval(refreshInterval.current)
      }
    }
  }, [autoRefresh, selectedJob])

  // API Calls
  const loadJobs = async () => {
    try {
      const response = await webIngestionService.getJobs({
        limit: 50,
        sort_by: 'start_time',
        sort_order: 'desc'
      })
      setJobs(response.jobs)
    } catch (error) {
      console.error('Failed to load jobs:', error)
    }
  }

  const loadSources = async () => {
    try {
      const data = await webIngestionService.getWebSources()
      setSources(data)
    } catch (error) {
      console.error('Failed to load sources:', error)
    }
  }

  const loadAnalytics = async () => {
    try {
      const data = await webIngestionService.getAnalytics('24h')
      setAnalytics(data)
    } catch (error) {
      console.error('Failed to load analytics:', error)
    }
  }

  const loadSystemStatus = async () => {
    try {
      const data = await webIngestionService.getSystemStatus()
      setSystemStatus(data)
    } catch (error) {
      console.error('Failed to load system status:', error)
    }
  }

  const loadJobDetails = async (jobId: string) => {
    try {
      const job = await webIngestionService.getJob(jobId)
      setSelectedJob(job)
    } catch (error) {
      console.error('Failed to load job details:', error)
    }
  }

  const loadJobResults = async (jobId: string) => {
    try {
      const response = await webIngestionService.getJobResults(jobId, { limit: 100 })
      setJobResults(response.results)
    } catch (error) {
      console.error('Failed to load job results:', error)
    }
  }

  const handleCreateJob = async () => {
    if (!newJobName.trim() || !newJobUrls.trim()) return

    setLoading(true)
    try {
      const urls = newJobUrls.split('\n').map(url => url.trim()).filter(url => url)
      
      const job = await webIngestionService.createJob({
        name: newJobName.trim(),
        mode: newJobMode,
        source_urls: urls,
        config: newJobConfig
      })

      setJobs(prev => [job, ...prev])
      setCreateJobDialogOpen(false)
      setNewJobName('')
      setNewJobUrls('')
      
      // Start the job immediately
      await webIngestionService.startJob(job.job_id)
      loadJobs()
    } catch (error) {
      console.error('Failed to create job:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleJobAction = async (jobId: string, action: 'start' | 'pause' | 'cancel') => {
    try {
      switch (action) {
        case 'start':
          await webIngestionService.startJob(jobId)
          break
        case 'pause':
          await webIngestionService.pauseJob(jobId)
          break
        case 'cancel':
          await webIngestionService.cancelJob(jobId)
          break
      }
      loadJobs()
    } catch (error) {
      console.error(`Failed to ${action} job:`, error)
    }
  }

  const handleViewJobDetails = (job: IngestionJob) => {
    setSelectedJob(job)
    setJobDetailsDialogOpen(true)
  }

  const handleViewJobResults = async (job: IngestionJob) => {
    setSelectedJob(job)
    await loadJobResults(job.job_id)
    setResultsDialogOpen(true)
  }

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue)
  }

  const getJobActionButton = (job: IngestionJob) => {
    switch (job.status) {
      case IngestionStatus.PENDING:
        return (
          <Button
            size="small"
            variant="contained"
            startIcon={<PlayIcon />}
            onClick={() => handleJobAction(job.job_id, 'start')}
          >
            Start
          </Button>
        )
      case IngestionStatus.RUNNING:
        return (
          <Box display="flex" gap={1}>
            <Button
              size="small"
              variant="outlined"
              startIcon={<PauseIcon />}
              onClick={() => handleJobAction(job.job_id, 'pause')}
            >
              Pause
            </Button>
            <Button
              size="small"
              variant="outlined"
              color="error"
              startIcon={<StopIcon />}
              onClick={() => handleJobAction(job.job_id, 'cancel')}
            >
              Cancel
            </Button>
          </Box>
        )
      case IngestionStatus.PAUSED:
        return (
          <Box display="flex" gap={1}>
            <Button
              size="small"
              variant="contained"
              startIcon={<PlayIcon />}
              onClick={() => handleJobAction(job.job_id, 'start')}
            >
              Resume
            </Button>
            <Button
              size="small"
              variant="outlined"
              color="error"
              startIcon={<StopIcon />}
              onClick={() => handleJobAction(job.job_id, 'cancel')}
            >
              Cancel
            </Button>
          </Box>
        )
      default:
        return null
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
          🌐 Web Ingestion Monitor
        </Typography>

        {/* System Status Bar */}
        {systemStatus && (
          <Fade in={true}>
            <Paper sx={{ p: 2, mb: 3, bgcolor: 'background.paper' }}>
              <Grid container spacing={3} alignItems="center">
                <Grid item xs={12} sm={6} md={2}>
                  <Box display="flex" alignItems="center">
                    <CloudDownloadIcon 
                      color={systemStatus.status === 'healthy' ? 'success' : 'error'} 
                      sx={{ mr: 1 }} 
                    />
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        System Status
                      </Typography>
                      <Typography variant="h6">
                        {systemStatus.status.toUpperCase()}
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={2}>
                  <Box display="flex" alignItems="center">
                    <PlayIcon color="primary" sx={{ mr: 1 }} />
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Active Jobs
                      </Typography>
                      <Typography variant="h6">
                        {systemStatus.active_jobs}
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={2}>
                  <Box display="flex" alignItems="center">
                    <SpeedIcon color="info" sx={{ mr: 1 }} />
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Processing Rate
                      </Typography>
                      <Typography variant="h6">
                        {systemStatus.processing_rate.toFixed(1)}/min
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
                    onClick={() => {
                      loadJobs()
                      loadSystemStatus()
                    }}
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
          {/* Action Bar */}
          <Grid item xs={12}>
            <GlassCard>
              <Box sx={{ p: 3 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Typography variant="h6">
                    🚀 Ingestion Control Center
                  </Typography>
                  <Button
                    variant="contained"
                    onClick={() => setCreateJobDialogOpen(true)}
                    startIcon={<AddIcon />}
                  >
                    Create New Job
                  </Button>
                </Box>
              </Box>
            </GlassCard>
          </Grid>

          {/* Main Content */}
          <Grid item xs={12}>
            <GlassCard>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={activeTab} onChange={handleTabChange} aria-label="ingestion dashboard tabs">
                  <Tab label="Active Jobs" />
                  <Tab label="Web Sources" />
                  <Tab label="Analytics" icon={<AnalyticsIcon />} />
                  <Tab label="System Health" />
                </Tabs>
              </Box>

              {/* Active Jobs Tab */}
              <TabPanel value={activeTab} index={0}>
                <Grid container spacing={2}>
                  {jobs.map((job) => (
                    <Grid item xs={12} key={job.job_id}>
                      <Card>
                        <CardContent>
                          <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                            <Box>
                              <Typography variant="h6" gutterBottom>
                                {job.name}
                              </Typography>
                              <Box display="flex" gap={1} mb={1}>
                                <Chip
                                  icon={getStatusIcon(job.status)}
                                  label={job.status.toUpperCase()}
                                  color={getStatusColor(job.status) as any}
                                  size="small"
                                />
                                <Chip
                                  label={job.mode.replace('_', ' ').toUpperCase()}
                                  variant="outlined"
                                  size="small"
                                />
                                <Chip
                                  label={`${job.source_urls.length} URLs`}
                                  variant="outlined"
                                  size="small"
                                />
                              </Box>
                            </Box>
                            <Box display="flex" gap={1}>
                              {getJobActionButton(job)}
                            </Box>
                          </Box>

                          {job.status === IngestionStatus.RUNNING && (
                            <Box sx={{ mb: 2 }}>
                              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                                <Typography variant="body2" color="text.secondary">
                                  Progress: {job.progress.processed_urls} / {job.progress.total_urls} URLs
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  {job.progress.percentage.toFixed(1)}%
                                </Typography>
                              </Box>
                              <LinearProgress 
                                variant="determinate" 
                                value={job.progress.percentage} 
                                sx={{ mb: 1 }}
                              />
                              <Box display="flex" justifyContent="space-between">
                                <Typography variant="caption" color="success.main">
                                  ✓ {job.progress.successful_urls} successful
                                </Typography>
                                <Typography variant="caption" color="error.main">
                                  ✗ {job.progress.failed_urls} failed
                                </Typography>
                              </Box>
                            </Box>
                          )}

                          <Box display="flex" gap={2} mb={1}>
                            <Typography variant="body2" color="text.secondary">
                              Started: {new Date(job.start_time).toLocaleString()}
                            </Typography>
                            {job.duration && (
                              <Typography variant="body2" color="text.secondary">
                                Duration: {(job.duration / 1000 / 60).toFixed(1)} min
                              </Typography>
                            )}
                          </Box>
                        </CardContent>

                        <CardActions>
                          <Button
                            size="small"
                            onClick={() => handleViewJobDetails(job)}
                            startIcon={<VisibilityIcon />}
                          >
                            Details
                          </Button>
                          <Button
                            size="small"
                            onClick={() => handleViewJobResults(job)}
                            startIcon={<StorageIcon />}
                            disabled={job.progress.processed_urls === 0}
                          >
                            Results ({job.progress.processed_urls})
                          </Button>
                        </CardActions>
                      </Card>
                    </Grid>
                  ))}
                </Grid>

                {jobs.length === 0 && (
                  <Box textAlign="center" py={8}>
                    <Typography variant="h6" color="text.secondary">
                      No ingestion jobs found
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      Create your first web ingestion job to get started
                    </Typography>
                  </Box>
                )}
              </TabPanel>

              {/* Web Sources Tab */}
              <TabPanel value={activeTab} index={1}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
                  <Typography variant="h6">
                    Web Sources
                  </Typography>
                  <Button
                    variant="outlined"
                    startIcon={<AddIcon />}
                    onClick={() => {/* TODO: Add source dialog */}}
                  >
                    Add Source
                  </Button>
                </Box>

                <Grid container spacing={2}>
                  {sources.map((source) => (
                    <Grid item xs={12} md={6} key={source.id}>
                      <Card>
                        <CardContent>
                          <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                            <Box>
                              <Typography variant="h6" gutterBottom>
                                {source.name}
                              </Typography>
                              <Typography variant="body2" color="text.secondary" gutterBottom>
                                {source.base_url}
                              </Typography>
                            </Box>
                            <Chip
                              label={source.status}
                              color={source.status === 'active' ? 'success' : 'error'}
                              size="small"
                            />
                          </Box>

                          <Box display="flex" gap={1} mb={2}>
                            <Chip
                              label={source.type.toUpperCase()}
                              variant="outlined"
                              size="small"
                            />
                            <Chip
                              label={`${source.statistics.total_pages} pages`}
                              variant="outlined"
                              size="small"
                            />
                          </Box>

                          <Typography variant="body2" color="text.secondary">
                            Last Update: {new Date(source.statistics.last_update).toLocaleString()}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Crawl Frequency: {source.config.crawl_frequency}
                          </Typography>
                        </CardContent>

                        <CardActions>
                          <Button
                            size="small"
                            onClick={() => webIngestionService.triggerSourceCrawl(source.id)}
                            startIcon={<PlayIcon />}
                          >
                            Crawl Now
                          </Button>
                          <Button
                            size="small"
                            startIcon={<SettingsIcon />}
                          >
                            Configure
                          </Button>
                        </CardActions>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </TabPanel>

              {/* Analytics Tab */}
              <TabPanel value={activeTab} index={2}>
                {analytics ? (
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            Performance Overview
                          </Typography>
                          <Grid container spacing={2}>
                            <Grid item xs={6}>
                              <Box textAlign="center">
                                <Typography variant="h4" color="primary.main">
                                  {analytics.total_jobs}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  Total Jobs
                                </Typography>
                              </Box>
                            </Grid>
                            <Grid item xs={6}>
                              <Box textAlign="center">
                                <Typography variant="h4" color="success.main">
                                  {(analytics.success_rate * 100).toFixed(1)}%
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  Success Rate
                                </Typography>
                              </Box>
                            </Grid>
                            <Grid item xs={6}>
                              <Box textAlign="center">
                                <Typography variant="h4" color="info.main">
                                  {analytics.total_pages_processed.toLocaleString()}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  Pages Processed
                                </Typography>
                              </Box>
                            </Grid>
                            <Grid item xs={6}>
                              <Box textAlign="center">
                                <Typography variant="h4" color="warning.main">
                                  {analytics.performance_metrics.pages_per_hour.toFixed(0)}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  Pages/Hour
                                </Typography>
                              </Box>
                            </Grid>
                          </Grid>
                        </CardContent>
                      </Card>
                    </Grid>

                    <Grid item xs={12} md={6}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            Data Volume
                          </Typography>
                          <Box display="flex" flexDirection="column" gap={2}>
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                Total Size
                              </Typography>
                              <Typography variant="h4">
                                {(analytics.data_volume.total_size / 1024 / 1024).toFixed(1)} MB
                              </Typography>
                            </Box>
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                Average Page Size
                              </Typography>
                              <Typography variant="h5">
                                {(analytics.data_volume.avg_page_size / 1024).toFixed(1)} KB
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
                            Content Types Distribution
                          </Typography>
                          <AnimatedChart
                            type="pie"
                            data={Object.entries(analytics.data_volume.content_types).map(([type, count]) => ({
                              name: type.toUpperCase(),
                              value: count
                            }))}
                            height={300}
                          />
                        </CardContent>
                      </Card>
                    </Grid>

                    <Grid item xs={12}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            Processing Timeline
                          </Typography>
                          <AnimatedChart
                            type="line"
                            data={analytics.timeline.map(item => ({
                              date: new Date(item.date).toLocaleDateString(),
                              'Jobs Started': item.jobs_started,
                              'Jobs Completed': item.jobs_completed,
                              'Pages Processed': item.pages_processed,
                              'Errors': item.errors
                            }))}
                            height={350}
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

              {/* System Health Tab */}
              <TabPanel value={activeTab} index={3}>
                {systemStatus ? (
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            Service Status
                          </Typography>
                          <List>
                            {Object.entries(systemStatus.services).map(([service, status]) => (
                              <ListItem key={service}>
                                <ListItemIcon>
                                  {status === 'up' ? (
                                    <CheckCircleIcon color="success" />
                                  ) : (
                                    <ErrorIcon color="error" />
                                  )}
                                </ListItemIcon>
                                <ListItemText
                                  primary={service.toUpperCase()}
                                  secondary={status.toUpperCase()}
                                />
                              </ListItem>
                            ))}
                          </List>
                        </CardContent>
                      </Card>
                    </Grid>

                    <Grid item xs={12} md={6}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            System Load
                          </Typography>
                          <Box display="flex" flexDirection="column" gap={3}>
                            <Box>
                              <Box display="flex" justifyContent="space-between" mb={1}>
                                <Typography variant="body2">CPU Usage</Typography>
                                <Typography variant="body2">
                                  {systemStatus.system_load.cpu_usage.toFixed(1)}%
                                </Typography>
                              </Box>
                              <LinearProgress
                                variant="determinate"
                                value={systemStatus.system_load.cpu_usage}
                                color={systemStatus.system_load.cpu_usage > 80 ? 'error' : 'primary'}
                              />
                            </Box>
                            <Box>
                              <Box display="flex" justifyContent="space-between" mb={1}>
                                <Typography variant="body2">Memory Usage</Typography>
                                <Typography variant="body2">
                                  {systemStatus.system_load.memory_usage.toFixed(1)}%
                                </Typography>
                              </Box>
                              <LinearProgress
                                variant="determinate"
                                value={systemStatus.system_load.memory_usage}
                                color={systemStatus.system_load.memory_usage > 80 ? 'error' : 'primary'}
                              />
                            </Box>
                            <Box>
                              <Box display="flex" justifyContent="space-between" mb={1}>
                                <Typography variant="body2">Disk Usage</Typography>
                                <Typography variant="body2">
                                  {systemStatus.system_load.disk_usage.toFixed(1)}%
                                </Typography>
                              </Box>
                              <LinearProgress
                                variant="determinate"
                                value={systemStatus.system_load.disk_usage}
                                color={systemStatus.system_load.disk_usage > 85 ? 'error' : 'primary'}
                              />
                            </Box>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>

                    <Grid item xs={12}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            Performance Metrics
                          </Typography>
                          <Grid container spacing={3}>
                            <Grid item xs={12} sm={6} md={3}>
                              <Box textAlign="center">
                                <Typography variant="h4" color="primary.main">
                                  {systemStatus.queue_length}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  Queue Length
                                </Typography>
                              </Box>
                            </Grid>
                            <Grid item xs={12} sm={6} md={3}>
                              <Box textAlign="center">
                                <Typography variant="h4" color="success.main">
                                  {systemStatus.processing_rate.toFixed(1)}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  Processing Rate/min
                                </Typography>
                              </Box>
                            </Grid>
                            <Grid item xs={12} sm={6} md={3}>
                              <Box textAlign="center">
                                <Typography variant="h4" color="error.main">
                                  {(systemStatus.error_rate * 100).toFixed(1)}%
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  Error Rate
                                </Typography>
                              </Box>
                            </Grid>
                            <Grid item xs={12} sm={6} md={3}>
                              <Box textAlign="center">
                                <Typography variant="h4" color="info.main">
                                  {systemStatus.active_jobs}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  Active Jobs
                                </Typography>
                              </Box>
                            </Grid>
                          </Grid>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                ) : (
                  <Box textAlign="center" py={4}>
                    <Button
                      variant="contained"
                      onClick={loadSystemStatus}
                      startIcon={<RefreshIcon />}
                    >
                      Load System Status
                    </Button>
                  </Box>
                )}
              </TabPanel>
            </GlassCard>
          </Grid>
        </Grid>

        {/* Create Job Dialog */}
        <Dialog
          open={createJobDialogOpen}
          onClose={() => setCreateJobDialogOpen(false)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>Create New Ingestion Job</DialogTitle>
          <DialogContent>
            <Grid container spacing={2} sx={{ mt: 1 }}>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Job Name"
                  value={newJobName}
                  onChange={(e) => setNewJobName(e.target.value)}
                  placeholder="Enter a descriptive name for this job"
                />
              </Grid>
              
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Ingestion Mode</InputLabel>
                  <Select
                    value={newJobMode}
                    label="Ingestion Mode"
                    onChange={(e) => setNewJobMode(e.target.value as IngestionMode)}
                  >
                    <MenuItem value={IngestionMode.SINGLE_PAGE}>Single Page</MenuItem>
                    <MenuItem value={IngestionMode.SITEMAP}>Sitemap</MenuItem>
                    <MenuItem value={IngestionMode.CRAWL}>Crawl</MenuItem>
                    <MenuItem value={IngestionMode.BATCH_URLS}>Batch URLs</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  multiline
                  rows={4}
                  label="URLs"
                  value={newJobUrls}
                  onChange={(e) => setNewJobUrls(e.target.value)}
                  placeholder="Enter URLs (one per line)"
                  helperText="Enter one URL per line. For crawl mode, enter the starting URL."
                />
              </Grid>
              
              <Grid item xs={12}>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>Advanced Configuration</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6}>
                        <TextField
                          fullWidth
                          type="number"
                          label="Max Pages"
                          value={newJobConfig.max_pages}
                          onChange={(e) => setNewJobConfig({
                            ...newJobConfig,
                            max_pages: parseInt(e.target.value) || 100
                          })}
                        />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <TextField
                          fullWidth
                          type="number"
                          label="Depth Limit"
                          value={newJobConfig.depth_limit}
                          onChange={(e) => setNewJobConfig({
                            ...newJobConfig,
                            depth_limit: parseInt(e.target.value) || 3
                          })}
                        />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <TextField
                          fullWidth
                          type="number"
                          label="Rate Limit (req/sec)"
                          value={newJobConfig.rate_limit}
                          onChange={(e) => setNewJobConfig({
                            ...newJobConfig,
                            rate_limit: parseInt(e.target.value) || 10
                          })}
                        />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={newJobConfig.respect_robots_txt}
                              onChange={(e) => setNewJobConfig({
                                ...newJobConfig,
                                respect_robots_txt: e.target.checked
                              })}
                            />
                          }
                          label="Respect robots.txt"
                        />
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              </Grid>
            </Grid>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setCreateJobDialogOpen(false)}>
              Cancel
            </Button>
            <Button 
              onClick={handleCreateJob}
              variant="contained"
              disabled={!newJobName.trim() || !newJobUrls.trim() || loading}
            >
              {loading ? 'Creating...' : 'Create & Start Job'}
            </Button>
          </DialogActions>
        </Dialog>

        {/* Job Details Dialog */}
        <Dialog
          open={jobDetailsDialogOpen}
          onClose={() => setJobDetailsDialogOpen(false)}
          maxWidth="lg"
          fullWidth
        >
          <DialogTitle>
            Job Details: {selectedJob?.name}
          </DialogTitle>
          <DialogContent>
            {selectedJob && (
              <Box>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>Configuration</Typography>
                    <TableContainer component={Paper} variant="outlined">
                      <Table size="small">
                        <TableBody>
                          <TableRow>
                            <TableCell>Mode</TableCell>
                            <TableCell>{selectedJob.mode}</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Max Pages</TableCell>
                            <TableCell>{selectedJob.config.max_pages || 'Unlimited'}</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Depth Limit</TableCell>
                            <TableCell>{selectedJob.config.depth_limit || 'Unlimited'}</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Rate Limit</TableCell>
                            <TableCell>{selectedJob.config.rate_limit || 'None'} req/sec</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Respect robots.txt</TableCell>
                            <TableCell>{selectedJob.config.respect_robots_txt ? 'Yes' : 'No'}</TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>Progress</Typography>
                    <Box display="flex" flexDirection="column" gap={2}>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Total URLs: {selectedJob.progress.total_urls}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Processed: {selectedJob.progress.processed_urls}
                        </Typography>
                        <Typography variant="body2" color="success.main">
                          Successful: {selectedJob.progress.successful_urls}
                        </Typography>
                        <Typography variant="body2" color="error.main">
                          Failed: {selectedJob.progress.failed_urls}
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={selectedJob.progress.percentage}
                        sx={{ height: 8, borderRadius: 4 }}
                      />
                      <Typography variant="body2" color="text.secondary">
                        {selectedJob.progress.percentage.toFixed(1)}% Complete
                      </Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12}>
                    <Typography variant="h6" gutterBottom>Source URLs</Typography>
                    <Box sx={{ maxHeight: 200, overflow: 'auto' }}>
                      {selectedJob.source_urls.map((url, index) => (
                        <Box key={index} display="flex" alignItems="center" gap={1} mb={1}>
                          <LinkIcon fontSize="small" color="primary" />
                          <Typography variant="body2" sx={{ wordBreak: 'break-all' }}>
                            {url}
                          </Typography>
                        </Box>
                      ))}
                    </Box>
                  </Grid>
                </Grid>
              </Box>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setJobDetailsDialogOpen(false)}>
              Close
            </Button>
          </DialogActions>
        </Dialog>

        {/* Job Results Dialog */}
        <Dialog
          open={resultsDialogOpen}
          onClose={() => setResultsDialogOpen(false)}
          maxWidth="lg"
          fullWidth
        >
          <DialogTitle>
            Job Results: {selectedJob?.name}
          </DialogTitle>
          <DialogContent>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>URL</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Title</TableCell>
                    <TableCell>Word Count</TableCell>
                    <TableCell>Processing Time</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {jobResults.map((result, index) => (
                    <TableRow key={index}>
                      <TableCell sx={{ maxWidth: 200, wordBreak: 'break-all' }}>
                        {result.url}
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={result.status}
                          color={result.status === 'success' ? 'success' : 'error'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell sx={{ maxWidth: 200 }}>
                        {result.title || '-'}
                      </TableCell>
                      <TableCell>
                        {result.metadata?.word_count || '-'}
                      </TableCell>
                      <TableCell>
                        {result.processing_time ? `${result.processing_time.toFixed(2)}ms` : '-'}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setResultsDialogOpen(false)}>
              Close
            </Button>
          </DialogActions>
        </Dialog>
      </motion.div>
    </PageWrapper>
  )
}

export default WebIngestionMonitor