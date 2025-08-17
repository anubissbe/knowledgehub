/**
 * Agent Workflows Page
 * Interface for LangGraph-based multi-agent workflow execution and monitoring
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
  Stepper,
  Step,
  StepLabel,
  StepContent,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material'
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Psychology as PsychologyIcon,
  Timeline as TimelineIcon,
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  History as HistoryIcon,
  Speed as SpeedIcon,
  Analytics as AnalyticsIcon,
  Settings as SettingsIcon,
  Person as PersonIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Schedule as ScheduleIcon,
  Visibility as VisibilityIcon
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'

import PageWrapper from '../components/PageWrapper'
import GlassCard from '../components/GlassCard'
import AnimatedChart from '../components/AnimatedChart'
import {
  agentWorkflowService,
  WorkflowType,
  AgentRole,
  WorkflowStatus,
  WorkflowExecution,
  WorkflowTemplate,
  AgentCoordinationMetrics,
  WorkflowAnalytics
} from '../services/agentWorkflowService'

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
      id={`workflow-tabpanel-${index}`}
      aria-labelledby={`workflow-tab-${index}`}
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

const getStatusIcon = (status: WorkflowStatus) => {
  switch (status) {
    case WorkflowStatus.COMPLETED:
      return <CheckCircleIcon color="success" />
    case WorkflowStatus.FAILED:
      return <ErrorIcon color="error" />
    case WorkflowStatus.RUNNING:
      return <ScheduleIcon color="primary" />
    default:
      return <ScheduleIcon color="disabled" />
  }
}

const getStatusColor = (status: WorkflowStatus) => {
  switch (status) {
    case WorkflowStatus.COMPLETED:
      return 'success'
    case WorkflowStatus.FAILED:
      return 'error'
    case WorkflowStatus.RUNNING:
      return 'primary'
    case WorkflowStatus.PENDING:
      return 'warning'
    default:
      return 'default'
  }
}

const AgentWorkflows: React.FC = () => {
  // State
  const [activeTab, setActiveTab] = useState(0)
  const [query, setQuery] = useState('')
  const [workflowType, setWorkflowType] = useState<WorkflowType>(WorkflowType.SIMPLE_QA)
  const [currentExecution, setCurrentExecution] = useState<WorkflowExecution | null>(null)
  const [executionHistory, setExecutionHistory] = useState<WorkflowExecution[]>([])
  const [templates, setTemplates] = useState<WorkflowTemplate[]>([])
  const [coordinationMetrics, setCoordinationMetrics] = useState<AgentCoordinationMetrics | null>(null)
  const [analytics, setAnalytics] = useState<WorkflowAnalytics | null>(null)
  
  // UI State
  const [loading, setLoading] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [selectedExecution, setSelectedExecution] = useState<string | null>(null)
  const [debugDialogOpen, setDebugDialogOpen] = useState(false)
  const [debugInfo, setDebugInfo] = useState<any>(null)
  
  const sessionId = useRef<string>(`workflow_session_${Date.now()}`)

  // Effects
  useEffect(() => {
    loadTemplates()
    loadCoordinationMetrics()
    loadExecutionHistory()
    loadAnalytics()
  }, [])

  useEffect(() => {
    let interval: NodeJS.Timeout
    if (autoRefresh) {
      interval = setInterval(() => {
        loadCoordinationMetrics()
        if (currentExecution && currentExecution.status === WorkflowStatus.RUNNING) {
          refreshCurrentExecution()
        }
      }, 2000)
    }
    return () => clearInterval(interval)
  }, [autoRefresh, currentExecution])

  // API Calls
  const loadTemplates = async () => {
    try {
      const data = await agentWorkflowService.getWorkflowTemplates()
      setTemplates(data)
    } catch (error) {
      console.error('Failed to load templates:', error)
    }
  }

  const loadCoordinationMetrics = async () => {
    try {
      const data = await agentWorkflowService.getCoordinationMetrics()
      setCoordinationMetrics(data)
    } catch (error) {
      console.error('Failed to load coordination metrics:', error)
    }
  }

  const loadExecutionHistory = async () => {
    try {
      const response = await agentWorkflowService.getWorkflowHistory(sessionId.current)
      setExecutionHistory(response.executions)
    } catch (error) {
      console.error('Failed to load execution history:', error)
    }
  }

  const loadAnalytics = async () => {
    try {
      const data = await agentWorkflowService.getAnalytics('24h')
      setAnalytics(data)
    } catch (error) {
      console.error('Failed to load analytics:', error)
    }
  }

  const refreshCurrentExecution = async () => {
    if (!currentExecution) return
    
    try {
      const updated = await agentWorkflowService.getWorkflowStatus(currentExecution.execution_id)
      setCurrentExecution(updated)
      
      if (updated.status !== WorkflowStatus.RUNNING) {
        setStreaming(false)
        loadExecutionHistory()
      }
    } catch (error) {
      console.error('Failed to refresh execution:', error)
    }
  }

  const handleExecuteWorkflow = async () => {
    if (!query.trim()) return

    setLoading(true)
    setStreaming(true)
    
    try {
      const request = {
        query: query.trim(),
        workflow_type: workflowType,
        session_id: sessionId.current,
        async_execution: false
      }

      // Start with immediate execution for now
      const execution = await agentWorkflowService.executeWorkflow(request)
      setCurrentExecution(execution)
      setSelectedExecution(execution.execution_id)
      
      // Switch to execution monitoring tab
      setActiveTab(1)
      
      // Start monitoring the execution
      if (execution.status === WorkflowStatus.RUNNING) {
        const checkStatus = setInterval(async () => {
          try {
            const updated = await agentWorkflowService.getWorkflowStatus(execution.execution_id)
            setCurrentExecution(updated)
            
            if (updated.status !== WorkflowStatus.RUNNING) {
              clearInterval(checkStatus)
              setStreaming(false)
              loadExecutionHistory()
            }
          } catch (error) {
            console.error('Status check failed:', error)
            clearInterval(checkStatus)
            setStreaming(false)
          }
        }, 1000)
      } else {
        setStreaming(false)
      }
      
    } catch (error) {
      console.error('Workflow execution failed:', error)
      setStreaming(false)
    } finally {
      setLoading(false)
    }
  }

  const handleCancelWorkflow = async () => {
    if (!currentExecution || currentExecution.status !== WorkflowStatus.RUNNING) return
    
    try {
      await agentWorkflowService.cancelWorkflow(currentExecution.execution_id)
      refreshCurrentExecution()
    } catch (error) {
      console.error('Failed to cancel workflow:', error)
    }
  }

  const handleDebugExecution = async (executionId: string) => {
    try {
      setLoading(true)
      const debugData = await agentWorkflowService.getWorkflowDebugInfo(executionId)
      setDebugInfo(debugData)
      setDebugDialogOpen(true)
    } catch (error) {
      console.error('Failed to load debug info:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue)
  }

  const getWorkflowTypeDescription = (type: WorkflowType) => {
    switch (type) {
      case WorkflowType.SIMPLE_QA:
        return 'Direct question answering with single-step retrieval'
      case WorkflowType.MULTI_STEP_RESEARCH:
        return 'Complex research involving multiple agents and analysis steps'
      case WorkflowType.DOCUMENT_ANALYSIS:
        return 'Deep analysis of documents with structured insights'
      case WorkflowType.CONVERSATION_SUMMARY:
        return 'Summarization of conversation threads with key points'
      case WorkflowType.KNOWLEDGE_SYNTHESIS:
        return 'Synthesis of knowledge from multiple sources'
      case WorkflowType.FACT_CHECKING:
        return 'Verification of claims with evidence gathering'
      default:
        return 'Custom workflow execution'
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
          🤖 Agent Workflows
        </Typography>

        {/* Coordination Metrics Bar */}
        {coordinationMetrics && (
          <Fade in={true}>
            <Paper sx={{ p: 2, mb: 3, bgcolor: 'background.paper' }}>
              <Grid container spacing={3} alignItems="center">
                <Grid item xs={12} sm={6} md={2}>
                  <Box display="flex" alignItems="center">
                    <PlayIcon color="primary" sx={{ mr: 1 }} />
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Active Workflows
                      </Typography>
                      <Typography variant="h6">
                        {coordinationMetrics.active_workflows}
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={2}>
                  <Box display="flex" alignItems="center">
                    <ScheduleIcon color="warning" sx={{ mr: 1 }} />
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Queued
                      </Typography>
                      <Typography variant="h6">
                        {coordinationMetrics.queued_workflows}
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box display="flex" alignItems="center">
                    <SpeedIcon color="info" sx={{ mr: 1 }} />
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        CPU Usage
                      </Typography>
                      <Typography variant="h6">
                        {coordinationMetrics.resource_utilization.cpu_usage.toFixed(1)}%
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={2}>
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
                    onClick={loadCoordinationMetrics}
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
          {/* Workflow Execution Interface */}
          <Grid item xs={12}>
            <GlassCard>
              <Box sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  🎯 Workflow Execution Interface
                </Typography>
                
                <Box sx={{ mb: 3 }}>
                  <TextField
                    fullWidth
                    multiline
                    rows={3}
                    variant="outlined"
                    placeholder="Enter your query for agent workflow processing..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    disabled={loading || streaming}
                    sx={{ mb: 2 }}
                  />
                  
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs={12} sm={6} md={4}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Workflow Type</InputLabel>
                        <Select
                          value={workflowType}
                          label="Workflow Type"
                          onChange={(e) => setWorkflowType(e.target.value as WorkflowType)}
                          disabled={loading || streaming}
                        >
                          <MenuItem value={WorkflowType.SIMPLE_QA}>Simple Q&A</MenuItem>
                          <MenuItem value={WorkflowType.MULTI_STEP_RESEARCH}>Multi-Step Research</MenuItem>
                          <MenuItem value={WorkflowType.DOCUMENT_ANALYSIS}>Document Analysis</MenuItem>
                          <MenuItem value={WorkflowType.CONVERSATION_SUMMARY}>Conversation Summary</MenuItem>
                          <MenuItem value={WorkflowType.KNOWLEDGE_SYNTHESIS}>Knowledge Synthesis</MenuItem>
                          <MenuItem value={WorkflowType.FACT_CHECKING}>Fact Checking</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    
                    <Grid item xs={12} sm={6} md={4}>
                      <Box display="flex" gap={1}>
                        <Button
                          variant="contained"
                          onClick={handleExecuteWorkflow}
                          disabled={loading || streaming || !query.trim()}
                          startIcon={<PlayIcon />}
                          fullWidth
                        >
                          {loading ? 'Starting...' : streaming ? 'Running...' : 'Execute Workflow'}
                        </Button>
                        {streaming && (
                          <Button
                            variant="outlined"
                            onClick={handleCancelWorkflow}
                            startIcon={<StopIcon />}
                            color="error"
                          >
                            Cancel
                          </Button>
                        )}
                      </Box>
                    </Grid>
                    
                    <Grid item xs={12} md={4}>
                      <Typography variant="body2" color="text.secondary">
                        {getWorkflowTypeDescription(workflowType)}
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>

                {(loading || streaming) && (
                  <LinearProgress sx={{ mb: 2 }} />
                )}
              </Box>
            </GlassCard>
          </Grid>

          {/* Results and Monitoring Section */}
          <Grid item xs={12}>
            <GlassCard>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={activeTab} onChange={handleTabChange} aria-label="workflow dashboard tabs">
                  <Tab label="Execution Monitor" icon={<TimelineIcon />} />
                  <Tab label="History" icon={<HistoryIcon />} />
                  <Tab label="Analytics" icon={<AnalyticsIcon />} />
                  <Tab label="Agent Status" icon={<PersonIcon />} />
                </Tabs>
              </Box>

              {/* Execution Monitor Tab */}
              <TabPanel value={activeTab} index={0}>
                {currentExecution ? (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Box sx={{ mb: 3 }}>
                      <Grid container spacing={2} alignItems="center">
                        <Grid item>
                          <Chip
                            icon={getStatusIcon(currentExecution.status)}
                            label={currentExecution.status.toUpperCase()}
                            color={getStatusColor(currentExecution.status) as any}
                          />
                        </Grid>
                        <Grid item>
                          <Chip
                            label={currentExecution.workflow_type.replace('_', ' ').toUpperCase()}
                            variant="outlined"
                          />
                        </Grid>
                        <Grid item>
                          <Chip
                            label={`${currentExecution.steps.length} steps`}
                            variant="outlined"
                          />
                        </Grid>
                        {currentExecution.total_execution_time && (
                          <Grid item>
                            <Chip
                              label={`${currentExecution.total_execution_time.toFixed(2)}ms`}
                              color="success"
                              variant="outlined"
                            />
                          </Grid>
                        )}
                      </Grid>
                    </Box>

                    <Typography variant="h6" gutterBottom>
                      Query: "{currentExecution.query}"
                    </Typography>

                    {currentExecution.steps.length > 0 && (
                      <Stepper orientation="vertical" sx={{ mt: 3 }}>
                        {currentExecution.steps.map((step, index) => (
                          <Step key={step.step_id} active={true} completed={true}>
                            <StepLabel>
                              <Box display="flex" alignItems="center" gap={1}>
                                <PersonIcon fontSize="small" />
                                <Typography variant="subtitle1">
                                  {step.agent_role.replace('_', ' ').toUpperCase()}: {step.action}
                                </Typography>
                                <Chip
                                  label={`${step.execution_time.toFixed(2)}ms`}
                                  size="small"
                                  variant="outlined"
                                />
                              </Box>
                            </StepLabel>
                            <StepContent>
                              <Box sx={{ mb: 2 }}>
                                <Typography variant="body2" color="text.secondary" gutterBottom>
                                  Reasoning: {step.reasoning}
                                </Typography>
                                
                                {step.input && (
                                  <Accordion>
                                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                      <Typography variant="subtitle2">Input Data</Typography>
                                    </AccordionSummary>
                                    <AccordionDetails>
                                      <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.8rem' }}>
                                        {JSON.stringify(step.input, null, 2)}
                                      </pre>
                                    </AccordionDetails>
                                  </Accordion>
                                )}
                                
                                {step.output && (
                                  <Accordion>
                                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                      <Typography variant="subtitle2">Output Data</Typography>
                                    </AccordionSummary>
                                    <AccordionDetails>
                                      <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.8rem' }}>
                                        {JSON.stringify(step.output, null, 2)}
                                      </pre>
                                    </AccordionDetails>
                                  </Accordion>
                                )}
                              </Box>
                            </StepContent>
                          </Step>
                        ))}
                      </Stepper>
                    )}

                    {currentExecution.result && (
                      <Card sx={{ mt: 3 }}>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            Final Result
                          </Typography>
                          <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.9rem' }}>
                            {JSON.stringify(currentExecution.result, null, 2)}
                          </pre>
                        </CardContent>
                      </Card>
                    )}

                    {currentExecution.error && (
                      <Alert severity="error" sx={{ mt: 3 }}>
                        <Typography variant="subtitle2">Error:</Typography>
                        <Typography variant="body2">{currentExecution.error}</Typography>
                      </Alert>
                    )}
                  </motion.div>
                ) : (
                  <Box textAlign="center" py={8}>
                    <Typography variant="h6" color="text.secondary">
                      Execute a workflow to see real-time monitoring
                    </Typography>
                  </Box>
                )}
              </TabPanel>

              {/* History Tab */}
              <TabPanel value={activeTab} index={1}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
                  <Typography variant="h6">
                    Execution History
                  </Typography>
                  <Button
                    variant="outlined"
                    onClick={loadExecutionHistory}
                    startIcon={<RefreshIcon />}
                    size="small"
                  >
                    Refresh
                  </Button>
                </Box>

                <Grid container spacing={2}>
                  {executionHistory.map((execution) => (
                    <Grid item xs={12} key={execution.execution_id}>
                      <Card 
                        sx={{ 
                          cursor: 'pointer',
                          '&:hover': { boxShadow: 4 }
                        }}
                        onClick={() => setSelectedExecution(execution.execution_id)}
                      >
                        <CardContent>
                          <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                            <Typography variant="h6" noWrap sx={{ flex: 1, mr: 2 }}>
                              {execution.query}
                            </Typography>
                            <Box display="flex" gap={1}>
                              <Chip
                                icon={getStatusIcon(execution.status)}
                                label={execution.status}
                                color={getStatusColor(execution.status) as any}
                                size="small"
                              />
                            </Box>
                          </Box>
                          
                          <Box display="flex" gap={2} mb={1}>
                            <Chip
                              label={execution.workflow_type.replace('_', ' ')}
                              size="small"
                              variant="outlined"
                            />
                            <Chip
                              label={`${execution.metadata.total_steps} steps`}
                              size="small"
                              variant="outlined"
                            />
                            {execution.total_execution_time && (
                              <Chip
                                label={`${execution.total_execution_time.toFixed(2)}ms`}
                                size="small"
                                variant="outlined"
                              />
                            )}
                          </Box>
                          
                          <Typography variant="body2" color="text.secondary">
                            Started: {new Date(execution.start_time).toLocaleString()}
                          </Typography>
                        </CardContent>
                        
                        <CardActions>
                          <Button
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation()
                              setCurrentExecution(execution)
                              setActiveTab(0)
                            }}
                            startIcon={<VisibilityIcon />}
                          >
                            View Details
                          </Button>
                          <Button
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDebugExecution(execution.execution_id)
                            }}
                            startIcon={<SettingsIcon />}
                          >
                            Debug
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
                          <Box display="flex" flexDirection="column" gap={2}>
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                Total Executions
                              </Typography>
                              <Typography variant="h4">
                                {analytics.total_executions.toLocaleString()}
                              </Typography>
                            </Box>
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                Success Rate
                              </Typography>
                              <Typography variant="h4" color="success.main">
                                {(analytics.success_rate * 100).toFixed(1)}%
                              </Typography>
                            </Box>
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                Avg Execution Time
                              </Typography>
                              <Typography variant="h4">
                                {analytics.avg_execution_time.toFixed(2)}ms
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
                            Workflow Distribution
                          </Typography>
                          <AnimatedChart
                            type="pie"
                            data={Object.entries(analytics.workflow_distribution).map(([type, count]) => ({
                              name: type.replace('_', ' ').toUpperCase(),
                              value: count
                            }))}
                            height={250}
                          />
                        </CardContent>
                      </Card>
                    </Grid>
                    
                    <Grid item xs={12}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            Agent Utilization
                          </Typography>
                          <Grid container spacing={2}>
                            {Object.entries(analytics.agent_utilization).map(([agent, metrics]) => (
                              <Grid item xs={12} sm={6} md={4} key={agent}>
                                <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                                  <Typography variant="subtitle1" gutterBottom>
                                    {agent.replace('_', ' ').toUpperCase()}
                                  </Typography>
                                  <Box display="flex" flexDirection="column" gap={1}>
                                    <Box display="flex" justifyContent="space-between">
                                      <Typography variant="body2" color="text.secondary">
                                        Total Steps:
                                      </Typography>
                                      <Typography variant="body2">
                                        {metrics.total_steps}
                                      </Typography>
                                    </Box>
                                    <Box display="flex" justifyContent="space-between">
                                      <Typography variant="body2" color="text.secondary">
                                        Avg Time:
                                      </Typography>
                                      <Typography variant="body2">
                                        {metrics.avg_execution_time.toFixed(2)}ms
                                      </Typography>
                                    </Box>
                                    <Box display="flex" justifyContent="space-between">
                                      <Typography variant="body2" color="text.secondary">
                                        Success Rate:
                                      </Typography>
                                      <Typography variant="body2" color="success.main">
                                        {(metrics.success_rate * 100).toFixed(1)}%
                                      </Typography>
                                    </Box>
                                  </Box>
                                </Paper>
                              </Grid>
                            ))}
                          </Grid>
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

              {/* Agent Status Tab */}
              <TabPanel value={activeTab} index={3}>
                {coordinationMetrics ? (
                  <Grid container spacing={3}>
                    <Grid item xs={12}>
                      <Typography variant="h6" gutterBottom>
                        Agent Status Overview
                      </Typography>
                    </Grid>
                    
                    {Object.entries(coordinationMetrics.agent_status).map(([agent, status]) => (
                      <Grid item xs={12} sm={6} md={4} key={agent}>
                        <Card>
                          <CardContent>
                            <Box display="flex" alignItems="center" gap={2} mb={2}>
                              <PersonIcon color="primary" />
                              <Typography variant="h6">
                                {agent.replace('_', ' ').toUpperCase()}
                              </Typography>
                              <Chip
                                label={status.status}
                                color={status.status === 'idle' ? 'success' : status.status === 'busy' ? 'warning' : 'error'}
                                size="small"
                              />
                            </Box>
                            
                            {status.current_task && (
                              <Typography variant="body2" color="text.secondary" gutterBottom>
                                Current Task: {status.current_task}
                              </Typography>
                            )}
                            
                            <Typography variant="body2" color="text.secondary">
                              Queue Length: {status.queue_length}
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                    
                    <Grid item xs={12}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            Resource Utilization
                          </Typography>
                          <Grid container spacing={3}>
                            <Grid item xs={12} md={4}>
                              <Box>
                                <Typography variant="body2" color="text.secondary">
                                  CPU Usage
                                </Typography>
                                <LinearProgress
                                  variant="determinate"
                                  value={coordinationMetrics.resource_utilization.cpu_usage}
                                  sx={{ mt: 1, mb: 1 }}
                                />
                                <Typography variant="body2">
                                  {coordinationMetrics.resource_utilization.cpu_usage.toFixed(1)}%
                                </Typography>
                              </Box>
                            </Grid>
                            <Grid item xs={12} md={4}>
                              <Box>
                                <Typography variant="body2" color="text.secondary">
                                  Memory Usage
                                </Typography>
                                <LinearProgress
                                  variant="determinate"
                                  value={coordinationMetrics.resource_utilization.memory_usage}
                                  sx={{ mt: 1, mb: 1 }}
                                />
                                <Typography variant="body2">
                                  {coordinationMetrics.resource_utilization.memory_usage.toFixed(1)}%
                                </Typography>
                              </Box>
                            </Grid>
                            <Grid item xs={12} md={4}>
                              <Box>
                                <Typography variant="body2" color="text.secondary">
                                  Active Connections
                                </Typography>
                                <Typography variant="h4">
                                  {coordinationMetrics.resource_utilization.active_connections}
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
                      onClick={loadCoordinationMetrics}
                      startIcon={<RefreshIcon />}
                    >
                      Load Agent Status
                    </Button>
                  </Box>
                )}
              </TabPanel>
            </GlassCard>
          </Grid>
        </Grid>

        {/* Debug Dialog */}
        <Dialog
          open={debugDialogOpen}
          onClose={() => setDebugDialogOpen(false)}
          maxWidth="lg"
          fullWidth
        >
          <DialogTitle>
            Workflow Debug Information
          </DialogTitle>
          <DialogContent>
            {debugInfo && (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Execution Details
                </Typography>
                <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.8rem', backgroundColor: '#f5f5f5', padding: '1rem', borderRadius: '4px' }}>
                  {JSON.stringify(debugInfo, null, 2)}
                </pre>
              </Box>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDebugDialogOpen(false)}>
              Close
            </Button>
          </DialogActions>
        </Dialog>
      </motion.div>
    </PageWrapper>
  )
}

export default AgentWorkflows