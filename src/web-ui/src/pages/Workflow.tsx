import React, { useState } from 'react'
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Tabs,
  Tab,
  IconButton,
  Button,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  LinearProgress,
  Alert,
  Stack,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Badge
} from '@mui/material'
import {
  AccountTree as GitCommit,
  Build as BuildIcon,
  BugReport as BugIcon,
  Code as CodeIcon,
  Refresh as RefreshIcon,
  Timeline as TimelineIcon,
  Analytics as AnalyticsIcon,
  Storage as StorageIcon,
  Settings as SettingsIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon
} from '@mui/icons-material'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/services/api'

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
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  )
}

function WorkflowPage() {
  const [tabValue, setTabValue] = useState(0)
  const [gitRepo, setGitRepo] = useState('')
  const [cicdProvider, setCicdProvider] = useState('github')
  const [issueProvider, setIssueProvider] = useState('github')
  const [setupDialog, setSetupDialog] = useState(false)
  const queryClient = useQueryClient()

  // Workflow health check
  const { data: workflowHealth, isLoading: healthLoading } = useQuery({
    queryKey: ['workflow-health'],
    queryFn: async () => {
      try {
        const response = await fetch('/api/v1/workflow/health')
        return await response.json()
      } catch (error) {
        return { status: 'error', error: error.message }
      }
    },
    refetchInterval: 30000
  })

  // Git statistics
  const { data: gitStats } = useQuery({
    queryKey: ['git-stats'],
    queryFn: async () => {
      if (!gitRepo) return null
      try {
        const response = await fetch(`/api/v1/workflow/git/activity?repository_path=${gitRepo}&days=30`)
        return await response.json()
      } catch (error) {
        return null
      }
    },
    enabled: !!gitRepo
  })

  // CI/CD analytics
  const { data: cicdAnalytics } = useQuery({
    queryKey: ['cicd-analytics'],
    queryFn: async () => {
      try {
        const response = await fetch('/api/v1/workflow/cicd/analytics?days=30')
        return await response.json()
      } catch (error) {
        return null
      }
    }
  })

  // Issue metrics
  const { data: issueMetrics } = useQuery({
    queryKey: ['issue-metrics'],
    queryFn: async () => {
      try {
        const response = await fetch('/api/v1/workflow/issues/metrics?project_name=knowledgehub')
        return await response.json()
      } catch (error) {
        return null
      }
    }
  })

  // Workflow statistics
  const { data: workflowStats } = useQuery({
    queryKey: ['workflow-stats'],
    queryFn: async () => {
      try {
        const response = await fetch('/api/v1/workflow/stats?days=30')
        return await response.json()
      } catch (error) {
        return null
      }
    }
  })

  // Setup auto-capture mutation
  const setupMutation = useMutation({
    mutationFn: async (config: any) => {
      const response = await fetch('/api/v1/workflow/setup/auto-capture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })
      return await response.json()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['workflow-health'] })
      setSetupDialog(false)
    }
  })

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': case 'active': case 'success': return 'success'
      case 'warning': return 'warning'
      case 'error': case 'failed': return 'error'
      default: return 'default'
    }
  }

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Development Workflow Integration
        </Typography>
        <Button
          variant="contained"
          startIcon={<SettingsIcon />}
          onClick={() => setSetupDialog(true)}
        >
          Setup Auto-Capture
        </Button>
      </Box>

      {/* Health Status */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            System Health
          </Typography>
          {healthLoading ? (
            <CircularProgress size={24} />
          ) : workflowHealth ? (
            <Stack direction="row" spacing={2}>
              <Chip
                label="Git Capture"
                color={getStatusColor(workflowHealth.services?.git_capture)}
                icon={<GitCommit />}
              />
              <Chip
                label="CI/CD Integration"
                color={getStatusColor(workflowHealth.services?.cicd_integration)}
                icon={<BuildIcon />}
              />
              <Chip
                label="Issue Tracker"
                color={getStatusColor(workflowHealth.services?.issue_tracker)}
                icon={<BugIcon />}
              />
              <Chip
                label="IDE Integration"
                color={getStatusColor(workflowHealth.services?.ide_integration)}
                icon={<CodeIcon />}
              />
            </Stack>
          ) : (
            <Alert severity="warning">Unable to check system health</Alert>
          )}
        </CardContent>
      </Card>

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab icon={<GitCommit />} label="Git Integration" />
          <Tab icon={<BuildIcon />} label="CI/CD Pipelines" />
          <Tab icon={<BugIcon />} label="Issue Tracking" />
          <Tab icon={<CodeIcon />} label="IDE Integration" />
          <Tab icon={<AnalyticsIcon />} label="Analytics" />
        </Tabs>
      </Box>

      {/* Git Integration Tab */}
      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Repository Configuration
                </Typography>
                <TextField
                  fullWidth
                  label="Repository Path"
                  value={gitRepo}
                  onChange={(e) => setGitRepo(e.target.value)}
                  placeholder="/path/to/your/repo"
                  sx={{ mb: 2 }}
                />
                <Button
                  variant="outlined"
                  startIcon={<RefreshIcon />}
                  onClick={() => queryClient.invalidateQueries({ queryKey: ['git-stats'] })}
                  disabled={!gitRepo}
                >
                  Refresh Stats
                </Button>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Recent Activity
                </Typography>
                {gitStats?.activity ? (
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Total Commits: {gitStats.activity.total_commits}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Contributors: {gitStats.activity.contributors?.length || 0}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Files Changed: {gitStats.activity.files_changed || 0}
                    </Typography>
                  </Box>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    Configure repository to see activity
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* CI/CD Pipelines Tab */}
      <TabPanel value={tabValue} index={1}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Pipeline Provider
                </Typography>
                <FormControl fullWidth>
                  <InputLabel>Provider</InputLabel>
                  <Select
                    value={cicdProvider}
                    onChange={(e) => setCicdProvider(e.target.value)}
                    label="Provider"
                  >
                    <MenuItem value="github">GitHub Actions</MenuItem>
                    <MenuItem value="gitlab">GitLab CI</MenuItem>
                    <MenuItem value="jenkins">Jenkins</MenuItem>
                    <MenuItem value="azure">Azure DevOps</MenuItem>
                  </Select>
                </FormControl>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Pipeline Analytics
                </Typography>
                {cicdAnalytics?.analytics ? (
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Total Runs: {cicdAnalytics.analytics.total_pipeline_runs || 0}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Success Rate: {cicdAnalytics.analytics.success_rate || 0}%
                      </Typography>
                    </Grid>
                  </Grid>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No pipeline data available
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Issue Tracking Tab */}
      <TabPanel value={tabValue} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Issue Provider
                </Typography>
                <FormControl fullWidth>
                  <InputLabel>Provider</InputLabel>
                  <Select
                    value={issueProvider}
                    onChange={(e) => setIssueProvider(e.target.value)}
                    label="Provider"
                  >
                    <MenuItem value="github">GitHub Issues</MenuItem>
                    <MenuItem value="gitlab">GitLab Issues</MenuItem>
                    <MenuItem value="jira">JIRA</MenuItem>
                    <MenuItem value="linear">Linear</MenuItem>
                  </Select>
                </FormControl>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Project Metrics
                </Typography>
                {issueMetrics?.metrics ? (
                  <Grid container spacing={2}>
                    <Grid item xs={4}>
                      <Typography variant="h4" color="primary">
                        {issueMetrics.metrics.total_issues}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Total Issues
                      </Typography>
                    </Grid>
                    <Grid item xs={4}>
                      <Typography variant="h4" color="success.main">
                        {issueMetrics.metrics.open_issues}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Open Issues
                      </Typography>
                    </Grid>
                    <Grid item xs={4}>
                      <Typography variant="h4" color="text.secondary">
                        {issueMetrics.metrics.closed_issues}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Closed Issues
                      </Typography>
                    </Grid>
                  </Grid>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No issue metrics available
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* IDE Integration Tab */}
      <TabPanel value={tabValue} index={3}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  IDE Integration Status
                </Typography>
                <Alert severity="info" sx={{ mb: 2 }}>
                  IDE integration provides real-time context awareness and intelligent suggestions.
                  Install the KnowledgeHub plugin in your IDE to enable this feature.
                </Alert>
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Supported IDEs: VS Code, IntelliJ IDEA, WebStorm, PyCharm, Sublime Text
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Analytics Tab */}
      <TabPanel value={tabValue} index={4}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Workflow Statistics (Last 30 Days)
                </Typography>
                {workflowStats?.statistics ? (
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={3}>
                      <Paper sx={{ p: 2, textAlign: 'center' }}>
                        <GitCommit sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
                        <Typography variant="h6">
                          {workflowStats.statistics.git?.commits || 0}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Git Commits
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <Paper sx={{ p: 2, textAlign: 'center' }}>
                        <BuildIcon sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
                        <Typography variant="h6">
                          {workflowStats.statistics.cicd?.pipeline_runs || 0}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Pipeline Runs
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <Paper sx={{ p: 2, textAlign: 'center' }}>
                        <BugIcon sx={{ fontSize: 40, color: 'warning.main', mb: 1 }} />
                        <Typography variant="h6">
                          {workflowStats.statistics.issues?.total_synced || 0}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Issues Synced
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <Paper sx={{ p: 2, textAlign: 'center' }}>
                        <CodeIcon sx={{ fontSize: 40, color: 'info.main', mb: 1 }} />
                        <Typography variant="h6">
                          {workflowStats.statistics.ide?.events || 0}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          IDE Events
                        </Typography>
                      </Paper>
                    </Grid>
                  </Grid>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No workflow statistics available
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Setup Dialog */}
      <Dialog open={setupDialog} onClose={() => setSetupDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Setup Auto-Capture</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Configure automatic workflow capture for seamless integration
          </Typography>
          <TextField
            fullWidth
            label="Repository Path"
            value={gitRepo}
            onChange={(e) => setGitRepo(e.target.value)}
            sx={{ mb: 2 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSetupDialog(false)}>Cancel</Button>
          <Button
            onClick={() => setupMutation.mutate({
              repository_path: gitRepo,
              enable_git_hooks: true,
              enable_cicd_webhooks: true,
              enable_issue_sync: true
            })}
            variant="contained"
            disabled={!gitRepo || setupMutation.isPending}
          >
            {setupMutation.isPending ? <CircularProgress size={20} /> : 'Setup'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default WorkflowPage