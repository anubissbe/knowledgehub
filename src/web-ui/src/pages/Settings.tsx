import {
  Box,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  Grid,
  Switch,
  FormControlLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  IconButton,
  Chip,
  Alert,
} from '@mui/material'
import {
  Edit as EditIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
} from '@mui/icons-material'
import { useState, useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/services/api'

interface SettingsForm {
  maxCrawlDepth: number
  maxPagesPerCrawl: number
  chunkSize: number
  chunkOverlap: number
  searchResultLimit: number
  vectorSearchWeight: number
  keywordSearchWeight: number
  enableMetrics: boolean
  logLevel: string
}

interface SchedulerConfig {
  enabled: boolean
  defaultInterval: number
  checkInterval: number
}

interface SourceSchedule {
  id: string
  name: string
  url: string
  refreshInterval: number
  lastRefresh: string | null
  nextRefresh: string | null
  status: string
}

const REFRESH_INTERVALS = [
  { value: 3600, label: '1 Hour' },
  { value: 21600, label: '6 Hours' },
  { value: 43200, label: '12 Hours' },
  { value: 86400, label: '24 Hours' },
  { value: 604800, label: '7 Days' },
  { value: 2592000, label: '30 Days' },
  { value: 0, label: 'Never' },
]

function Settings() {
  const [saved, setSaved] = useState(false)
  const [editingSchedule, setEditingSchedule] = useState<string | null>(null)
  const [tempInterval, setTempInterval] = useState<number>(86400)
  const [schedulerConfig, setSchedulerConfig] = useState<SchedulerConfig>({
    enabled: true,
    defaultInterval: 86400,
    checkInterval: 3600,
  })
  const queryClient = useQueryClient()

  const { register, handleSubmit } = useForm<SettingsForm>({
    defaultValues: {
      maxCrawlDepth: 3,
      maxPagesPerCrawl: 100,
      chunkSize: 800,
      chunkOverlap: 200,
      searchResultLimit: 10,
      vectorSearchWeight: 0.7,
      keywordSearchWeight: 0.3,
      enableMetrics: true,
      logLevel: 'INFO',
    },
  })

  // Fetch sources for scheduler configuration
  const { data: sourcesData } = useQuery({
    queryKey: ['sources'],
    queryFn: api.getSources,
  })

  const sources = sourcesData?.sources || []

  // Update source schedule mutation
  const updateSourceMutation = useMutation({
    mutationFn: ({ id, refreshInterval }: { id: string; refreshInterval: number }) =>
      api.updateSource(id, { refresh_interval: refreshInterval }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sources'] })
      setEditingSchedule(null)
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    },
  })

  const onSubmit = (data: SettingsForm) => {
    // In a real app, this would save to backend
    console.log('Saving settings:', data)
    setSaved(true)
    setTimeout(() => setSaved(false), 3000)
  }

  const handleScheduleEdit = (sourceId: string, currentInterval: number) => {
    setEditingSchedule(sourceId)
    setTempInterval(currentInterval)
  }

  const handleScheduleSave = (sourceId: string) => {
    updateSourceMutation.mutate({ id: sourceId, refreshInterval: tempInterval })
  }

  const handleScheduleCancel = () => {
    setEditingSchedule(null)
    setTempInterval(86400)
  }

  const formatInterval = (seconds: number): string => {
    const interval = REFRESH_INTERVALS.find(i => i.value === seconds)
    return interval ? interval.label : `${seconds}s`
  }

  const calculateNextRefresh = (lastRefresh: string | null, intervalSeconds: number): string => {
    if (!lastRefresh || intervalSeconds === 0) return 'Never'
    const last = new Date(lastRefresh)
    const next = new Date(last.getTime() + intervalSeconds * 1000)
    return next.toLocaleString()
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success'
      case 'pending':
        return 'warning'
      case 'error':
        return 'error'
      default:
        return 'default'
    }
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>

      <form onSubmit={handleSubmit(onSubmit)}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Crawling Configuration
                </Typography>
                <TextField
                  {...register('maxCrawlDepth', { valueAsNumber: true })}
                  label="Max Crawl Depth"
                  type="number"
                  fullWidth
                  margin="normal"
                />
                <TextField
                  {...register('maxPagesPerCrawl', { valueAsNumber: true })}
                  label="Max Pages Per Crawl"
                  type="number"
                  fullWidth
                  margin="normal"
                />
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Chunking Configuration
                </Typography>
                <TextField
                  {...register('chunkSize', { valueAsNumber: true })}
                  label="Chunk Size (tokens)"
                  type="number"
                  fullWidth
                  margin="normal"
                />
                <TextField
                  {...register('chunkOverlap', { valueAsNumber: true })}
                  label="Chunk Overlap (tokens)"
                  type="number"
                  fullWidth
                  margin="normal"
                />
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Search Configuration
                </Typography>
                <TextField
                  {...register('searchResultLimit', { valueAsNumber: true })}
                  label="Search Result Limit"
                  type="number"
                  fullWidth
                  margin="normal"
                />
                <TextField
                  {...register('vectorSearchWeight', { valueAsNumber: true })}
                  label="Vector Search Weight"
                  type="number"
                  inputProps={{ step: 0.1, min: 0, max: 1 }}
                  fullWidth
                  margin="normal"
                />
                <TextField
                  {...register('keywordSearchWeight', { valueAsNumber: true })}
                  label="Keyword Search Weight"
                  type="number"
                  inputProps={{ step: 0.1, min: 0, max: 1 }}
                  fullWidth
                  margin="normal"
                />
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  System Configuration
                </Typography>
                <FormControlLabel
                  control={
                    <Switch {...register('enableMetrics')} defaultChecked />
                  }
                  label="Enable Metrics"
                  sx={{ mt: 2, mb: 1 }}
                />
                <TextField
                  {...register('logLevel')}
                  label="Log Level"
                  select
                  fullWidth
                  margin="normal"
                  SelectProps={{
                    native: true,
                  }}
                >
                  <option value="DEBUG">DEBUG</option>
                  <option value="INFO">INFO</option>
                  <option value="WARNING">WARNING</option>
                  <option value="ERROR">ERROR</option>
                </TextField>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Scheduler Configuration
                </Typography>
                <Alert severity="info" sx={{ mb: 2 }}>
                  Configure automatic refresh schedules for your knowledge sources. 
                  The scheduler will automatically re-scrape sources based on their refresh intervals.
                </Alert>
                
                <Box sx={{ mb: 3 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={schedulerConfig.enabled}
                        onChange={(e) => setSchedulerConfig(prev => ({ ...prev, enabled: e.target.checked }))}
                      />
                    }
                    label="Enable Automatic Scheduler"
                  />
                </Box>
                
                <Grid container spacing={2} sx={{ mb: 3 }}>
                  <Grid item xs={12} md={6}>
                    <FormControl fullWidth>
                      <InputLabel>Default Refresh Interval</InputLabel>
                      <Select
                        value={schedulerConfig.defaultInterval}
                        label="Default Refresh Interval"
                        onChange={(e) => setSchedulerConfig(prev => ({ ...prev, defaultInterval: Number(e.target.value) }))}
                      >
                        {REFRESH_INTERVALS.map(interval => (
                          <MenuItem key={interval.value} value={interval.value}>
                            {interval.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      label="Check Interval (seconds)"
                      type="number"
                      value={schedulerConfig.checkInterval}
                      onChange={(e) => setSchedulerConfig(prev => ({ ...prev, checkInterval: Number(e.target.value) }))}
                      fullWidth
                      helperText="How often to check for sources that need refreshing"
                    />
                  </Grid>
                </Grid>
                
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Source Schedules
                </Typography>
                
                <TableContainer component={Paper} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Source</TableCell>
                        <TableCell>URL</TableCell>
                        <TableCell>Refresh Interval</TableCell>
                        <TableCell>Last Refresh</TableCell>
                        <TableCell>Next Refresh</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {sources.map((source: any) => (
                        <TableRow key={source.id}>
                          <TableCell>
                            <Typography variant="body2" fontWeight="medium">
                              {source.name}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 300, wordBreak: 'break-all' }}>
                              {source.url}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            {editingSchedule === source.id ? (
                              <FormControl size="small" sx={{ minWidth: 120 }}>
                                <Select
                                  value={tempInterval}
                                  onChange={(e) => setTempInterval(Number(e.target.value))}
                                >
                                  {REFRESH_INTERVALS.map(interval => (
                                    <MenuItem key={interval.value} value={interval.value}>
                                      {interval.label}
                                    </MenuItem>
                                  ))}
                                </Select>
                              </FormControl>
                            ) : (
                              <Chip
                                label={formatInterval(source.refresh_interval || 86400)}
                                size="small"
                                color={source.refresh_interval === 0 ? 'default' : 'primary'}
                              />
                            )}
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" color="text.secondary">
                              {source.last_scraped_at
                                ? new Date(source.last_scraped_at).toLocaleString()
                                : 'Never'}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" color="text.secondary">
                              {calculateNextRefresh(source.last_scraped_at, source.refresh_interval || 86400)}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={source.status}
                              size="small"
                              color={getStatusColor(source.status)}
                            />
                          </TableCell>
                          <TableCell>
                            {editingSchedule === source.id ? (
                              <Box sx={{ display: 'flex', gap: 1 }}>
                                <IconButton
                                  size="small"
                                  onClick={() => handleScheduleSave(source.id)}
                                  disabled={updateSourceMutation.isPending}
                                >
                                  <SaveIcon />
                                </IconButton>
                                <IconButton
                                  size="small"
                                  onClick={handleScheduleCancel}
                                  disabled={updateSourceMutation.isPending}
                                >
                                  <CancelIcon />
                                </IconButton>
                              </Box>
                            ) : (
                              <IconButton
                                size="small"
                                onClick={() => handleScheduleEdit(source.id, source.refresh_interval || 86400)}
                              >
                                <EditIcon />
                              </IconButton>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
                
                {sources.length === 0 && (
                  <Box sx={{ textAlign: 'center', py: 3 }}>
                    <Typography color="text.secondary">
                      No sources configured. Add sources to set up automatic refresh schedules.
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
              <Button variant="outlined" type="button">
                Reset to Defaults
              </Button>
              <Button variant="contained" type="submit">
                Save Settings
              </Button>
            </Box>
            {saved && (
              <Typography color="success.main" sx={{ mt: 2, textAlign: 'right' }}>
                Settings saved successfully!
              </Typography>
            )}
          </Grid>
        </Grid>
      </form>
    </Box>
  )
}

export default Settings