import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Grid,
  Chip,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  LinearProgress,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tooltip,
  Snackbar,
  CircularProgress,
  Stack,
  Divider
} from '@mui/material'
import {
  Download as DownloadIcon,
  FileUpload as ExportIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  ExpandMore as ExpandMoreIcon,
  Info as InfoIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Schedule as ScheduleIcon,
  Analytics as AnalyticsIcon,
  Storage as StorageIcon,
  Description as DocumentIcon,
  Memory as MemoryIcon,
  Security as SecurityIcon,
  Dashboard as DashboardIcon
} from '@mui/icons-material'

interface ExportFormat {
  name: string
  description: string
  file_extension: string
  mime_type: string
  supports_complex_data: boolean
}

interface ExportType {
  name: string
  description: string
  estimated_records: string
  common_filters: string[]
}

interface ExportResult {
  export_id: string
  export_type: string
  export_format: string
  status: string
  file_path?: string
  file_size?: number
  record_count: number
  created_at: string
  download_url?: string
  expires_at?: string
  metadata: Record<string, any>
}

interface QuickExportOption {
  id: string
  name: string
  description: string
  export_type: string
  format: string
  record_count: number | string
  filters: Record<string, any>
}

interface ExportTemplate {
  name: string
  description: string
  request: Record<string, any>
}

const ExportManager: React.FC = () => {
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [formats, setFormats] = useState<Record<string, ExportFormat>>({})
  const [types, setTypes] = useState<Record<string, ExportType>>({})
  const [quickOptions, setQuickOptions] = useState<QuickExportOption[]>([])
  const [templates, setTemplates] = useState<Record<string, ExportTemplate>>({})
  const [recentExports, setRecentExports] = useState<ExportResult[]>([])
  const [stats, setStats] = useState<any>(null)
  
  // Form state
  const [selectedType, setSelectedType] = useState('')
  const [selectedFormat, setSelectedFormat] = useState('')
  const [includeMetadata, setIncludeMetadata] = useState(true)
  const [includeStatistics, setIncludeStatistics] = useState(false)
  const [recordLimit, setRecordLimit] = useState('')
  const [filters, setFilters] = useState<Record<string, string>>({})
  const [selectedFields, setSelectedFields] = useState<string[]>([])
  
  // UI state
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' as 'success' | 'error' | 'warning' | 'info' })
  const [activeTab, setActiveTab] = useState('quick')
  
  useEffect(() => {
    if (open) {
      loadExportData()
    }
  }, [open])
  
  const loadExportData = async () => {
    try {
      setLoading(true)
      
      // Load formats, types, quick options, templates, and stats in parallel
      const [formatsRes, typesRes, quickRes, templatesRes, statsRes] = await Promise.all([
        fetch('/api/v1/exports/formats'),
        fetch('/api/v1/exports/types'),
        fetch('/api/v1/exports/quick-exports'),
        fetch('/api/v1/exports/templates'),
        fetch('/api/v1/exports/stats')
      ])
      
      if (formatsRes.ok) {
        const data = await formatsRes.json()
        setFormats(data.formats)
      }
      
      if (typesRes.ok) {
        const data = await typesRes.json()
        setTypes(data.types)
      }
      
      if (quickRes.ok) {
        const data = await quickRes.json()
        setQuickOptions(data.quick_exports)
      }
      
      if (templatesRes.ok) {
        const data = await templatesRes.json()
        setTemplates(data.templates)
      }
      
      if (statsRes.ok) {
        const data = await statsRes.json()
        setStats(data)
      }
      
    } catch (error) {
      console.error('Failed to load export data:', error)
      showSnackbar('Failed to load export data', 'error')
    } finally {
      setLoading(false)
    }
  }
  
  const showSnackbar = (message: string, severity: 'success' | 'error' | 'warning' | 'info') => {
    setSnackbar({ open: true, message, severity })
  }
  
  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false })
  }
  
  const createCustomExport = async () => {
    if (!selectedType || !selectedFormat) {
      showSnackbar('Please select export type and format', 'warning')
      return
    }
    
    try {
      setLoading(true)
      
      const exportRequest = {
        export_type: selectedType,
        export_format: selectedFormat,
        include_metadata: includeMetadata,
        include_statistics: includeStatistics,
        limit: recordLimit ? parseInt(recordLimit) : undefined,
        filters: Object.keys(filters).length > 0 ? filters : undefined,
        fields: selectedFields.length > 0 ? selectedFields : undefined
      }
      
      const response = await fetch('/api/v1/exports/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(exportRequest)
      })
      
      if (response.ok) {
        const result: ExportResult = await response.json()
        showSnackbar(`Export created successfully! ID: ${result.export_id}`, 'success')
        
        // Add to recent exports
        setRecentExports(prev => [result, ...prev])
        
        // Start download
        downloadExport(result.export_id)
      } else {
        const error = await response.json()
        showSnackbar(`Export failed: ${error.detail}`, 'error')
      }
      
    } catch (error) {
      console.error('Export creation failed:', error)
      showSnackbar('Export creation failed', 'error')
    } finally {
      setLoading(false)
    }
  }
  
  const createQuickExport = async (optionId: string) => {
    try {
      setLoading(true)
      
      const response = await fetch(`/api/v1/exports/quick/${optionId}`, {
        method: 'POST'
      })
      
      if (response.ok) {
        const result: ExportResult = await response.json()
        showSnackbar(`Quick export created! ID: ${result.export_id}`, 'success')
        
        // Add to recent exports
        setRecentExports(prev => [result, ...prev])
        
        // Start download
        downloadExport(result.export_id)
      } else {
        const error = await response.json()
        showSnackbar(`Quick export failed: ${error.detail}`, 'error')
      }
      
    } catch (error) {
      console.error('Quick export failed:', error)
      showSnackbar('Quick export failed', 'error')
    } finally {
      setLoading(false)
    }
  }
  
  const createTemplateExport = async (templateId: string) => {
    try {
      setLoading(true)
      
      const response = await fetch(`/api/v1/exports/templates/${templateId}`, {
        method: 'POST'
      })
      
      if (response.ok) {
        const result: ExportResult = await response.json()
        showSnackbar(`Template export created! ID: ${result.export_id}`, 'success')
        
        // Add to recent exports
        setRecentExports(prev => [result, ...prev])
        
        // Start download
        downloadExport(result.export_id)
      } else {
        const error = await response.json()
        showSnackbar(`Template export failed: ${error.detail}`, 'error')
      }
      
    } catch (error) {
      console.error('Template export failed:', error)
      showSnackbar('Template export failed', 'error')
    } finally {
      setLoading(false)
    }
  }
  
  const downloadExport = async (exportId: string) => {
    try {
      window.open(`/api/v1/exports/${exportId}/download`, '_blank')
    } catch (error) {
      console.error('Download failed:', error)
      showSnackbar('Download failed', 'error')
    }
  }
  
  const deleteExport = async (exportId: string) => {
    try {
      const response = await fetch(`/api/v1/exports/${exportId}`, {
        method: 'DELETE'
      })
      
      if (response.ok) {
        showSnackbar('Export deleted successfully', 'success')
        setRecentExports(prev => prev.filter(exp => exp.export_id !== exportId))
      } else {
        const error = await response.json()
        showSnackbar(`Delete failed: ${error.detail}`, 'error')
      }
      
    } catch (error) {
      console.error('Delete failed:', error)
      showSnackbar('Delete failed', 'error')
    }
  }
  
  const formatFileSize = (bytes?: number) => {
    if (!bytes) return 'Unknown'
    
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
  }
  
  const getTypeIcon = (type: string) => {
    const iconMap: Record<string, React.ReactElement> = {
      sources: <StorageIcon />,
      documents: <DocumentIcon />,
      jobs: <ScheduleIcon />,
      memories: <MemoryIcon />,
      analytics: <AnalyticsIcon />,
      system_data: <DashboardIcon />,
      all: <ExportIcon />
    }
    return iconMap[type] || <ExportIcon />
  }
  
  const renderQuickExports = () => (
    <Box>
      <Typography variant="h6" gutterBottom>
        Quick Export Options
      </Typography>
      <Typography variant="body2" color="textSecondary" paragraph>
        Pre-configured exports for common use cases
      </Typography>
      
      <Grid container spacing={2}>
        {quickOptions.map((option) => (
          <Grid item xs={12} md={6} key={option.id}>
            <Card variant="outlined">
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  {getTypeIcon(option.export_type)}
                  <Typography variant="h6" sx={{ ml: 1 }}>
                    {option.name}
                  </Typography>
                </Box>
                
                <Typography variant="body2" color="textSecondary" paragraph>
                  {option.description}
                </Typography>
                
                <Box display="flex" gap={1} mb={2}>
                  <Chip 
                    label={option.format.toUpperCase()} 
                    size="small" 
                    variant="outlined" 
                  />
                  <Chip 
                    label={`${option.record_count} records`} 
                    size="small" 
                    color="primary" 
                    variant="outlined" 
                  />
                </Box>
                
                <Button
                  variant="contained"
                  startIcon={<DownloadIcon />}
                  onClick={() => createQuickExport(option.id)}
                  disabled={loading}
                  fullWidth
                >
                  Export Now
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  )
  
  const renderTemplates = () => (
    <Box>
      <Typography variant="h6" gutterBottom>
        Export Templates
      </Typography>
      <Typography variant="body2" color="textSecondary" paragraph>
        Professional export templates for specific use cases
      </Typography>
      
      <Grid container spacing={2}>
        {Object.entries(templates).map(([templateId, template]) => (
          <Grid item xs={12} md={6} key={templateId}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {template.name}
                </Typography>
                
                <Typography variant="body2" color="textSecondary" paragraph>
                  {template.description}
                </Typography>
                
                <Box display="flex" gap={1} mb={2}>
                  <Chip 
                    label={template.request.export_type?.toUpperCase() || 'MIXED'} 
                    size="small" 
                    variant="outlined" 
                  />
                  <Chip 
                    label={template.request.export_format?.toUpperCase() || 'ZIP'} 
                    size="small" 
                    color="secondary" 
                    variant="outlined" 
                  />
                </Box>
                
                <Button
                  variant="contained"
                  startIcon={<DownloadIcon />}
                  onClick={() => createTemplateExport(templateId)}
                  disabled={loading}
                  fullWidth
                >
                  Use Template
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  )
  
  const renderCustomExport = () => (
    <Box>
      <Typography variant="h6" gutterBottom>
        Custom Export
      </Typography>
      <Typography variant="body2" color="textSecondary" paragraph>
        Create a custom export with specific filters and options
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel>Export Type</InputLabel>
            <Select
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
              label="Export Type"
            >
              {Object.entries(types).map(([key, type]) => (
                <MenuItem key={key} value={key}>
                  <Box display="flex" alignItems="center">
                    {getTypeIcon(key)}
                    <Box ml={1}>
                      <Typography variant="body2">{type.name}</Typography>
                      <Typography variant="caption" color="textSecondary">
                        {type.description}
                      </Typography>
                    </Box>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel>Export Format</InputLabel>
            <Select
              value={selectedFormat}
              onChange={(e) => setSelectedFormat(e.target.value)}
              label="Export Format"
            >
              {Object.entries(formats).map(([key, format]) => (
                <MenuItem key={key} value={key}>
                  <Box>
                    <Typography variant="body2">{format.name}</Typography>
                    <Typography variant="caption" color="textSecondary">
                      {format.description}
                    </Typography>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Record Limit (optional)"
            type="number"
            value={recordLimit}
            onChange={(e) => setRecordLimit(e.target.value)}
            helperText="Maximum 10,000 records"
            inputProps={{ max: 10000 }}
          />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Box>
            <FormControlLabel
              control={
                <Switch
                  checked={includeMetadata}
                  onChange={(e) => setIncludeMetadata(e.target.checked)}
                />
              }
              label="Include Metadata"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={includeStatistics}
                  onChange={(e) => setIncludeStatistics(e.target.checked)}
                />
              }
              label="Include Statistics"
            />
          </Box>
        </Grid>
        
        <Grid item xs={12}>
          <Button
            variant="contained"
            startIcon={<ExportIcon />}
            onClick={createCustomExport}
            disabled={loading || !selectedType || !selectedFormat}
            size="large"
          >
            {loading ? 'Creating Export...' : 'Create Export'}
          </Button>
        </Grid>
      </Grid>
    </Box>
  )
  
  const renderStats = () => (
    <Box mb={3}>
      <Typography variant="h6" gutterBottom>
        Export Statistics
      </Typography>
      
      {stats && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" color="primary">
                  {Object.values(stats.available_records || {}).reduce((a: number, b: any) => a + (typeof b === 'number' ? b : 0), 0).toLocaleString()}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Total Records Available
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" color="secondary">
                  {stats.current_exports?.active_files || 0}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Active Export Files
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" color="info.main">
                  {stats.current_exports?.total_size_mb?.toFixed(1) || 0} MB
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Storage Used
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  )
  
  return (
    <>
      <Button
        variant="contained"
        startIcon={<ExportIcon />}
        onClick={() => setOpen(true)}
        color="primary"
      >
        Export Data
      </Button>
      
      <Dialog 
        open={open} 
        onClose={() => setOpen(false)}
        maxWidth="lg"
        fullWidth
        PaperProps={{
          sx: { height: '90vh' }
        }}
      >
        <DialogTitle>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Box display="flex" alignItems="center">
              <ExportIcon sx={{ mr: 1 }} />
              <Typography variant="h6">Export Data</Typography>
            </Box>
            <Box>
              <Button
                size="small"
                startIcon={<RefreshIcon />}
                onClick={loadExportData}
                disabled={loading}
              >
                Refresh
              </Button>
            </Box>
          </Box>
        </DialogTitle>
        
        <DialogContent dividers>
          {loading && <LinearProgress sx={{ mb: 2 }} />}
          
          {renderStats()}
          
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
            <Box display="flex" gap={2}>
              <Button
                variant={activeTab === 'quick' ? 'contained' : 'text'}
                onClick={() => setActiveTab('quick')}
              >
                Quick Exports
              </Button>
              <Button
                variant={activeTab === 'templates' ? 'contained' : 'text'}
                onClick={() => setActiveTab('templates')}
              >
                Templates
              </Button>
              <Button
                variant={activeTab === 'custom' ? 'contained' : 'text'}
                onClick={() => setActiveTab('custom')}
              >
                Custom Export
              </Button>
            </Box>
          </Box>
          
          {activeTab === 'quick' && renderQuickExports()}
          {activeTab === 'templates' && renderTemplates()}
          {activeTab === 'custom' && renderCustomExport()}
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setOpen(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
      
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </>
  )
}

export default ExportManager