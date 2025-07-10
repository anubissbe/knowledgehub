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
  Tooltip,
  Alert,
  Button,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
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
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Stack,
  Divider,
  Avatar,
  Badge,
  Snackbar
} from '@mui/material'
import {
  Dashboard as DashboardIcon,
  Security as SecurityIcon,
  Settings as SettingsIcon,
  People as PeopleIcon,
  Storage as StorageIcon,
  Speed as SpeedIcon,
  Timeline as TimelineIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  ExpandMore as ExpandMoreIcon,
  Build as BuildIcon,
  Code as CodeIcon,
  Memory as MemoryIcon,
  CloudQueue as CloudQueueIcon,
  Assignment as AssignmentIcon,
  NetworkCheck as NetworkCheckIcon,
  Assessment as AssessmentIcon,
  BugReport as BugReportIcon,
  Schedule as ScheduleIcon,
  Notifications as NotificationsIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Info as InfoIcon,
  FileUpload as ExportIcon
} from '@mui/icons-material'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/services/api'
import { formatDistanceToNow } from 'date-fns'
import ExportManager from '@/components/ExportManager'

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
      id={`admin-tabpanel-${index}`}
      aria-labelledby={`admin-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  )
}

// Admin Dashboard Overview Component
function AdminOverview() {
  const { data: dashboard, isLoading, error, refetch } = useQuery({
    queryKey: ['admin-dashboard'],
    queryFn: async () => {
      const apiUrl = import.meta.env.VITE_API_URL || ''
      const response = await fetch(`${apiUrl}/api/v1/admin/dashboard`, {
        headers: {
          'Accept': 'application/json',
          'X-Requested-With': 'XMLHttpRequest',
          ...(localStorage.getItem('token') && {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          })
        }
      })
      if (!response.ok) {
        throw new Error('Failed to fetch admin dashboard data')
      }
      return response.json()
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  if (isLoading) return <CircularProgress />
  if (error) return <Alert severity="error">Failed to load admin dashboard</Alert>

  const getHealthColor = (score: number) => {
    if (score >= 90) return 'success'
    if (score >= 70) return 'warning'
    return 'error'
  }

  return (
    <Grid container spacing={3}>
      {/* System Overview Cards */}
      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <DashboardIcon color="primary" sx={{ mr: 1 }} />
              <Typography variant="h6">System Health</Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Typography variant="h4" color={getHealthColor(dashboard?.system_health?.overall_health)}>
                {dashboard?.system_health?.overall_health || 0}%
              </Typography>
              <Chip
                label={dashboard?.system_health?.status || 'unknown'}
                color={getHealthColor(dashboard?.system_health?.overall_health)}
                size="small"
                sx={{ ml: 1 }}
              />
            </Box>
            <LinearProgress
              variant="determinate"
              value={dashboard?.system_health?.overall_health || 0}
              color={getHealthColor(dashboard?.system_health?.overall_health)}
              sx={{ mb: 1 }}
            />
            <Typography variant="caption" color="text.secondary">
              Last check: {dashboard?.system_health?.last_check
                ? formatDistanceToNow(new Date(dashboard.system_health.last_check), { addSuffix: true })
                : 'never'}
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <PeopleIcon color="primary" sx={{ mr: 1 }} />
              <Typography variant="h6">Users</Typography>
            </Box>
            <Typography variant="h4">{dashboard?.user_summary?.total_users || 0}</Typography>
            <Typography variant="body2" color="text.secondary">
              {dashboard?.user_summary?.active_users || 0} active, {dashboard?.user_summary?.admin_users || 0} admin
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <StorageIcon color="primary" sx={{ mr: 1 }} />
              <Typography variant="h6">Storage</Typography>
            </Box>
            <Typography variant="h4">{dashboard?.system_stats?.storage_stats?.storage_usage_percent || 0}%</Typography>
            <Typography variant="body2" color="text.secondary">
              {dashboard?.system_stats?.storage_stats?.used_storage || '0'} / {dashboard?.system_stats?.storage_stats?.total_storage || '0'}
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <SpeedIcon color="primary" sx={{ mr: 1 }} />
              <Typography variant="h6">Performance</Typography>
            </Box>
            <Typography variant="h4">{dashboard?.system_stats?.performance_metrics?.avg_response_time || 'N/A'}</Typography>
            <Typography variant="body2" color="text.secondary">
              {dashboard?.system_stats?.performance_metrics?.requests_per_second || 0} req/s
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      {/* Recent Activities */}
      <Grid item xs={12} md={8}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
              <Typography variant="h6">Recent Activities</Typography>
              <IconButton size="small" onClick={() => refetch()}>
                <RefreshIcon />
              </IconButton>
            </Box>
            <List dense>
              {dashboard?.recent_activities?.slice(0, 10).map((activity: any, index: number) => (
                <ListItem key={index}>
                  <ListItemText
                    primary={activity.description}
                    secondary={formatDistanceToNow(new Date(activity.timestamp), { addSuffix: true })}
                  />
                  <Chip
                    label={activity.type}
                    size="small"
                    color={activity.type === 'error' ? 'error' : 'default'}
                  />
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      </Grid>

      {/* System Services */}
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>Service Status</Typography>
            <List dense>
              {dashboard?.system_health?.services && Object.entries(dashboard.system_health.services).map(([service, status]) => (
                <ListItem key={service}>
                  <ListItemText primary={service} />
                  <Chip
                    label={status as string}
                    size="small"
                    color={status === 'healthy' ? 'success' : 'error'}
                    icon={status === 'healthy' ? <CheckCircleIcon /> : <ErrorIcon />}
                  />
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      </Grid>

      {/* Recommendations */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>System Recommendations</Typography>
            {dashboard?.recommendations?.map((rec: any, index: number) => (
              <Alert
                key={index}
                severity={rec.priority === 'high' ? 'error' : rec.priority === 'medium' ? 'warning' : 'info'}
                sx={{ mb: 1 }}
              >
                <Typography variant="subtitle2">{rec.title}</Typography>
                <Typography variant="body2">{rec.description}</Typography>
              </Alert>
            ))}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  )
}

// User Management Component
function UserManagement() {
  const [selectedUser, setSelectedUser] = useState<any>(null)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [createUserOpen, setCreateUserOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [roleFilter, setRoleFilter] = useState('all')

  const { data: users, isLoading, refetch } = useQuery({
    queryKey: ['admin-users', searchTerm, statusFilter, roleFilter],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (searchTerm) params.append('search', searchTerm)
      if (statusFilter !== 'all') params.append('status_filter', statusFilter)
      if (roleFilter !== 'all') params.append('role_filter', roleFilter)
      
      const apiUrl = import.meta.env.VITE_API_URL || ''
      const response = await fetch(`${apiUrl}/api/v1/admin/users/management?${params}`, {
        headers: {
          'Accept': 'application/json',
          'X-Requested-With': 'XMLHttpRequest',
          ...(localStorage.getItem('token') && {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          })
        }
      })
      if (!response.ok) {
        throw new Error('Failed to fetch users')
      }
      return response.json()
    },
  })

  const handleUserAction = async (action: string, userId?: string, userData?: any) => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || ''
      const response = await fetch(`${apiUrl}/api/v1/admin/users/manage`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Requested-With': 'XMLHttpRequest',
          ...(localStorage.getItem('token') && {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          })
        },
        body: JSON.stringify({
          action,
          user_id: userId,
          user_data: userData,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to perform user action')
      }

      refetch()
      setDialogOpen(false)
      setCreateUserOpen(false)
    } catch (error) {
      console.error('User action failed:', error)
    }
  }

  if (isLoading) return <CircularProgress />

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h5">User Management</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setCreateUserOpen(true)}
        >
          Create User
        </Button>
      </Box>

      {/* Filters */}
      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <TextField
          label="Search users"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          size="small"
          sx={{ minWidth: 200 }}
        />
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Status</InputLabel>
          <Select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            label="Status"
          >
            <MenuItem value="all">All</MenuItem>
            <MenuItem value="active">Active</MenuItem>
            <MenuItem value="inactive">Inactive</MenuItem>
          </Select>
        </FormControl>
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Role</InputLabel>
          <Select
            value={roleFilter}
            onChange={(e) => setRoleFilter(e.target.value)}
            label="Role"
          >
            <MenuItem value="all">All</MenuItem>
            <MenuItem value="admin">Admin</MenuItem>
            <MenuItem value="user">User</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* User Table */}
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>User</TableCell>
              <TableCell>Email</TableCell>
              <TableCell>Role</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Last Active</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {users?.users?.map((user: any) => (
              <TableRow key={user.id}>
                <TableCell>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Avatar sx={{ mr: 2 }}>{user.name?.charAt(0)}</Avatar>
                    <Box>
                      <Typography variant="subtitle2">{user.name}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        ID: {user.id}
                      </Typography>
                    </Box>
                  </Box>
                </TableCell>
                <TableCell>{user.email}</TableCell>
                <TableCell>
                  <Chip
                    label={user.role}
                    size="small"
                    color={user.role === 'admin' ? 'primary' : 'default'}
                  />
                </TableCell>
                <TableCell>
                  <Chip
                    label={user.status}
                    size="small"
                    color={user.status === 'active' ? 'success' : 'error'}
                  />
                </TableCell>
                <TableCell>
                  {user.last_active ? formatDistanceToNow(new Date(user.last_active), { addSuffix: true }) : 'Never'}
                </TableCell>
                <TableCell>
                  <IconButton
                    size="small"
                    onClick={() => {
                      setSelectedUser(user)
                      setDialogOpen(true)
                    }}
                  >
                    <EditIcon />
                  </IconButton>
                  <IconButton
                    size="small"
                    color="error"
                    onClick={() => handleUserAction('delete', user.id)}
                  >
                    <DeleteIcon />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* User Details/Edit Dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Edit User</DialogTitle>
        <DialogContent>
          {selectedUser && (
            <Box sx={{ pt: 2 }}>
              <TextField
                fullWidth
                label="Name"
                value={selectedUser.name}
                onChange={(e) => setSelectedUser({ ...selectedUser, name: e.target.value })}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Email"
                value={selectedUser.email}
                onChange={(e) => setSelectedUser({ ...selectedUser, email: e.target.value })}
                sx={{ mb: 2 }}
              />
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Role</InputLabel>
                <Select
                  value={selectedUser.role}
                  onChange={(e) => setSelectedUser({ ...selectedUser, role: e.target.value })}
                  label="Role"
                >
                  <MenuItem value="user">User</MenuItem>
                  <MenuItem value="admin">Admin</MenuItem>
                </Select>
              </FormControl>
              <FormControlLabel
                control={
                  <Switch
                    checked={selectedUser.status === 'active'}
                    onChange={(e) => setSelectedUser({ ...selectedUser, status: e.target.checked ? 'active' : 'inactive' })}
                  />
                }
                label="Active"
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={() => handleUserAction('update', selectedUser?.id, selectedUser)}
            variant="contained"
          >
            Save Changes
          </Button>
        </DialogActions>
      </Dialog>

      {/* Create User Dialog */}
      <Dialog open={createUserOpen} onClose={() => setCreateUserOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create New User</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField
              fullWidth
              label="Name"
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              label="Email"
              type="email"
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              label="Password"
              type="password"
              sx={{ mb: 2 }}
            />
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Role</InputLabel>
              <Select
                value="user"
                label="Role"
              >
                <MenuItem value="user">User</MenuItem>
                <MenuItem value="admin">Admin</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateUserOpen(false)}>Cancel</Button>
          <Button
            onClick={() => handleUserAction('create')}
            variant="contained"
          >
            Create User
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

// System Configuration Component
function SystemConfiguration() {
  const [configSection, setConfigSection] = useState('')
  const [snackbarOpen, setSnackbarOpen] = useState(false)

  const { data: config, isLoading, refetch } = useQuery({
    queryKey: ['admin-config'],
    queryFn: async () => {
      const apiUrl = import.meta.env.VITE_API_URL || ''
      const response = await fetch(`${apiUrl}/api/v1/admin/system/configuration`, {
        headers: {
          'Accept': 'application/json',
          'X-Requested-With': 'XMLHttpRequest',
          ...(localStorage.getItem('token') && {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          })
        }
      })
      if (!response.ok) {
        throw new Error('Failed to fetch configuration')
      }
      return response.json()
    },
  })

  const handleConfigUpdate = async (section: string, key: string, value: any) => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || ''
      const response = await fetch(`${apiUrl}/api/v1/admin/system/configuration`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Requested-With': 'XMLHttpRequest',
          ...(localStorage.getItem('token') && {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          })
        },
        body: JSON.stringify({
          section,
          key,
          value,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to update configuration')
      }

      refetch()
      setSnackbarOpen(true)
    } catch (error) {
      console.error('Configuration update failed:', error)
    }
  }

  if (isLoading) return <CircularProgress />

  return (
    <Box>
      <Typography variant="h5" gutterBottom>System Configuration</Typography>
      
      {config?.configuration && Object.entries(config.configuration).map(([sectionName, sectionData]) => (
        <Accordion key={sectionName} sx={{ mb: 1 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6" sx={{ textTransform: 'capitalize' }}>
              {sectionName} Configuration
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              {Object.entries(sectionData as any).map(([key, value]) => (
                <Grid item xs={12} sm={6} key={key}>
                  <TextField
                    fullWidth
                    label={key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    value={value as string}
                    onChange={(e) => handleConfigUpdate(sectionName, key, e.target.value)}
                    variant="outlined"
                    size="small"
                  />
                </Grid>
              ))}
            </Grid>
          </AccordionDetails>
        </Accordion>
      ))}

      <Snackbar
        open={snackbarOpen}
        autoHideDuration={3000}
        onClose={() => setSnackbarOpen(false)}
        message="Configuration updated successfully"
      />
    </Box>
  )
}

// System Tools Component
function SystemTools() {
  const [commandOutput, setCommandOutput] = useState<string>('')
  const [isExecuting, setIsExecuting] = useState(false)

  const executeCommand = async (command: string) => {
    setIsExecuting(true)
    setCommandOutput('')

    try {
      const apiUrl = import.meta.env.VITE_API_URL || ''
      const response = await fetch(`${apiUrl}/api/v1/admin/system/command`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Requested-With': 'XMLHttpRequest',  // Required for CSRF exemption
          // Add auth token if available
          ...(localStorage.getItem('token') && {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          })
        },
        body: JSON.stringify({
          command,
          confirm: true,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to execute command')
      }

      const result = await response.json()
      setCommandOutput(JSON.stringify(result, null, 2))
    } catch (error) {
      setCommandOutput(`Error: ${error}`)
    } finally {
      setIsExecuting(false)
    }
  }

  const systemCommands = [
    { name: 'Clear Cache', command: 'clear_cache', description: 'Clear application cache' },
    { name: 'Restart Services', command: 'restart_services', description: 'Restart background services' },
    { name: 'Backup Database', command: 'backup_database', description: 'Create database backup' },
    { name: 'Cleanup Logs', command: 'cleanup_logs', description: 'Clean up old log files' },
    { name: 'Update Indexes', command: 'update_indexes', description: 'Update database indexes' },
    { name: 'Vacuum Database', command: 'vacuum_database', description: 'Run VACUUM to reclaim storage space' },
    { name: 'Analyze Database', command: 'analyze_database', description: 'Run ANALYZE to update statistics' },
    { name: 'Full DB Maintenance', command: 'full_db_maintenance', description: 'Run VACUUM ANALYZE for complete optimization' },
    { name: 'Health Check', command: 'health_check', description: 'Run comprehensive health check' },
  ]

  return (
    <Box>
      <Typography variant="h5" gutterBottom>System Tools</Typography>
      
      <Grid container spacing={2}>
        {systemCommands.map((cmd) => (
          <Grid item xs={12} sm={6} md={4} key={cmd.command}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <BuildIcon color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h6">{cmd.name}</Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {cmd.description}
                </Typography>
                <Button
                  fullWidth
                  variant="contained"
                  onClick={() => executeCommand(cmd.command)}
                  disabled={isExecuting}
                  startIcon={isExecuting ? <CircularProgress size={20} /> : <BuildIcon />}
                >
                  Execute
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {commandOutput && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Command Output</Typography>
            <Box
              component="pre"
              sx={{
                bgcolor: 'grey.100',
                p: 2,
                borderRadius: 1,
                overflow: 'auto',
                fontFamily: 'monospace',
                fontSize: '0.875rem',
              }}
            >
              {commandOutput}
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  )
}

// Main Admin Dashboard Component
function AdminDashboard() {
  const [currentTab, setCurrentTab] = useState(0)

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue)
  }

  return (
    <Box sx={{ width: '100%' }}>
      <Typography variant="h4" gutterBottom>
        Admin Dashboard
      </Typography>
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={currentTab} onChange={handleTabChange} aria-label="admin dashboard tabs">
          <Tab
            label="Overview"
            icon={<DashboardIcon />}
            iconPosition="start"
          />
          <Tab
            label="User Management"
            icon={<PeopleIcon />}
            iconPosition="start"
          />
          <Tab
            label="Configuration"
            icon={<SettingsIcon />}
            iconPosition="start"
          />
          <Tab
            label="System Tools"
            icon={<BuildIcon />}
            iconPosition="start"
          />
          <Tab
            label="Export Data"
            icon={<ExportIcon />}
            iconPosition="start"
          />
        </Tabs>
      </Box>

      <TabPanel value={currentTab} index={0}>
        <AdminOverview />
      </TabPanel>

      <TabPanel value={currentTab} index={1}>
        <UserManagement />
      </TabPanel>

      <TabPanel value={currentTab} index={2}>
        <SystemConfiguration />
      </TabPanel>

      <TabPanel value={currentTab} index={3}>
        <SystemTools />
      </TabPanel>

      <TabPanel value={currentTab} index={4}>
        <Box>
          <Typography variant="h5" gutterBottom>
            Data Export
          </Typography>
          <Typography variant="body1" color="textSecondary" paragraph>
            Export system data in various formats for backup, analysis, or migration purposes.
          </Typography>
          
          <ExportManager />
        </Box>
      </TabPanel>
    </Box>
  )
}

export default AdminDashboard