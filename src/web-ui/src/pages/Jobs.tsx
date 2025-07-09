import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  TablePagination,
  Tooltip,
  CircularProgress,
} from '@mui/material'
import {
  Refresh as RefreshIcon,
  Cancel as CancelIcon,
  Sync as SyncIcon,
} from '@mui/icons-material'
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/services/api'
import { useWebSocket } from '@/contexts/WebSocketContext'

function Jobs() {
  const [page, setPage] = useState(0)
  const [rowsPerPage, setRowsPerPage] = useState(10)
  const [cancellingJobs, setCancellingJobs] = useState<Set<string>>(new Set())
  const queryClient = useQueryClient()
  const { isConnected } = useWebSocket()

  const { data: jobsResponse, isLoading, refetch, isRefetching } = useQuery({
    queryKey: ['jobs', page, rowsPerPage],
    queryFn: () => api.getJobs(),
    refetchInterval: 5000, // Refresh every 5 seconds for more frequent job updates
    refetchIntervalInBackground: true, // Continue refreshing even when tab is not active
  })
  
  const jobs = Array.isArray(jobsResponse?.jobs) ? jobsResponse.jobs : []
  const totalJobs = jobsResponse?.total || 0

  const retryMutation = useMutation({
    mutationFn: api.retryJob,
    onSuccess: () => {
      // Invalidate multiple queries to ensure UI updates
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] })
      queryClient.invalidateQueries({ queryKey: ['sources'] })
    },
  })

  const cancelMutation = useMutation({
    mutationFn: (jobId: string) => {
      setCancellingJobs(prev => new Set(prev).add(jobId))
      return api.cancelJob(jobId)
    },
    onSuccess: (_, jobId) => {
      // Remove from cancelling set
      setCancellingJobs(prev => {
        const newSet = new Set(prev)
        newSet.delete(jobId)
        return newSet
      })
      // Invalidate multiple queries to ensure UI updates
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] })
      queryClient.invalidateQueries({ queryKey: ['sources'] })
    },
    onError: (_, jobId) => {
      // Remove from cancelling set on error too
      setCancellingJobs(prev => {
        const newSet = new Set(prev)
        newSet.delete(jobId)
        return newSet
      })
    },
  })

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success'
      case 'running':
        return 'warning'
      case 'failed':
        return 'error'
      case 'cancelled':
        return 'default'
      default:
        return 'info'
    }
  }

  const formatDuration = (seconds?: number) => {
    if (!seconds) return '-'
    if (seconds < 60) return `${seconds}s`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString()
  }

  const handleChangePage = (_event: unknown, newPage: number) => {
    setPage(newPage)
  }

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10))
    setPage(0)
  }

  if (isLoading) {
    return <Typography>Loading...</Typography>
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" sx={{ flexGrow: 1 }}>
          Jobs
        </Typography>
        {isConnected && (
          <Chip 
            label="Live" 
            color="success" 
            size="small" 
            sx={{ mr: 2 }}
          />
        )}
        <Tooltip title="Refresh jobs">
          <IconButton 
            onClick={() => refetch()} 
            disabled={isRefetching}
            color="primary"
          >
            <SyncIcon />
          </IconButton>
        </Tooltip>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Type</TableCell>
              <TableCell>Source</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Created</TableCell>
              <TableCell>Duration</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {jobs.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((job) => (
              <TableRow key={job.id}>
                <TableCell>{job.type}</TableCell>
                <TableCell>
                  <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                    {job.source_id.slice(0, 8)}...
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip
                    label={cancellingJobs.has(job.id) ? 'cancelling...' : job.status}
                    size="small"
                    color={cancellingJobs.has(job.id) ? 'warning' : getStatusColor(job.status)}
                    icon={cancellingJobs.has(job.id) ? <CircularProgress size={16} /> : undefined}
                  />
                </TableCell>
                <TableCell>{formatDate(job.created_at)}</TableCell>
                <TableCell>{formatDuration(job.duration)}</TableCell>
                <TableCell>
                  {job.status === 'failed' && (
                    <IconButton
                      size="small"
                      onClick={() => retryMutation.mutate(job.id)}
                      disabled={retryMutation.isPending}
                    >
                      <RefreshIcon />
                    </IconButton>
                  )}
                  {(job.status === 'pending' || job.status === 'running') && !cancellingJobs.has(job.id) && (
                    <IconButton
                      size="small"
                      onClick={() => cancelMutation.mutate(job.id)}
                      disabled={cancelMutation.isPending || cancellingJobs.has(job.id)}
                    >
                      <CancelIcon />
                    </IconButton>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <TablePagination
          rowsPerPageOptions={[5, 10, 25]}
          component="div"
          count={totalJobs}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </TableContainer>
    </Box>
  )
}

export default Jobs