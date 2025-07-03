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
} from '@mui/material'
import {
  Refresh as RefreshIcon,
  Cancel as CancelIcon,
} from '@mui/icons-material'
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/services/api'

function Jobs() {
  const [page, setPage] = useState(0)
  const [rowsPerPage, setRowsPerPage] = useState(10)
  const queryClient = useQueryClient()

  const { data: jobsResponse, isLoading } = useQuery({
    queryKey: ['jobs', page, rowsPerPage],
    queryFn: () => api.getJobs(),
  })
  
  const jobs = Array.isArray(jobsResponse?.jobs) ? jobsResponse.jobs : []
  const totalJobs = jobsResponse?.total || 0

  const retryMutation = useMutation({
    mutationFn: api.retryJob,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
  })

  const cancelMutation = useMutation({
    mutationFn: api.cancelJob,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
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
      <Typography variant="h4" gutterBottom>
        Jobs
      </Typography>

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
                    label={job.status}
                    size="small"
                    color={getStatusColor(job.status)}
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
                  {(job.status === 'pending' || job.status === 'running') && (
                    <IconButton
                      size="small"
                      onClick={() => cancelMutation.mutate(job.id)}
                      disabled={cancelMutation.isPending}
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