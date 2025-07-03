import { useState } from 'react'
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  CardActions,
  Grid,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  MenuItem,
  CircularProgress,
} from '@mui/material'
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
} from '@mui/icons-material'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useForm } from 'react-hook-form'
import { api } from '@/services/api'
import { Source, SourceCreate } from '@/types'

function Sources() {
  const [openDialog, setOpenDialog] = useState(false)
  const [editingSource, setEditingSource] = useState<Source | null>(null)
  const queryClient = useQueryClient()

  const { data: sourcesResponse, isLoading } = useQuery({
    queryKey: ['sources'],
    queryFn: api.getSources,
  })
  
  const sources = Array.isArray(sourcesResponse?.sources) ? sourcesResponse.sources : []

  const createMutation = useMutation({
    mutationFn: api.createSource,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sources'] })
      setOpenDialog(false)
      reset()
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<SourceCreate> }) =>
      api.updateSource(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sources'] })
      setOpenDialog(false)
      setEditingSource(null)
      reset()
    },
  })

  const deleteMutation = useMutation({
    mutationFn: api.deleteSource,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sources'] })
    },
  })

  const refreshMutation = useMutation({
    mutationFn: api.refreshSource,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sources'] })
    },
  })

  const { register, handleSubmit, reset, setValue } = useForm<SourceCreate>({
    defaultValues: {
      type: 'website',
      refresh_interval: 86400, // 24 hours
    },
  })

  const handleEdit = (source: Source) => {
    setEditingSource(source)
    setValue('url', source.url)
    setValue('name', source.name)
    setValue('type', source.type)
    setValue('refresh_interval', source.refresh_interval)
    setOpenDialog(true)
  }

  const handleClose = () => {
    setOpenDialog(false)
    setEditingSource(null)
    reset()
  }

  const onSubmit = (data: SourceCreate) => {
    if (editingSource) {
      updateMutation.mutate({ id: editingSource.id, data })
    } else {
      createMutation.mutate(data)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success'
      case 'crawling':
      case 'indexing':
        return 'warning'
      case 'error':
        return 'error'
      default:
        return 'default'
    }
  }

  if (isLoading) {
    return <CircularProgress />
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4">Knowledge Sources</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setOpenDialog(true)}
        >
          Add Source
        </Button>
      </Box>

      <Grid container spacing={3}>
        {sources.map((source) => (
          <Grid item xs={12} sm={6} md={4} key={source.id}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {source.name}
                </Typography>
                <Typography
                  variant="body2"
                  color="text.secondary"
                  gutterBottom
                  sx={{ wordBreak: 'break-all' }}
                >
                  {source.url}
                </Typography>
                <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                  <Chip
                    label={source.status}
                    size="small"
                    color={getStatusColor(source.status)}
                  />
                  <Chip label={source.type} size="small" />
                </Box>
                {source.statistics && (
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                    Pages: {source.statistics.total_pages || 0} | 
                    Chunks: {source.statistics.total_chunks || 0}
                  </Typography>
                )}
              </CardContent>
              <CardActions>
                <IconButton
                  size="small"
                  onClick={() => refreshMutation.mutate(source.id)}
                  disabled={refreshMutation.isPending}
                >
                  <RefreshIcon />
                </IconButton>
                <IconButton size="small" onClick={() => handleEdit(source)}>
                  <EditIcon />
                </IconButton>
                <IconButton
                  size="small"
                  onClick={() => deleteMutation.mutate(source.id)}
                  disabled={deleteMutation.isPending}
                >
                  <DeleteIcon />
                </IconButton>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Dialog open={openDialog} onClose={handleClose} maxWidth="sm" fullWidth>
        <form onSubmit={handleSubmit(onSubmit)}>
          <DialogTitle>
            {editingSource ? 'Edit Source' : 'Add New Source'}
          </DialogTitle>
          <DialogContent>
            <TextField
              {...register('url', { required: true })}
              label="URL"
              fullWidth
              margin="normal"
              required
            />
            <TextField
              {...register('name', { required: true })}
              label="Name"
              fullWidth
              margin="normal"
              required
            />
            <TextField
              {...register('type')}
              label="Type"
              select
              fullWidth
              margin="normal"
              defaultValue="website"
            >
              <MenuItem value="website">Website</MenuItem>
              <MenuItem value="documentation">Documentation</MenuItem>
              <MenuItem value="github">GitHub</MenuItem>
              <MenuItem value="api">API</MenuItem>
            </TextField>
            <TextField
              {...register('refresh_interval', { valueAsNumber: true })}
              label="Refresh Interval (seconds)"
              type="number"
              fullWidth
              margin="normal"
              defaultValue={86400}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={handleClose}>Cancel</Button>
            <Button
              type="submit"
              variant="contained"
              disabled={createMutation.isPending || updateMutation.isPending}
            >
              {editingSource ? 'Update' : 'Create'}
            </Button>
          </DialogActions>
        </form>
      </Dialog>
    </Box>
  )
}

export default Sources