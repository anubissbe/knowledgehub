import { useState } from 'react'
import {
  Box,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  Chip,
  Grid,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Paper,
} from '@mui/material'
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Search as SearchIcon,
} from '@mui/icons-material'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useForm } from 'react-hook-form'
import { api } from '@/services/api'

function MemoryPage() {
  const [openDialog, setOpenDialog] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const queryClient = useQueryClient()

  const { data: memories, isLoading } = useQuery({
    queryKey: ['memories', searchQuery],
    queryFn: async () => {
      const result = await api.getMemories(searchQuery ? { search: searchQuery } : undefined)
      return Array.isArray(result) ? result : []
    },
  })

  const createMutation = useMutation({
    mutationFn: ({ content, tags }: { content: string; tags: string[] }) =>
      api.createMemory({ content, tags }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memories'] })
      setOpenDialog(false)
      reset()
    },
  })

  const deleteMutation = useMutation({
    mutationFn: api.deleteMemory,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memories'] })
    },
  })

  const { register, handleSubmit, reset } = useForm<{
    content: string
    tags: string
  }>()

  const onSubmit = (data: { content: string; tags: string }) => {
    const tags = data.tags.split(',').map(t => t.trim()).filter(t => t)
    createMutation.mutate({ content: data.content, tags })
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString()
  }

  if (isLoading) {
    return <Typography>Loading...</Typography>
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4">Memory System</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setOpenDialog(true)}
        >
          Add Memory
        </Button>
      </Box>

      <Box sx={{ mb: 3 }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Search memories..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          InputProps={{
            startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
          }}
        />
      </Box>

      <Grid container spacing={2}>
        {memories?.map((memory) => (
          <Grid item xs={12} md={6} key={memory.id}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    {formatDate(memory.created_at)} • Accessed {memory.access_count} times
                  </Typography>
                  <IconButton
                    size="small"
                    onClick={() => deleteMutation.mutate(memory.id)}
                    disabled={deleteMutation.isPending}
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </Box>
                <Typography variant="body2" paragraph>
                  {memory.content}
                </Typography>
                <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                  {Array.isArray(memory.tags) && memory.tags.map((tag: string, index: number) => (
                    <Chip key={index} label={tag} size="small" />
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {memories?.length === 0 && (
        <Paper sx={{ p: 3, textAlign: 'center' }}>
          <Typography color="text.secondary">
            No memories found. Create your first memory to get started.
          </Typography>
        </Paper>
      )}

      <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="sm" fullWidth>
        <form onSubmit={handleSubmit(onSubmit)}>
          <DialogTitle>Add New Memory</DialogTitle>
          <DialogContent>
            <TextField
              {...register('content', { required: true })}
              label="Content"
              multiline
              rows={4}
              fullWidth
              margin="normal"
              required
            />
            <TextField
              {...register('tags')}
              label="Tags (comma-separated)"
              fullWidth
              margin="normal"
              placeholder="tag1, tag2, tag3"
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
            <Button
              type="submit"
              variant="contained"
              disabled={createMutation.isPending}
            >
              Create
            </Button>
          </DialogActions>
        </form>
      </Dialog>
    </Box>
  )
}

export default MemoryPage