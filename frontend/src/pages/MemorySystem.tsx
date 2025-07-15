import { useEffect, useState } from 'react'
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  Chip,
  Grid,
  Card,
  CardContent,
  CircularProgress,
} from '@mui/material'
import { DataGrid, GridColDef } from '@mui/x-data-grid'
import { api } from '../services/api'

interface Memory {
  id: string
  content: string
  metadata: {
    type: string
    timestamp: string
    user_id: string
    tags?: string[]
  }
  embedding_id?: string
}

const columns: GridColDef[] = [
  { field: 'id', headerName: 'ID', width: 200 },
  { field: 'content', headerName: 'Content', flex: 1 },
  {
    field: 'type',
    headerName: 'Type',
    width: 120,
    valueGetter: (params) => params.row.metadata?.type || 'unknown',
  },
  {
    field: 'timestamp',
    headerName: 'Created',
    width: 180,
    valueGetter: (params) => 
      new Date(params.row.metadata?.timestamp || '').toLocaleString(),
  },
  {
    field: 'tags',
    headerName: 'Tags',
    width: 200,
    renderCell: (params) => (
      <Box>
        {params.row.metadata?.tags?.map((tag: string) => (
          <Chip key={tag} label={tag} size="small" sx={{ mr: 0.5 }} />
        ))}
      </Box>
    ),
  },
]

export default function MemorySystem() {
  const [memories, setMemories] = useState<Memory[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    fetchMemories()
  }, [])

  const fetchMemories = async () => {
    try {
      const response = await api.get('/api/memories')
      setMemories(response.data.memories || [])
    } catch (error) {
      console.error('Error fetching memories:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = async () => {
    setLoading(true)
    try {
      const response = await api.post('/api/memories/search', {
        query: searchQuery,
        limit: 50,
      })
      setMemories(response.data.results || [])
    } catch (error) {
      console.error('Error searching memories:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Memory System
      </Typography>

      {/* Search */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Box display="flex" gap={2}>
          <TextField
            fullWidth
            label="Search memories"
            variant="outlined"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          />
          <Button variant="contained" onClick={handleSearch}>
            Search
          </Button>
          <Button variant="outlined" onClick={fetchMemories}>
            Reset
          </Button>
        </Box>
      </Paper>

      {/* Stats */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Memories
              </Typography>
              <Typography variant="h4">
                {memories.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Memory Types
              </Typography>
              <Typography variant="h4">
                {new Set(memories.map(m => m.metadata?.type)).size}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Recent Activity
              </Typography>
              <Typography variant="h4">
                Active
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Memory Grid */}
      <Paper sx={{ height: 600, width: '100%' }}>
        {loading ? (
          <Box display="flex" justifyContent="center" alignItems="center" height="100%">
            <CircularProgress />
          </Box>
        ) : (
          <DataGrid
            rows={memories}
            columns={columns}
            pageSize={10}
            rowsPerPageOptions={[10, 25, 50]}
            checkboxSelection
            disableSelectionOnClick
          />
        )}
      </Paper>
    </Box>
  )
}