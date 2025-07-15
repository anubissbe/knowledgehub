import { useState, useEffect } from 'react'
import {
  Box,
  Paper,
  Typography,
  Button,
  TextField,
  Grid,
  Card,
  CardContent,
  IconButton,
  Tooltip,
} from '@mui/material'
import { Refresh, ZoomIn, ZoomOut, CenterFocusStrong } from '@mui/icons-material'
import { api } from '../services/api'

interface GraphStats {
  total_nodes: number
  total_edges: number
  node_types: Record<string, number>
  edge_types: Record<string, number>
}

export default function KnowledgeGraph() {
  const [stats, setStats] = useState<GraphStats | null>(null)
  const [queryResult, setQueryResult] = useState<any>(null)
  const [cypherQuery, setCypherQuery] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchGraphStats()
  }, [])

  const fetchGraphStats = async () => {
    try {
      const response = await api.get('/api/knowledge-graph/stats')
      setStats(response.data)
    } catch (error) {
      console.error('Error fetching graph stats:', error)
    }
  }

  const executeCypherQuery = async () => {
    setLoading(true)
    try {
      const response = await api.post('/api/knowledge-graph/query', {
        query: cypherQuery,
      })
      setQueryResult(response.data)
    } catch (error) {
      console.error('Error executing query:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Knowledge Graph
      </Typography>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Nodes
              </Typography>
              <Typography variant="h4">
                {stats?.total_nodes?.toLocaleString() || '0'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Edges
              </Typography>
              <Typography variant="h4">
                {stats?.total_edges?.toLocaleString() || '0'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Node Types
              </Typography>
              <Typography variant="h4">
                {Object.keys(stats?.node_types || {}).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Edge Types
              </Typography>
              <Typography variant="h4">
                {Object.keys(stats?.edge_types || {}).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Graph Visualization Placeholder */}
      <Paper sx={{ p: 2, mb: 3, height: 400 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">Graph Visualization</Typography>
          <Box>
            <Tooltip title="Zoom In">
              <IconButton><ZoomIn /></IconButton>
            </Tooltip>
            <Tooltip title="Zoom Out">
              <IconButton><ZoomOut /></IconButton>
            </Tooltip>
            <Tooltip title="Center">
              <IconButton><CenterFocusStrong /></IconButton>
            </Tooltip>
            <Tooltip title="Refresh">
              <IconButton onClick={fetchGraphStats}><Refresh /></IconButton>
            </Tooltip>
          </Box>
        </Box>
        <Box
          sx={{
            height: 300,
            bgcolor: 'background.default',
            borderRadius: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Typography color="text.secondary">
            Graph visualization would be rendered here using vis.js or similar library
          </Typography>
        </Box>
      </Paper>

      {/* Cypher Query */}
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Cypher Query Console
        </Typography>
        <TextField
          fullWidth
          multiline
          rows={4}
          label="Enter Cypher Query"
          value={cypherQuery}
          onChange={(e) => setCypherQuery(e.target.value)}
          placeholder="MATCH (n) RETURN n LIMIT 10"
          sx={{ mb: 2 }}
        />
        <Button
          variant="contained"
          onClick={executeCypherQuery}
          disabled={loading || !cypherQuery}
        >
          Execute Query
        </Button>

        {queryResult && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Query Result:
            </Typography>
            <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
              <pre style={{ margin: 0, overflow: 'auto' }}>
                {JSON.stringify(queryResult, null, 2)}
              </pre>
            </Paper>
          </Box>
        )}
      </Paper>
    </Box>
  )
}