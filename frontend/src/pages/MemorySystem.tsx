import { useEffect, useState } from 'react'
import {
  Box,
  Typography,
  TextField,
  IconButton,
  Chip,
  Grid,
  alpha,
  InputAdornment,
} from '@mui/material'
import { DataGrid, GridColDef } from '@mui/x-data-grid'
import { 
  Search,
  Refresh,
  Memory as MemoryIcon,
  Timeline,
  Category,
  Storage,
  AutoAwesome,
  FilterList,
  CloudSync,
  Delete as DeleteIcon,
} from '@mui/icons-material'
import { motion } from 'framer-motion'
import PageContainer from '../components/ultra/PageContainer'
import UltraHeader from '../components/ultra/UltraHeader'
import MetricCard from '../components/ultra/MetricCard'
import GlassCard from '../components/GlassCard'
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

const MEMORY_TYPE_COLORS: Record<string, string> = {
  code: '#2196F3',
  documentation: '#00FF88',
  error: '#FF3366',
  decision: '#FFD700',
  workflow: '#00FFFF',
  pattern: '#8B5CF6',
  session: '#FF00FF',
  default: '#EC4899',
}

const columns: GridColDef[] = [
  { 
    field: 'id', 
    headerName: 'ID', 
    width: 150,
    renderCell: (params) => (
      <Typography variant="caption" sx={{ fontFamily: 'monospace', opacity: 0.7 }}>
        {params.value.slice(0, 8)}...
      </Typography>
    ),
  },
  { 
    field: 'content', 
    headerName: 'Content', 
    flex: 1,
    renderCell: (params) => (
      <Box sx={{ py: 1 }}>
        <Typography variant="body2" sx={{ 
          overflow: 'hidden', 
          textOverflow: 'ellipsis',
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
        }}>
          {params.value}
        </Typography>
      </Box>
    ),
  },
  {
    field: 'type',
    headerName: 'Type',
    width: 120,
    valueGetter: (params) => params?.row?.metadata?.type || 'unknown',
    renderCell: (params) => (
      <Chip
        label={params.value}
        size="small"
        sx={{
          backgroundColor: alpha(MEMORY_TYPE_COLORS[params.value] || MEMORY_TYPE_COLORS.default, 0.2),
          color: MEMORY_TYPE_COLORS[params.value] || MEMORY_TYPE_COLORS.default,
          fontWeight: 600,
        }}
      />
    ),
  },
  {
    field: 'timestamp',
    headerName: 'Created',
    width: 180,
    valueGetter: (params) => params?.row?.metadata?.timestamp || new Date().toISOString(),
    renderCell: (params) => (
      <Typography variant="caption" color="text.secondary">
        {params.value ? new Date(params.value).toLocaleString() : 'N/A'}
      </Typography>
    ),
  },
  {
    field: 'tags',
    headerName: 'Tags',
    width: 200,
    renderCell: (params) => (
      <Box display="flex" gap={0.5} flexWrap="wrap">
        {params.row.metadata?.tags?.map((tag: string) => (
          <Chip 
            key={tag} 
            label={tag} 
            size="small" 
            sx={{ 
              height: 20,
              fontSize: '0.75rem',
              backgroundColor: theme => alpha(theme.palette.primary.main, 0.1),
              color: 'primary.main',
            }} 
          />
        ))}
      </Box>
    ),
  },
  {
    field: 'actions',
    headerName: 'Actions',
    width: 100,
    sortable: false,
    renderCell: (params) => (
      <IconButton
        size="small"
        color="error"
        onClick={() => handleDelete(params.row.id)}
        sx={{ 
          '&:hover': { 
            backgroundColor: theme => alpha(theme.palette.error.main, 0.1) 
          } 
        }}
      >
        <DeleteIcon fontSize="small" />
      </IconButton>
    ),
  },
]

export default function MemorySystem() {
  const [memories, setMemories] = useState<Memory[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedRows, setSelectedRows] = useState<string[]>([])
  const [stats, setStats] = useState({
    total: 0,
    types: {} as Record<string, number>,
    recentActivity: 0,
    storageUsed: 0,
    syncStatus: 'active',
    lastSync: new Date(),
  })

  useEffect(() => {
    fetchMemories()
  }, [])

  const fetchMemories = async () => {
    try {
      // Try the memory API search endpoint with empty query to get all memories
      const response = await api.get('/api/v1/memories/search?q=&limit=100')
      const rawData = Array.isArray(response.data) ? response.data : (response.data.results || response.data.memories || [])
      
      // Map the API response to our Memory interface
      const memoriesData = rawData.map((mem: any) => ({
        id: mem.id,
        content: mem.content,
        metadata: {
          type: mem.memory_type?.toLowerCase() || 'unknown',
          timestamp: mem.created_at || mem.metadata?.timestamp || new Date().toISOString(),
          user_id: mem.user_id || 'unknown',
          tags: mem.tags || [],
          ...mem.metadata
        }
      }))
      
      setMemories(memoriesData)
      
      // Calculate stats
      const types = memoriesData.reduce((acc: Record<string, number>, mem: Memory) => {
        const type = mem.metadata?.type || 'unknown'
        acc[type] = (acc[type] || 0) + 1
        return acc
      }, {})
      
      setStats({
        total: memoriesData.length,
        types,
        recentActivity: memoriesData.filter((m: Memory) => {
          const timestamp = new Date(m.metadata?.timestamp || 0)
          const hourAgo = new Date(Date.now() - 60 * 60 * 1000)
          return timestamp > hourAgo
        }).length,
        storageUsed: memoriesData.length * 0.5, // Rough estimate in MB
        syncStatus: 'active',
        lastSync: new Date(),
      })
    } catch (error) {
      console.error('Error fetching memories:', error)
      setMemories([])
      setStats({
        totalMemories: 0,
        types: {},
        trends: [],
        lastSync: new Date(),
      })
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      fetchMemories()
      return
    }
    
    setLoading(true)
    try {
      const response = await api.post('/api/memory/search', {
        query: searchQuery,
        limit: 50,
      })
      setMemories(response.data.memories || response.data.results || [])
    } catch (error) {
      console.error('Error searching memories:', error)
      setMemories([])
    } finally {
      setLoading(false)
    }
  }

  const generateSparkline = (trends: number[] | undefined): number[] => {
    return trends || []
  }

  const handleDelete = async (memoryId: string) => {
    try {
      await api.delete(`/api/memory/${memoryId}`)
      // Remove the memory from local state
      setMemories(prevMemories => prevMemories.filter(m => m.id !== memoryId))
      // Update stats
      setStats(prevStats => ({
        ...prevStats,
        total: prevStats.total - 1,
      }))
    } catch (error) {
      console.error('Error deleting memory:', error)
    }
  }

  const handleBulkDelete = async () => {
    if (selectedRows.length === 0) return
    
    try {
      // Delete all selected memories
      await Promise.all(selectedRows.map(id => api.delete(`/api/memory/${id}`)))
      
      // Remove deleted memories from local state
      setMemories(prevMemories => prevMemories.filter(m => !selectedRows.includes(m.id)))
      
      // Update stats
      setStats(prevStats => ({
        ...prevStats,
        total: prevStats.total - selectedRows.length,
      }))
      
      // Clear selection
      setSelectedRows([])
    } catch (error) {
      console.error('Error deleting memories:', error)
    }
  }

  return (
    <PageContainer>
      <UltraHeader 
        title="Memory System" 
        subtitle="COGNITIVE MEMORY MANAGEMENT"
      />

      <Box sx={{ px: 3, pb: 6 }}>
        {/* Search Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <GlassCard sx={{ mb: 4 }}>
            <Box sx={{ p: 3 }}>
              <Box display="flex" gap={2} alignItems="center">
                <TextField
                  fullWidth
                  placeholder="Search memories by content, tags, or type..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      backgroundColor: alpha('#ffffff', 0.05),
                      '& fieldset': {
                        borderColor: alpha('#ffffff', 0.1),
                      },
                      '&:hover fieldset': {
                        borderColor: alpha('#ffffff', 0.2),
                      },
                      '&.Mui-focused fieldset': {
                        borderColor: 'primary.main',
                      },
                    },
                  }}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <Search sx={{ color: 'primary.main' }} />
                      </InputAdornment>
                    ),
                  }}
                />
                <IconButton
                  onClick={handleSearch}
                  sx={{
                    backgroundColor: 'primary.main',
                    color: 'white',
                    '&:hover': {
                      backgroundColor: 'primary.dark',
                    },
                  }}
                >
                  <Search />
                </IconButton>
                <IconButton
                  onClick={fetchMemories}
                  sx={{
                    backgroundColor: alpha('#ffffff', 0.1),
                    '&:hover': {
                      backgroundColor: alpha('#ffffff', 0.2),
                    },
                  }}
                >
                  <Refresh />
                </IconButton>
                {selectedRows.length > 0 && (
                  <IconButton
                    onClick={handleBulkDelete}
                    sx={{
                      backgroundColor: alpha('#ff0000', 0.2),
                      color: 'error.main',
                      '&:hover': {
                        backgroundColor: alpha('#ff0000', 0.3),
                      },
                    }}
                  >
                    <DeleteIcon />
                  </IconButton>
                )}
              </Box>
            </Box>
          </GlassCard>
        </motion.div>

        {/* Metrics */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              icon={<MemoryIcon />}
              label="Total Memories"
              value={stats.total}
              trend={12.5}
              color="#2196F3"
              sparkline={generateSparkline(stats.trends)}
              delay={0.3}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              icon={<Category />}
              label="Memory Types"
              value={Object.keys(stats.types).length}
              trend={0}
              color="#FF00FF"
              sparkline={generateSparkline(stats.trends)}
              delay={0.4}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              icon={<Timeline />}
              label="Recent Activity"
              value={stats.recentActivity}
              trend={25.3}
              unit="last hour"
              color="#00FF88"
              sparkline={generateSparkline(stats.trends)}
              delay={0.5}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              icon={<Storage />}
              label="Storage Used"
              value={stats.storageUsed.toFixed(1)}
              trend={-5.2}
              unit="MB"
              color="#FFD700"
              sparkline={generateSparkline(stats.trends)}
              delay={0.6}
            />
          </Grid>
        </Grid>

        {/* Memory Types Overview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
        >
          <GlassCard sx={{ mb: 4 }}>
            <Box sx={{ p: 3 }}>
              <Box display="flex" alignItems="center" gap={2} mb={3}>
                <FilterList sx={{ color: 'primary.main' }} />
                <Typography variant="h6" fontWeight="bold">
                  Memory Types Distribution
                </Typography>
                <Chip
                  icon={<CloudSync />}
                  label={`Synced ${new Date(stats.lastSync).toLocaleTimeString()}`}
                  size="small"
                  color="success"
                  sx={{ ml: 'auto' }}
                />
              </Box>
              
              <Box display="flex" gap={2} flexWrap="wrap">
                {Object.entries(stats.types).map(([type, count]) => (
                  <motion.div
                    key={type}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.8 }}
                  >
                    <Box
                      sx={{
                        p: 2,
                        borderRadius: 2,
                        backgroundColor: alpha(MEMORY_TYPE_COLORS[type] || MEMORY_TYPE_COLORS.default, 0.1),
                        border: `1px solid ${alpha(MEMORY_TYPE_COLORS[type] || MEMORY_TYPE_COLORS.default, 0.3)}`,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 2,
                      }}
                    >
                      <Box
                        sx={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          backgroundColor: MEMORY_TYPE_COLORS[type] || MEMORY_TYPE_COLORS.default,
                          boxShadow: `0 0 10px ${MEMORY_TYPE_COLORS[type] || MEMORY_TYPE_COLORS.default}`,
                        }}
                      />
                      <Typography variant="body2" fontWeight="bold" textTransform="capitalize">
                        {type}
                      </Typography>
                      <Chip
                        label={count}
                        size="small"
                        sx={{
                          height: 24,
                          backgroundColor: alpha(MEMORY_TYPE_COLORS[type] || MEMORY_TYPE_COLORS.default, 0.2),
                          color: MEMORY_TYPE_COLORS[type] || MEMORY_TYPE_COLORS.default,
                          fontWeight: 'bold',
                        }}
                      />
                    </Box>
                  </motion.div>
                ))}
              </Box>
            </Box>
          </GlassCard>
        </motion.div>

        {/* Memory Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9 }}
        >
          <GlassCard sx={{ height: 600 }}>
            {loading ? (
              <Box display="flex" alignItems="center" justifyContent="center" height="100%">
                <Box textAlign="center">
                  <Box
                    sx={{
                      width: 80,
                      height: 80,
                      margin: '0 auto',
                      position: 'relative',
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        inset: 0,
                        borderRadius: '50%',
                        border: '3px solid transparent',
                        borderTopColor: 'primary.main',
                        animation: 'spin 1s linear infinite',
                      },
                      '@keyframes spin': {
                        '0%': { transform: 'rotate(0deg)' },
                        '100%': { transform: 'rotate(360deg)' },
                      },
                    }}
                  >
                    <AutoAwesome
                      sx={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        fontSize: 32,
                        color: 'primary.main',
                        animation: 'pulse 2s infinite',
                      }}
                    />
                  </Box>
                  <Typography variant="h6" sx={{ mt: 2 }}>
                    Loading memories...
                  </Typography>
                </Box>
              </Box>
            ) : (
              <DataGrid
                rows={memories}
                columns={columns}
                initialState={{
                  pagination: {
                    paginationModel: { pageSize: 10 },
                  },
                }}
                pageSizeOptions={[10, 25, 50]}
                checkboxSelection
                disableRowSelectionOnClick
                onRowSelectionModelChange={(newSelection) => {
                  setSelectedRows(newSelection as string[])
                }}
                sx={{
                  border: 'none',
                  '& .MuiDataGrid-root': {
                    border: 'none',
                  },
                  '& .MuiDataGrid-cell': {
                    borderBottom: theme => `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                  },
                  '& .MuiDataGrid-columnHeaders': {
                    backgroundColor: theme => alpha(theme.palette.background.default, 0.5),
                    borderBottom: theme => `1px solid ${alpha(theme.palette.divider, 0.2)}`,
                  },
                  '& .MuiDataGrid-columnHeaderTitle': {
                    fontWeight: 'bold',
                  },
                  '& .MuiDataGrid-row': {
                    '&:hover': {
                      backgroundColor: theme => alpha(theme.palette.primary.main, 0.05),
                    },
                  },
                  '& .MuiCheckbox-root': {
                    color: theme => alpha(theme.palette.primary.main, 0.6),
                  },
                  '& .MuiDataGrid-footerContainer': {
                    borderTop: theme => `1px solid ${alpha(theme.palette.divider, 0.2)}`,
                  },
                }}
              />
            )}
          </GlassCard>
        </motion.div>
      </Box>
    </PageContainer>
  )
}