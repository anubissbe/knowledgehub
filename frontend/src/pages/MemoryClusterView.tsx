/**
 * Memory Cluster View
 * Visual exploration of memory clustering and episodic vs semantic memory organization
 */

import React, { useState, useEffect } from 'react'
import {
  Box,
  Grid,
  Typography,
  Card,
  CardContent,
  Button,
  Chip,
  Tab,
  Tabs,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel,
  Alert,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material'
import {
  AccountTree as ClusterIcon,
  Memory as MemoryIcon,
  Psychology as PsychologyIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  Timeline as TimelineIcon,
  Category as CategoryIcon
} from '@mui/icons-material'
import { motion } from 'framer-motion'

import PageWrapper from '../components/PageWrapper'
import GlassCard from '../components/GlassCard'
import Network3D from '../components/Network3D'
import { api } from '../services/api'

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
      id={`memory-tabpanel-${index}`}
      aria-labelledby={`memory-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  )
}

interface MemoryCluster {
  id: string
  name: string
  type: 'episodic' | 'semantic'
  size: number
  centroid: number[]
  memories: Memory[]
  created_at: string
  updated_at: string
  cohesion_score: number
  tags: string[]
}

interface Memory {
  id: string
  content: string
  type: 'episodic' | 'semantic'
  importance: number
  timestamp: string
  session_id: string
  cluster_id?: string
  embedding?: number[]
  metadata: Record<string, any>
}

interface ClusterStats {
  total_clusters: number
  episodic_clusters: number
  semantic_clusters: number
  avg_cluster_size: number
  total_memories: number
  cluster_distribution: Record<string, number>
}

const MemoryClusterView: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0)
  const [clusters, setClusters] = useState<MemoryCluster[]>([])
  const [selectedCluster, setSelectedCluster] = useState<MemoryCluster | null>(null)
  const [stats, setStats] = useState<ClusterStats | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [memoryTypeFilter, setMemoryTypeFilter] = useState<'all' | 'episodic' | 'semantic'>('all')
  const [loading, setLoading] = useState(false)
  const [networkData, setNetworkData] = useState<any>(null)

  useEffect(() => {
    loadClusters()
    loadStats()
  }, [memoryTypeFilter])

  const loadClusters = async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (memoryTypeFilter !== 'all') {
        params.append('type', memoryTypeFilter)
      }
      if (searchQuery) {
        params.append('search', searchQuery)
      }

      const response = await api.get(`/api/memory/clusters?${params}`)
      const clusterData = response.data.clusters || []
      setClusters(clusterData)
      
      // Transform clusters into network data
      const nodes = clusterData.map((cluster: MemoryCluster) => ({
        id: cluster.id,
        label: cluster.name,
        group: cluster.type,
        size: Math.sqrt(cluster.size) * 10,
        color: cluster.type === 'episodic' ? '#2196F3' : '#4CAF50',
        font: { size: 14 }
      }))

      const edges = clusterData.flatMap((cluster: MemoryCluster, i: number) =>
        clusterData.slice(i + 1).map((otherCluster: MemoryCluster) => ({
          from: cluster.id,
          to: otherCluster.id,
          width: calculateClusterSimilarity(cluster, otherCluster) * 5,
          color: { opacity: 0.3 }
        }))
      ).filter(edge => edge.width > 1)

      setNetworkData({ nodes, edges })
    } catch (error) {
      console.error('Failed to load clusters:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadStats = async () => {
    try {
      const response = await api.get('/api/memory/clusters/stats')
      setStats(response.data)
    } catch (error) {
      console.error('Failed to load stats:', error)
    }
  }

  const calculateClusterSimilarity = (cluster1: MemoryCluster, cluster2: MemoryCluster): number => {
    // Simple similarity based on shared tags and type
    const sharedTags = cluster1.tags.filter(tag => cluster2.tags.includes(tag)).length
    const maxTags = Math.max(cluster1.tags.length, cluster2.tags.length)
    const typeBonus = cluster1.type === cluster2.type ? 0.2 : 0
    
    return Math.min(1, (sharedTags / maxTags) + typeBonus)
  }

  const handleClusterClick = (cluster: MemoryCluster) => {
    setSelectedCluster(cluster)
  }

  const handleSearch = () => {
    loadClusters()
  }

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue)
  }

  const getMemoryTypeColor = (type: 'episodic' | 'semantic') => {
    return type === 'episodic' ? 'primary' : 'success'
  }

  return (
    <PageWrapper>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Typography variant="h4" gutterBottom sx={{ mb: 4, fontWeight: 600 }}>
          🧠 Memory Cluster View
        </Typography>

        {/* Stats Overview */}
        {stats && (
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={1}>
                    <ClusterIcon color="primary" sx={{ mr: 1 }} />
                    <Typography variant="h6">Total Clusters</Typography>
                  </Box>
                  <Typography variant="h3" color="primary.main">
                    {stats.total_clusters}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={1}>
                    <TimelineIcon color="info" sx={{ mr: 1 }} />
                    <Typography variant="h6">Episodic</Typography>
                  </Box>
                  <Typography variant="h3" color="info.main">
                    {stats.episodic_clusters}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={1}>
                    <CategoryIcon color="success" sx={{ mr: 1 }} />
                    <Typography variant="h6">Semantic</Typography>
                  </Box>
                  <Typography variant="h3" color="success.main">
                    {stats.semantic_clusters}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={1}>
                    <MemoryIcon color="warning" sx={{ mr: 1 }} />
                    <Typography variant="h6">Total Memories</Typography>
                  </Box>
                  <Typography variant="h3" color="warning.main">
                    {stats.total_memories}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}

        {/* Controls */}
        <GlassCard sx={{ mb: 3 }}>
          <Box sx={{ p: 3 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={6} md={4}>
                <TextField
                  fullWidth
                  size="small"
                  placeholder="Search memories and clusters..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  InputProps={{
                    startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />
                  }}
                />
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <FormControl fullWidth size="small">
                  <InputLabel>Memory Type</InputLabel>
                  <Select
                    value={memoryTypeFilter}
                    label="Memory Type"
                    onChange={(e) => setMemoryTypeFilter(e.target.value as any)}
                  >
                    <MenuItem value="all">All Types</MenuItem>
                    <MenuItem value="episodic">Episodic</MenuItem>
                    <MenuItem value="semantic">Semantic</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} sm={6} md={2}>
                <Button
                  fullWidth
                  variant="contained"
                  onClick={handleSearch}
                  startIcon={<SearchIcon />}
                  disabled={loading}
                >
                  Search
                </Button>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Button
                  fullWidth
                  variant="outlined"
                  onClick={loadClusters}
                  startIcon={<RefreshIcon />}
                  disabled={loading}
                >
                  Refresh
                </Button>
              </Grid>
            </Grid>
          </Box>
        </GlassCard>

        <Grid container spacing={3}>
          {/* Main Visualization */}
          <Grid item xs={12} lg={8}>
            <GlassCard>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={activeTab} onChange={handleTabChange}>
                  <Tab label="Cluster Network" icon={<ClusterIcon />} />
                  <Tab label="Memory Timeline" icon={<TimelineIcon />} />
                </Tabs>
              </Box>

              <TabPanel value={activeTab} index={0}>
                <Box sx={{ height: 600 }}>
                  <Typography variant="h6" gutterBottom>
                    Memory Cluster Network
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Interactive visualization of memory clusters. Blue nodes represent episodic memories,
                    green nodes represent semantic memories. Edge thickness indicates cluster similarity.
                  </Typography>
                  
                  {networkData ? (
                    <Network3D
                      data={networkData}
                      height={500}
                      onNodeClick={(node) => {
                        const cluster = clusters.find(c => c.id === node.id)
                        if (cluster) handleClusterClick(cluster)
                      }}
                    />
                  ) : (
                    <Box 
                      display="flex" 
                      alignItems="center" 
                      justifyContent="center" 
                      height={400}
                      bgcolor="background.default"
                      borderRadius={1}
                    >
                      <Typography variant="h6" color="text.secondary">
                        Loading cluster visualization...
                      </Typography>
                    </Box>
                  )}
                </Box>
              </TabPanel>

              <TabPanel value={activeTab} index={1}>
                <Box sx={{ height: 600 }}>
                  <Typography variant="h6" gutterBottom>
                    Memory Formation Timeline
                  </Typography>
                  <Alert severity="info" sx={{ mb: 2 }}>
                    Timeline visualization will show memory formation patterns over time.
                    This feature will be enhanced as more temporal data is collected.
                  </Alert>
                  
                  <Box 
                    display="flex" 
                    alignItems="center" 
                    justifyContent="center" 
                    height={400}
                    bgcolor="background.default"
                    borderRadius={1}
                  >
                    <Typography variant="h6" color="text.secondary">
                      Timeline visualization coming soon...
                    </Typography>
                  </Box>
                </Box>
              </TabPanel>
            </GlassCard>
          </Grid>

          {/* Cluster Details Sidebar */}
          <Grid item xs={12} lg={4}>
            <GlassCard>
              <Box sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  🔍 Cluster Explorer
                </Typography>
                
                {selectedCluster ? (
                  <Box>
                    <Box display="flex" alignItems="center" gap={1} mb={2}>
                      <Chip
                        label={selectedCluster.type.toUpperCase()}
                        color={getMemoryTypeColor(selectedCluster.type) as any}
                        size="small"
                      />
                      <Chip
                        label={`${selectedCluster.size} memories`}
                        variant="outlined"
                        size="small"
                      />
                    </Box>
                    
                    <Typography variant="h6" gutterBottom>
                      {selectedCluster.name}
                    </Typography>
                    
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Cohesion Score: {selectedCluster.cohesion_score.toFixed(3)}
                    </Typography>
                    
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Created: {new Date(selectedCluster.created_at).toLocaleString()}
                    </Typography>
                    
                    {selectedCluster.tags.length > 0 && (
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          Tags:
                        </Typography>
                        <Box display="flex" flexWrap="wrap" gap={0.5}>
                          {selectedCluster.tags.map((tag, index) => (
                            <Chip
                              key={index}
                              label={tag}
                              size="small"
                              variant="outlined"
                            />
                          ))}
                        </Box>
                      </Box>
                    )}
                    
                    <Accordion>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography variant="subtitle1">
                          Memories ({selectedCluster.memories.length})
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <List dense>
                          {selectedCluster.memories.slice(0, 10).map((memory) => (
                            <ListItem key={memory.id} sx={{ px: 0 }}>
                              <ListItemIcon>
                                <MemoryIcon fontSize="small" />
                              </ListItemIcon>
                              <ListItemText
                                primary={memory.content.substring(0, 100) + '...'}
                                secondary={
                                  <Box>
                                    <Typography variant="caption" display="block">
                                      Importance: {memory.importance.toFixed(2)}
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary">
                                      {new Date(memory.timestamp).toLocaleString()}
                                    </Typography>
                                  </Box>
                                }
                              />
                            </ListItem>
                          ))}
                          {selectedCluster.memories.length > 10 && (
                            <ListItem sx={{ px: 0 }}>
                              <ListItemText
                                primary={`... and ${selectedCluster.memories.length - 10} more memories`}
                                sx={{ fontStyle: 'italic', color: 'text.secondary' }}
                              />
                            </ListItem>
                          )}
                        </List>
                      </AccordionDetails>
                    </Accordion>
                  </Box>
                ) : (
                  <Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Click on a cluster in the visualization to explore its details.
                    </Typography>
                    
                    <Typography variant="subtitle2" gutterBottom sx={{ mt: 3 }}>
                      Available Clusters:
                    </Typography>
                    
                    <List dense>
                      {clusters.slice(0, 8).map((cluster) => (
                        <ListItem
                          key={cluster.id}
                          button
                          onClick={() => handleClusterClick(cluster)}
                          sx={{ px: 0, borderRadius: 1, mb: 0.5 }}
                        >
                          <ListItemIcon>
                            <ClusterIcon fontSize="small" />
                          </ListItemIcon>
                          <ListItemText
                            primary={cluster.name}
                            secondary={
                              <Box display="flex" alignItems="center" gap={1}>
                                <Chip
                                  label={cluster.type}
                                  color={getMemoryTypeColor(cluster.type) as any}
                                  size="small"
                                />
                                <Typography variant="caption">
                                  {cluster.size} memories
                                </Typography>
                              </Box>
                            }
                          />
                        </ListItem>
                      ))}
                    </List>
                    
                    {clusters.length > 8 && (
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                        ... and {clusters.length - 8} more clusters
                      </Typography>
                    )}
                  </Box>
                )}
              </Box>
            </GlassCard>
          </Grid>
        </Grid>
      </motion.div>
    </PageWrapper>
  )
}

export default MemoryClusterView