import { useState, useEffect, useRef } from 'react'
import {
  Box,
  Typography,
  IconButton,
  ToggleButton,
  ToggleButtonGroup,
  Slider,
  Chip,
  LinearProgress,
  alpha,
} from '@mui/material'
import { 
  Refresh, 
  ZoomIn, 
  ZoomOut, 
  CenterFocusStrong,
  PlayArrow,
  Pause,
  AutoAwesome,
  BubbleChart,
  AccountTree,
  Hub,
  Fullscreen,
  FilterList,
} from '@mui/icons-material'
import { Network } from 'vis-network/standalone'
import { DataSet } from 'vis-data'
import { motion } from 'framer-motion'
import PageWrapper from '../components/PageWrapper'
import UltraHeader from '../components/ultra/UltraHeader'
import GlassCard from '../components/GlassCard'
import { api } from '../services/api'

interface GraphNode {
  id: string
  label: string
  type: string
  properties?: Record<string, any>
}

interface GraphEdge {
  from: string
  to: string
  label?: string
  type?: string
}

interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

const NODE_COLORS = {
  memory: '#2196F3',
  session: '#FF00FF',
  decision: '#00FF88',
  error: '#FF3366',
  pattern: '#FFD700',
  workflow: '#00FFFF',
  code: '#8B5CF6',
  default: '#EC4899',
  primary: '#2196F3',
}

const LAYOUTS = [
  { value: 'physics', label: 'Physics', icon: <BubbleChart /> },
  { value: 'hierarchical', label: 'Hierarchical', icon: <AccountTree /> },
  { value: 'circular', label: 'Circular', icon: <Hub /> },
]

export default function KnowledgeGraph() {
  const containerRef = useRef<HTMLDivElement>(null)
  const networkRef = useRef<Network | null>(null)
  const animationRef = useRef<number | null>(null)
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(true)
  const [layout, setLayout] = useState('physics')
  const [autoRotate, setAutoRotate] = useState(true)
  const [nodeSize, setNodeSize] = useState(25)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [stats, setStats] = useState({
    nodes: 0,
    edges: 0,
    types: {} as Record<string, number>,
  })

  useEffect(() => {
    fetchGraphData()
    return () => {
      // Clean up animation
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
        animationRef.current = null
      }
      // Clean up network
      if (networkRef.current) {
        networkRef.current.destroy()
        networkRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (graphData && containerRef.current) {
      renderGraph()
    }
  }, [graphData, layout, nodeSize])

  const fetchGraphData = async () => {
    try {
      const response = await api.get('/api/knowledge-graph/full')
      const data = response.data as GraphData
      
      // Calculate stats
      const nodeTypes = data.nodes.reduce((acc, node) => {
        acc[node.type] = (acc[node.type] || 0) + 1
        return acc
      }, {} as Record<string, number>)
      
      setStats({
        nodes: data.nodes.length,
        edges: data.edges.length,
        types: nodeTypes,
      })
      
      setGraphData(data)
      setLoading(false)
    } catch (error) {
      console.error('Error fetching graph data:', error)
      setGraphData({ nodes: [], edges: [] })
      setLoading(false)
    }
  }

  const renderGraph = () => {
    if (!containerRef.current || !graphData) return

    // Clean up any existing animation
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }

    const nodes = new DataSet(
      graphData.nodes.map(node => ({
        id: node.id,
        label: node.label,
        color: {
          background: NODE_COLORS[node.type as keyof typeof NODE_COLORS] || NODE_COLORS.default,
          border: '#ffffff',
          highlight: {
            background: NODE_COLORS[node.type as keyof typeof NODE_COLORS] || NODE_COLORS.default,
            border: '#ffffff',
          },
        },
        font: {
          color: '#ffffff',
          size: 14,
          face: 'Inter',
        },
        size: nodeSize,
        shape: 'dot',
        borderWidth: 2,
        shadow: {
          enabled: true,
          color: 'rgba(0,0,0,0.5)',
          size: 10,
          x: 3,
          y: 3,
        },
      }))
    )

    const edges = new DataSet(
      graphData.edges.map((edge, index) => ({
        id: index.toString(),
        from: edge.from,
        to: edge.to,
        label: edge.label,
        color: {
          color: alpha('#ffffff', 0.3),
          highlight: NODE_COLORS.primary,
        },
        width: 2,
        smooth: {
          enabled: true,
          type: 'dynamic',
          roundness: 0.5,
        },
        arrows: {
          to: {
            enabled: true,
            scaleFactor: 0.5,
          },
        },
      }))
    )

    const options = {
      nodes: {
        borderWidth: 2,
        shadow: true,
      },
      edges: {
        smooth: {
          enabled: true,
          type: 'dynamic',
          roundness: 0.5,
        },
      },
      physics: {
        enabled: layout === 'physics',
        stabilization: {
          enabled: true,
          iterations: 100,
        },
        barnesHut: {
          gravitationalConstant: -2000,
          centralGravity: 0.3,
          springLength: 95,
          springConstant: 0.04,
        },
      },
      layout: {
        improvedLayout: true,
        hierarchical: layout === 'hierarchical' ? {
          enabled: true,
          direction: 'UD',
          sortMethod: 'directed',
          nodeSpacing: 150,
          levelSeparation: 150,
        } : false,
      },
      interaction: {
        hover: true,
        navigationButtons: true,
        keyboard: true,
        multiselect: true,
      },
    }

    if (networkRef.current) {
      networkRef.current.destroy()
    }

    networkRef.current = new Network(containerRef.current, { nodes, edges }, options)

    // Handle node selection
    networkRef.current.on('selectNode', (params) => {
      if (params.nodes.length > 0) {
        setSelectedNode(params.nodes[0])
      }
    })

    networkRef.current.on('deselectNode', () => {
      setSelectedNode(null)
    })

    // Auto rotation effect with proper cleanup
    if (autoRotate && layout === 'physics') {
      let angle = 0
      const rotate = () => {
        // Check if network still exists and auto-rotate is enabled
        if (!networkRef.current || !autoRotate) {
          if (animationRef.current) {
            cancelAnimationFrame(animationRef.current)
            animationRef.current = null
          }
          return
        }
        
        angle += 0.01
        try {
          networkRef.current.moveTo({
            position: { x: Math.sin(angle) * 50, y: Math.cos(angle) * 50 },
            scale: 1,
            animation: false,
          })
        } catch (error) {
          console.warn('Error in rotation animation:', error)
          if (animationRef.current) {
            cancelAnimationFrame(animationRef.current)
            animationRef.current = null
          }
          return
        }
        
        animationRef.current = requestAnimationFrame(rotate)
      }
      animationRef.current = requestAnimationFrame(rotate)
    }
  }

  const handleZoomIn = () => {
    if (networkRef.current) {
      try {
        networkRef.current.moveTo({ scale: networkRef.current.getScale() * 1.2 })
      } catch (error) {
        console.warn('Error in zoom in:', error)
      }
    }
  }
  
  const handleZoomOut = () => {
    if (networkRef.current) {
      try {
        networkRef.current.moveTo({ scale: networkRef.current.getScale() * 0.8 })
      } catch (error) {
        console.warn('Error in zoom out:', error)
      }
    }
  }
  
  const handleFit = () => {
    if (networkRef.current) {
      try {
        networkRef.current.fit()
      } catch (error) {
        console.warn('Error in fit:', error)
      }
    }
  }
  const handleFullscreen = () => {
    if (containerRef.current?.requestFullscreen) {
      containerRef.current.requestFullscreen()
    }
  }

  return (
    <PageWrapper>
      <UltraHeader 
        title="Knowledge Graph" 
        subtitle="NEURAL NETWORK VISUALIZATION"
      />

      <Box sx={{ px: { xs: 2, sm: 3, md: 4 }, pb: 6, maxWidth: '100%', overflow: 'hidden' }}>
        {/* Controls */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <GlassCard sx={{ mb: 3 }}>
            <Box sx={{ p: 2 }}>
              <Box display="flex" alignItems="center" justifyContent="space-between" flexWrap="wrap" gap={2}>
                {/* Layout Toggle */}
                <ToggleButtonGroup
                  value={layout}
                  exclusive
                  onChange={(_, value) => value && setLayout(value)}
                  size="small"
                >
                  {LAYOUTS.map((l) => (
                    <ToggleButton key={l.value} value={l.value}>
                      {l.icon}
                      <Typography variant="caption" sx={{ ml: 1 }}>
                        {l.label}
                      </Typography>
                    </ToggleButton>
                  ))}
                </ToggleButtonGroup>

                {/* Node Size Slider */}
                <Box display="flex" alignItems="center" gap={2} sx={{ minWidth: 200 }}>
                  <Typography variant="caption" color="text.secondary">
                    Node Size
                  </Typography>
                  <Slider
                    value={nodeSize}
                    onChange={(_: Event, value: number | number[]) => {
                      setNodeSize(value as number)
                      if (graphData) renderGraph()
                    }}
                    min={15}
                    max={50}
                    sx={{ flex: 1 }}
                  />
                </Box>

                {/* Controls */}
                <Box display="flex" gap={1}>
                  <IconButton onClick={() => setAutoRotate(!autoRotate)} color={autoRotate ? 'primary' : 'default'}>
                    {autoRotate ? <Pause /> : <PlayArrow />}
                  </IconButton>
                  <IconButton onClick={handleZoomIn}>
                    <ZoomIn />
                  </IconButton>
                  <IconButton onClick={handleZoomOut}>
                    <ZoomOut />
                  </IconButton>
                  <IconButton onClick={handleFit}>
                    <CenterFocusStrong />
                  </IconButton>
                  <IconButton onClick={handleFullscreen}>
                    <Fullscreen />
                  </IconButton>
                  <IconButton onClick={fetchGraphData}>
                    <Refresh />
                  </IconButton>
                </Box>
              </Box>
            </Box>
          </GlassCard>
        </motion.div>

        {/* Main Graph */}
        <Box display="flex" gap={3} sx={{ height: 600 }}>
          {/* Graph Container */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3, type: 'spring', stiffness: 100 }}
            style={{ flex: 1 }}
          >
            <GlassCard sx={{ height: '100%', position: 'relative', overflow: 'hidden' }}>
              {loading ? (
                <Box display="flex" alignItems="center" justifyContent="center" height="100%">
                  <Box textAlign="center">
                    <AutoAwesome
                      sx={{
                        fontSize: 60,
                        color: 'primary.main',
                        animation: 'pulse 2s infinite',
                      }}
                    />
                    <Typography variant="h6" sx={{ mt: 2 }}>
                      Loading Knowledge Graph...
                    </Typography>
                  </Box>
                </Box>
              ) : (
                <Box
                  ref={containerRef}
                  sx={{
                    width: '100%',
                    height: '100%',
                    '& .vis-network': {
                      backgroundColor: 'transparent !important',
                    },
                  }}
                />
              )}

              {/* Selected Node Info */}
              {selectedNode && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  style={{
                    position: 'absolute',
                    bottom: 16,
                    left: 16,
                    right: 16,
                  }}
                >
                  <GlassCard>
                    <Box sx={{ p: 2 }}>
                      <Typography variant="body2" fontWeight="bold">
                        Selected Node: {selectedNode}
                      </Typography>
                    </Box>
                  </GlassCard>
                </motion.div>
              )}
            </GlassCard>
          </motion.div>

          {/* Stats Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            style={{ width: 300 }}
          >
            <GlassCard sx={{ height: '100%' }}>
              <Box sx={{ p: 3 }}>
                <Box display="flex" alignItems="center" gap={2} mb={3}>
                  <FilterList sx={{ color: 'primary.main' }} />
                  <Typography variant="h6" fontWeight="bold">
                    Graph Statistics
                  </Typography>
                </Box>

                {/* Node & Edge Count */}
                <Box sx={{ mb: 3 }}>
                  <Box display="flex" justifyContent="space-between" mb={2}>
                    <Box>
                      <Typography variant="h4" fontWeight="bold" color="primary.main">
                        {stats.nodes}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Total Nodes
                      </Typography>
                    </Box>
                    <Box textAlign="right">
                      <Typography variant="h4" fontWeight="bold" color="secondary.main">
                        {stats.edges}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Total Edges
                      </Typography>
                    </Box>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={(stats.edges / (stats.nodes * 2)) * 100}
                    sx={{
                      height: 8,
                      borderRadius: 4,
                      backgroundColor: theme => alpha(theme.palette.primary.main, 0.1),
                      '& .MuiLinearProgress-bar': {
                        borderRadius: 4,
                        background: theme => `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                      },
                    }}
                  />
                </Box>

                {/* Node Types */}
                <Box>
                  <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                    Node Types
                  </Typography>
                  <Box display="flex" flexDirection="column" gap={1}>
                    {Object.entries(stats.types).map(([type, count]) => (
                      <Box
                        key={type}
                        sx={{
                          p: 1.5,
                          borderRadius: 2,
                          backgroundColor: theme => alpha(theme.palette.background.default, 0.5),
                          borderLeft: `3px solid ${NODE_COLORS[type as keyof typeof NODE_COLORS] || NODE_COLORS.default}`,
                        }}
                      >
                        <Box display="flex" justifyContent="space-between" alignItems="center">
                          <Typography variant="body2" textTransform="capitalize">
                            {type}
                          </Typography>
                          <Chip
                            label={count}
                            size="small"
                            sx={{
                              backgroundColor: alpha(NODE_COLORS[type as keyof typeof NODE_COLORS] || NODE_COLORS.default, 0.2),
                              color: NODE_COLORS[type as keyof typeof NODE_COLORS] || NODE_COLORS.default,
                              fontWeight: 'bold',
                            }}
                          />
                        </Box>
                      </Box>
                    ))}
                  </Box>
                </Box>
              </Box>
            </GlassCard>
          </motion.div>
        </Box>
      </Box>
    </PageWrapper>
  )
}