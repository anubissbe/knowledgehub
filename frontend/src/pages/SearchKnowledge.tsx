import { useState } from 'react'
import {
  Box,
  Typography,
  TextField,
  IconButton,
  Chip,
  Tabs,
  Tab,
  Rating,
  alpha,
  InputAdornment,
  LinearProgress,
} from '@mui/material'
import { 
  Search as SearchIcon,
  AutoAwesome,
  Explore,
  TextFields,
  Psychology,
  TrendingUp,
  LocalOffer,
  AccessTime,
  Insights,
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'
import PageContainer from '../components/ultra/PageContainer'
import UltraHeader from '../components/ultra/UltraHeader'
import GlassCard from '../components/GlassCard'
import { api } from '../services/api'

interface SearchResult {
  id: string
  content: string
  score: number
  metadata: {
    type: string
    source: string
    timestamp: string
    tags?: string[]
  }
}


const SEARCH_TYPES = [
  { 
    id: 'semantic', 
    label: 'Semantic Search', 
    icon: <Psychology />,
    description: 'AI-powered understanding of context and meaning',
    color: '#2196F3',
  },
  { 
    id: 'hybrid', 
    label: 'Hybrid Search', 
    icon: <Insights />,
    description: 'Best of both semantic and keyword matching',
    color: '#FF00FF',
  },
  { 
    id: 'text', 
    label: 'Text Search', 
    icon: <TextFields />,
    description: 'Traditional full-text search',
    color: '#00FF88',
  },
]

const RESULT_TYPE_COLORS: Record<string, string> = {
  code: '#2196F3',
  documentation: '#00FF88',
  memory: '#FF00FF',
  decision: '#FFD700',
  workflow: '#00FFFF',
  pattern: '#8B5CF6',
  default: '#EC4899',
}

export default function SearchKnowledge() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [tabValue, setTabValue] = useState(0)
  const [searchStats, setSearchStats] = useState({
    totalResults: 0,
    searchTime: 0,
    relevanceScore: 0,
  })

  const handleSearch = async (searchType: string) => {
    if (!query.trim()) return
    
    setLoading(true)
    const startTime = Date.now()
    
    try {
      const endpoint = searchType === 'semantic' 
        ? '/api/search/semantic'
        : searchType === 'hybrid'
        ? '/api/search/hybrid'
        : '/api/search/text'

      const response = await api.post(endpoint, {
        query,
        limit: 20,
      })
      
      const searchResults = response.data.results || []
      setResults(searchResults)
      
      // Calculate stats
      const searchTime = Date.now() - startTime
      const avgScore = searchResults.reduce((acc: number, r: SearchResult) => acc + r.score, 0) / searchResults.length || 0
      
      setSearchStats({
        totalResults: searchResults.length,
        searchTime,
        relevanceScore: avgScore * 100,
      })
    } catch (error) {
      console.error('Error searching:', error)
      setResults([])
      setSearchStats({
        totalResults: 0,
        searchTime: 0,
        relevanceScore: 0,
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <PageContainer>
      <UltraHeader 
        title="Knowledge Search" 
        subtitle="INTELLIGENT INFORMATION DISCOVERY"
      />

      <Box sx={{ px: 3, pb: 6 }}>
        {/* Search Input */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <GlassCard sx={{ mb: 4 }}>
            <Box sx={{ p: 3 }}>
              <TextField
                fullWidth
                placeholder="Ask anything... I'll search across all knowledge bases"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    handleSearch(SEARCH_TYPES[tabValue].id)
                  }
                }}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    fontSize: '1.2rem',
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
                      <SearchIcon sx={{ color: 'primary.main', fontSize: 28 }} />
                    </InputAdornment>
                  ),
                  endAdornment: (
                    <IconButton
                      onClick={() => handleSearch(SEARCH_TYPES[tabValue].id)}
                      disabled={!query || loading}
                      sx={{
                        backgroundColor: 'primary.main',
                        color: 'white',
                        '&:hover': {
                          backgroundColor: 'primary.dark',
                        },
                        '&.Mui-disabled': {
                          backgroundColor: alpha('#ffffff', 0.1),
                        },
                      }}
                    >
                      {loading ? (
                        <Box
                          sx={{
                            width: 24,
                            height: 24,
                            position: 'relative',
                            '&::before': {
                              content: '""',
                              position: 'absolute',
                              inset: 0,
                              borderRadius: '50%',
                              border: '2px solid transparent',
                              borderTopColor: 'white',
                              animation: 'spin 1s linear infinite',
                            },
                            '@keyframes spin': {
                              '0%': { transform: 'rotate(0deg)' },
                              '100%': { transform: 'rotate(360deg)' },
                            },
                          }}
                        />
                      ) : (
                        <AutoAwesome />
                      )}
                    </IconButton>
                  ),
                }}
              />

              {/* Search Type Tabs */}
              <Box sx={{ mt: 3 }}>
                <Tabs 
                  value={tabValue} 
                  onChange={(_: React.SyntheticEvent, newValue: number) => setTabValue(newValue)}
                  sx={{
                    '& .MuiTabs-indicator': {
                      height: 3,
                      borderRadius: 1.5,
                      background: 'linear-gradient(90deg, #2196F3, #FF00FF)',
                    },
                    '& .MuiTab-root': {
                      textTransform: 'none',
                      fontSize: '1rem',
                      fontWeight: 600,
                      '&.Mui-selected': {
                        color: SEARCH_TYPES[tabValue].color,
                      },
                    },
                  }}
                >
                  {SEARCH_TYPES.map((type) => (
                    <Tab
                      key={type.id}
                      label={
                        <Box display="flex" alignItems="center" gap={1}>
                          {type.icon}
                          {type.label}
                        </Box>
                      }
                    />
                  ))}
                </Tabs>
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ mt: 2, fontStyle: 'italic' }}
                >
                  {SEARCH_TYPES[tabValue].description}
                </Typography>
              </Box>
            </Box>
          </GlassCard>
        </motion.div>

        {/* Search Stats */}
        {results.length > 0 && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Box display="flex" gap={2} mb={3} flexWrap="wrap">
              <GlassCard sx={{ px: 3, py: 2 }}>
                <Box display="flex" alignItems="center" gap={2}>
                  <Explore sx={{ color: 'primary.main' }} />
                  <Box>
                    <Typography variant="h6" fontWeight="bold">
                      {searchStats.totalResults}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Results Found
                    </Typography>
                  </Box>
                </Box>
              </GlassCard>
              
              <GlassCard sx={{ px: 3, py: 2 }}>
                <Box display="flex" alignItems="center" gap={2}>
                  <AccessTime sx={{ color: 'secondary.main' }} />
                  <Box>
                    <Typography variant="h6" fontWeight="bold">
                      {searchStats.searchTime}ms
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Search Time
                    </Typography>
                  </Box>
                </Box>
              </GlassCard>
              
              <GlassCard sx={{ px: 3, py: 2 }}>
                <Box display="flex" alignItems="center" gap={2}>
                  <TrendingUp sx={{ color: 'success.main' }} />
                  <Box>
                    <Typography variant="h6" fontWeight="bold">
                      {searchStats.relevanceScore.toFixed(0)}%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Avg Relevance
                    </Typography>
                  </Box>
                </Box>
              </GlassCard>
            </Box>
          </motion.div>
        )}

        {/* Search Results */}
        <AnimatePresence mode="wait">
          {loading ? (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <Box display="flex" alignItems="center" justifyContent="center" py={8}>
                <Box textAlign="center">
                  <Box
                    sx={{
                      width: 100,
                      height: 100,
                      margin: '0 auto',
                      position: 'relative',
                    }}
                  >
                    <LinearProgress
                      variant="determinate"
                      value={75}
                      sx={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        width: 80,
                        height: 80,
                        transform: 'translate(-50%, -50%)',
                        borderRadius: '50%',
                        '& .MuiLinearProgress-bar': {
                          borderRadius: '50%',
                          background: `conic-gradient(from 0deg, ${SEARCH_TYPES[tabValue].color}, transparent)`,
                          animation: 'rotate 1s linear infinite',
                        },
                        '@keyframes rotate': {
                          '0%': { transform: 'rotate(0deg)' },
                          '100%': { transform: 'rotate(360deg)' },
                        },
                      }}
                    />
                    <AutoAwesome
                      sx={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        fontSize: 40,
                        color: SEARCH_TYPES[tabValue].color,
                        animation: 'pulse 2s infinite',
                      }}
                    />
                  </Box>
                  <Typography variant="h6" sx={{ mt: 3 }}>
                    Searching with {SEARCH_TYPES[tabValue].label}...
                  </Typography>
                </Box>
              </Box>
            </motion.div>
          ) : results.length > 0 ? (
            <motion.div
              key="results"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <Box>
                {results.map((result, index) => (
                  <motion.div
                    key={result.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 + index * 0.1 }}
                  >
                    <GlassCard sx={{ mb: 3 }}>
                      <Box sx={{ p: 3 }}>
                        {/* Result Header */}
                        <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                          <Box display="flex" gap={1} alignItems="center">
                            <Chip
                              label={result.metadata?.type || result.type || 'unknown'}
                              size="small"
                              sx={{
                                backgroundColor: alpha(RESULT_TYPE_COLORS[result.metadata?.type || result.type] || RESULT_TYPE_COLORS.default, 0.2),
                                color: RESULT_TYPE_COLORS[result.metadata?.type || result.type] || RESULT_TYPE_COLORS.default,
                                fontWeight: 600,
                              }}
                            />
                            <Chip
                              label={result.metadata?.source || result.source || 'unknown'}
                              size="small"
                              variant="outlined"
                              sx={{
                                borderColor: alpha('#ffffff', 0.2),
                              }}
                            />
                          </Box>
                          <Box display="flex" alignItems="center" gap={1}>
                            <Rating 
                              value={result.score * 5} 
                              readOnly 
                              size="small"
                              sx={{
                                '& .MuiRating-iconFilled': {
                                  color: SEARCH_TYPES[tabValue].color,
                                },
                              }}
                            />
                            <Typography variant="caption" color="text.secondary">
                              {(result.score * 100).toFixed(0)}% match
                            </Typography>
                          </Box>
                        </Box>

                        {/* Result Content */}
                        <Typography 
                          variant="body1" 
                          paragraph
                          sx={{
                            lineHeight: 1.8,
                            color: 'text.primary',
                          }}
                        >
                          {result.content}
                        </Typography>

                        {/* Tags and Metadata */}
                        <Box display="flex" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={2}>
                          <Box display="flex" gap={1} flexWrap="wrap">
                            {result.metadata.tags?.map((tag) => (
                              <motion.div
                                key={tag}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                              >
                                <Chip 
                                  icon={<LocalOffer sx={{ fontSize: 14 }} />}
                                  label={tag} 
                                  size="small"
                                  sx={{
                                    backgroundColor: alpha('#ffffff', 0.05),
                                    borderColor: alpha('#ffffff', 0.1),
                                    '&:hover': {
                                      backgroundColor: alpha('#ffffff', 0.1),
                                    },
                                  }}
                                  variant="outlined"
                                />
                              </motion.div>
                            ))}
                          </Box>
                          <Typography variant="caption" color="text.secondary">
                            <AccessTime sx={{ fontSize: 14, mr: 0.5, verticalAlign: 'middle' }} />
                            {new Date(result.metadata.timestamp).toLocaleString()}
                          </Typography>
                        </Box>
                      </Box>
                    </GlassCard>
                  </motion.div>
                ))}
              </Box>
            </motion.div>
          ) : query && !loading ? (
            <motion.div
              key="no-results"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
            >
              <Box textAlign="center" py={8}>
                <SearchIcon 
                  sx={{ 
                    fontSize: 80, 
                    color: 'text.disabled',
                    opacity: 0.3,
                    mb: 2,
                  }} 
                />
                <Typography variant="h5" gutterBottom>
                  No results found
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  Try adjusting your search query or switching to a different search type
                </Typography>
              </Box>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </Box>
    </PageContainer>
  )
}