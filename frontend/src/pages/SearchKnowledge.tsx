import { useState } from 'react'
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Tabs,
  Tab,
  Rating,
} from '@mui/material'
import { Search as SearchIcon } from '@mui/icons-material'
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

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`search-tabpanel-${index}`}
      aria-labelledby={`search-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  )
}

export default function SearchKnowledge() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [tabValue, setTabValue] = useState(0)

  const handleSearch = async (searchType: string) => {
    setLoading(true)
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
      setResults(response.data.results || [])
    } catch (error) {
      console.error('Error searching:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Search Knowledge Base
      </Typography>

      {/* Search Input */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <TextField
          fullWidth
          label="Enter your search query"
          variant="outlined"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              handleSearch(tabValue === 0 ? 'semantic' : tabValue === 1 ? 'hybrid' : 'text')
            }
          }}
          InputProps={{
            endAdornment: (
              <Button
                variant="contained"
                onClick={() => handleSearch(tabValue === 0 ? 'semantic' : tabValue === 1 ? 'hybrid' : 'text')}
                disabled={!query || loading}
              >
                <SearchIcon />
              </Button>
            ),
          }}
        />

        <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)} sx={{ mt: 2 }}>
          <Tab label="Semantic Search" />
          <Tab label="Hybrid Search" />
          <Tab label="Text Search" />
        </Tabs>
      </Paper>

      {/* Search Results */}
      {loading ? (
        <Box display="flex" justifyContent="center" p={4}>
          <CircularProgress />
        </Box>
      ) : (
        <>
          <TabPanel value={tabValue} index={0}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Using AI embeddings to find semantically similar content
            </Typography>
          </TabPanel>
          <TabPanel value={tabValue} index={1}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Combining semantic and keyword matching for best results
            </Typography>
          </TabPanel>
          <TabPanel value={tabValue} index={2}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Traditional full-text search across all content
            </Typography>
          </TabPanel>

          {results.length > 0 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                {results.length} Results Found
              </Typography>
              {results.map((result) => (
                <Card key={result.id} sx={{ mb: 2 }}>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" mb={1}>
                      <Box display="flex" gap={1} alignItems="center">
                        <Chip
                          label={result.metadata.type}
                          size="small"
                          color="primary"
                        />
                        <Chip
                          label={result.metadata.source}
                          size="small"
                          variant="outlined"
                        />
                      </Box>
                      <Rating value={result.score * 5} readOnly size="small" />
                    </Box>
                    <Typography variant="body1" paragraph>
                      {result.content}
                    </Typography>
                    <Box display="flex" gap={1} flexWrap="wrap">
                      {result.metadata.tags?.map((tag) => (
                        <Chip key={tag} label={tag} size="small" />
                      ))}
                    </Box>
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                      {new Date(result.metadata.timestamp).toLocaleString()}
                    </Typography>
                  </CardContent>
                </Card>
              ))}
            </Box>
          )}
        </>
      )}
    </Box>
  )
}