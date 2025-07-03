import { useState } from 'react'
import {
  Box,
  Paper,
  TextField,
  Button,
  Typography,
  List,
  Chip,
  CircularProgress,
  ToggleButtonGroup,
  ToggleButton,
  Card,
  CardContent,
  Link,
} from '@mui/material'
import { Search as SearchIcon } from '@mui/icons-material'
import { useMutation } from '@tanstack/react-query'
import ReactMarkdown from 'react-markdown'
import { api } from '@/services/api'
import { SearchResult } from '@/types'

function Search() {
  const [query, setQuery] = useState('')
  const [searchType, setSearchType] = useState<'hybrid' | 'vector' | 'keyword'>('hybrid')
  const [results, setResults] = useState<SearchResult[]>([])

  const searchMutation = useMutation({
    mutationFn: (params: { query: string; type: string }) => 
      api.search(params.query, params.type),
    onSuccess: (data) => {
      setResults(Array.isArray(data?.results) ? data.results : [])
    },
  })

  const handleSearch = () => {
    if (query.trim()) {
      searchMutation.mutate({ query, type: searchType })
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch()
    }
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Search Knowledge Base
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Enter your search query..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            InputProps={{
              startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
            }}
          />
          <Button
            variant="contained"
            onClick={handleSearch}
            disabled={!query.trim() || searchMutation.isPending}
            sx={{ minWidth: 120 }}
          >
            {searchMutation.isPending ? <CircularProgress size={24} /> : 'Search'}
          </Button>
        </Box>

        <ToggleButtonGroup
          value={searchType}
          exclusive
          onChange={(_, value) => value && setSearchType(value)}
          size="small"
        >
          <ToggleButton value="hybrid">Hybrid</ToggleButton>
          <ToggleButton value="vector">Semantic</ToggleButton>
          <ToggleButton value="keyword">Keyword</ToggleButton>
        </ToggleButtonGroup>
      </Paper>

      {searchMutation.isPending && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {results.length > 0 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Found {results.length} results
          </Typography>
          <List>
            {results.map((result, index) => (
              <Card key={index} sx={{ mb: 2 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="subtitle1" component="div">
                      <Link href={result.url} target="_blank" rel="noopener">
                        {result.source_name}
                      </Link>
                    </Typography>
                    <Box>
                      <Chip
                        label={result.chunk_type}
                        size="small"
                        sx={{ mr: 1 }}
                      />
                      <Chip
                        label={`Score: ${result.score.toFixed(3)}`}
                        size="small"
                        color="primary"
                      />
                    </Box>
                  </Box>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    <ReactMarkdown>{result.content}</ReactMarkdown>
                  </Typography>
                  {result.metadata?.title && (
                    <Typography variant="caption" color="text.secondary">
                      From: {result.metadata.title}
                    </Typography>
                  )}
                </CardContent>
              </Card>
            ))}
          </List>
        </Box>
      )}

      {searchMutation.isSuccess && results.length === 0 && (
        <Paper sx={{ p: 3, textAlign: 'center' }}>
          <Typography color="text.secondary">
            No results found for your query. Try different keywords or search type.
          </Typography>
        </Paper>
      )}
    </Box>
  )
}

export default Search