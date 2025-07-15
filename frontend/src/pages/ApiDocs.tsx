import { Box, Paper, Typography, Button } from '@mui/material'
import { OpenInNew } from '@mui/icons-material'

export default function ApiDocs() {
  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:3000'

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        API Documentation
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Interactive API Documentation
        </Typography>
        <Typography variant="body1" paragraph>
          KnowledgeHub provides comprehensive API documentation with interactive testing capabilities.
        </Typography>
        <Button
          variant="contained"
          startIcon={<OpenInNew />}
          href={`${apiUrl}/docs`}
          target="_blank"
          rel="noopener noreferrer"
        >
          Open Swagger UI
        </Button>
      </Paper>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          API Endpoints Overview
        </Typography>
        <Typography variant="body2" component="pre" sx={{ overflow: 'auto' }}>
{`Core Endpoints:
- GET    /health                          - System health check
- GET    /api/metrics/dashboard           - Dashboard metrics

Memory System:
- GET    /api/memories                    - List memories
- POST   /api/memories                    - Create memory
- POST   /api/memories/search             - Search memories
- GET    /api/memories/{id}               - Get memory by ID

AI Intelligence:
- POST   /api/claude-auto/session         - Session management
- POST   /api/mistake-learning            - Record mistakes
- POST   /api/proactive/suggestions       - Get AI suggestions
- POST   /api/decisions                   - Record decisions
- GET    /api/performance/metrics         - Performance data

Knowledge Graph:
- GET    /api/knowledge-graph/stats       - Graph statistics
- POST   /api/knowledge-graph/query       - Execute Cypher query
- POST   /api/knowledge-graph/add-node    - Add node
- POST   /api/knowledge-graph/add-edge    - Add relationship

Search:
- POST   /api/search/semantic             - Semantic search
- POST   /api/search/hybrid               - Hybrid search
- POST   /api/search/text                 - Full-text search`}
        </Typography>
      </Paper>

      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Authentication
        </Typography>
        <Typography variant="body1" paragraph>
          The API supports Bearer token authentication. Include your API key in the Authorization header:
        </Typography>
        <Typography variant="body2" component="pre" sx={{ bgcolor: 'background.default', p: 2, borderRadius: 1 }}>
{`Authorization: Bearer YOUR_API_KEY`}
        </Typography>
      </Paper>
    </Box>
  )
}