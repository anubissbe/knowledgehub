import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Link,
} from '@mui/material';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';

const ApiDocs: React.FC = () => {
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:3000';

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        API Documentation
      </Typography>
      
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Interactive API Documentation
          </Typography>
          
          <Typography variant="body1" paragraph>
            KnowledgeHub provides comprehensive REST API documentation with an interactive interface powered by FastAPI and Swagger UI.
          </Typography>
          
          <Box sx={{ my: 3 }}>
            <Button
              variant="contained"
              size="large"
              endIcon={<OpenInNewIcon />}
              href={`${apiUrl}/docs`}
              target="_blank"
              rel="noopener noreferrer"
            >
              Open API Documentation
            </Button>
          </Box>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
            Key API Endpoints
          </Typography>
          
          <Box component="ul" sx={{ pl: 3 }}>
            <li>
              <Typography variant="body2">
                <strong>GET /health</strong> - Health check endpoint
              </Typography>
            </li>
            <li>
              <Typography variant="body2">
                <strong>GET /api/ai-features/summary</strong> - AI features overview
              </Typography>
            </li>
            <li>
              <Typography variant="body2">
                <strong>POST /api/claude-auto/session/start</strong> - Start a new session
              </Typography>
            </li>
            <li>
              <Typography variant="body2">
                <strong>POST /api/mistake-learning/track</strong> - Track and learn from mistakes
              </Typography>
            </li>
            <li>
              <Typography variant="body2">
                <strong>POST /api/decisions/record</strong> - Record technical decisions
              </Typography>
            </li>
            <li>
              <Typography variant="body2">
                <strong>POST /api/search</strong> - Search knowledge base
              </Typography>
            </li>
            <li>
              <Typography variant="body2">
                <strong>GET /api/memory</strong> - Retrieve memories
              </Typography>
            </li>
            <li>
              <Typography variant="body2">
                <strong>WS /ws</strong> - WebSocket for real-time updates
              </Typography>
            </li>
          </Box>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
            Authentication
          </Typography>
          
          <Typography variant="body2" paragraph>
            The API supports optional JWT authentication. Include the token in the Authorization header:
          </Typography>
          
          <Box sx={{ bgcolor: 'grey.100', p: 2, borderRadius: 1 }}>
            <code>Authorization: Bearer YOUR_JWT_TOKEN</code>
          </Box>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
            Example Usage
          </Typography>
          
          <Typography variant="body2" paragraph>
            Start a new session:
          </Typography>
          
          <Box sx={{ bgcolor: 'grey.100', p: 2, borderRadius: 1, mb: 2 }}>
            <pre style={{ margin: 0 }}>
{`curl -X POST "${apiUrl}/api/claude-auto/session/start?cwd=/my/project"
  -H "Content-Type: application/json"`}
            </pre>
          </Box>
          
          <Typography variant="body2" paragraph>
            Search the knowledge base:
          </Typography>
          
          <Box sx={{ bgcolor: 'grey.100', p: 2, borderRadius: 1 }}>
            <pre style={{ margin: 0 }}>
{`curl -X POST "${apiUrl}/api/search"
  -H "Content-Type: application/json"
  -d '{"query": "authentication implementation"}'`}
            </pre>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default ApiDocs;