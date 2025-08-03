import { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Tooltip,
  Grid,
  Alert,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  ExpandMore as ExpandMoreIcon,
  Language as WebsiteIcon,
  Description as DocsIcon,
  Code as RepoIcon,
  Api as ApiIcon,
  Book as WikiIcon
} from '@mui/icons-material';
import { api } from '../services/api';

interface Source {
  id: string;
  name: string;
  url: string;
  type: 'website' | 'documentation' | 'repository' | 'api' | 'wiki';
  status: 'pending' | 'crawling' | 'indexing' | 'completed' | 'error' | 'paused';
  stats?: {
    documents: number;
    chunks: number;
    errors: number;
  };
  last_scraped_at: string | null;
  created_at: string;
  config?: {
    max_depth?: number;
    max_pages?: number;
    crawl_delay?: number;
    follow_patterns?: string[];
    exclude_patterns?: string[];
  };
  scraping_status?: {
    job_status: 'processing' | 'queued';
    priority: string;
    position?: number;
  };
}

const sourceTypeIcons = {
  website: <WebsiteIcon />,
  documentation: <DocsIcon />,
  repository: <RepoIcon />,
  api: <ApiIcon />,
  wiki: <WikiIcon />
};

const statusColors = {
  pending: 'default',
  crawling: 'info',
  indexing: 'warning',
  completed: 'success',
  error: 'error',
  paused: 'secondary'
} as const;

export default function Sources() {
  const [sources, setSources] = useState<Source[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [editingSource, setEditingSource] = useState<Source | null>(null);
  const [formData, setFormData] = useState({
    name: '',
    url: '',
    type: 'website' as Source['type'],
    max_depth: 3,
    max_pages: 500,
    crawl_delay: 1.0,
    follow_patterns: '',
    exclude_patterns: ''
  });

  const fetchSources = async () => {
    try {
      setLoading(true);
      const response = await api.get('/api/v1/sources/');
      console.log('Sources API response:', response);
      const sourcesData = response.data.sources || response.data || [];
      // Ensure it's always an array
      const sourcesArray = Array.isArray(sourcesData) ? sourcesData : [];
      setSources(sourcesArray);
    } catch (err) {
      console.error('Error fetching sources:', err);
      setError('Failed to fetch sources');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSources();
    
    // Auto-refresh every 2 seconds for more responsive status updates
    const interval = setInterval(() => {
      fetchSources();
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);

  const handleOpenDialog = (source?: Source) => {
    if (source) {
      setEditingSource(source);
      setFormData({
        name: source.name,
        url: source.url,
        type: source.type,
        max_depth: source.config?.max_depth || 3,
        max_pages: source.config?.max_pages || 500,
        crawl_delay: source.config?.crawl_delay || 1.0,
        follow_patterns: source.config?.follow_patterns?.join('\n') || '',
        exclude_patterns: source.config?.exclude_patterns?.join('\n') || ''
      });
    } else {
      setEditingSource(null);
      setFormData({
        name: '',
        url: '',
        type: 'website',
        max_depth: 3,
        max_pages: 500,
        crawl_delay: 1.0,
        follow_patterns: '',
        exclude_patterns: ''
      });
    }
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setEditingSource(null);
  };

  const handleSubmit = async () => {
    try {
      const payload = {
        name: formData.name,
        url: formData.url,
        type: formData.type,
        config: {
          max_depth: formData.max_depth,
          max_pages: formData.max_pages,
          crawl_delay: formData.crawl_delay,
          follow_patterns: formData.follow_patterns ? formData.follow_patterns.split('\n').filter(p => p.trim()) : [],
          exclude_patterns: formData.exclude_patterns ? formData.exclude_patterns.split('\n').filter(p => p.trim()) : []
        }
      };

      await api.post('/api/v1/sources/', payload);
      await fetchSources();
      handleCloseDialog();
    } catch (err) {
      setError('Network error');
    }
  };

  const handleRefresh = async (sourceId: string) => {
    try {
      await api.post(`/api/v1/sources/${sourceId}/refresh`);
      await fetchSources();
    } catch (err) {
      setError('Failed to refresh source');
    }
  };

  const handleDelete = async (sourceId: string) => {
    if (!window.confirm('Are you sure you want to delete this source? This will remove all associated documents.')) {
      return;
    }

    try {
      await api.delete(`/api/v1/sources/${sourceId}`);
      await fetchSources();
    } catch (err) {
      setError('Failed to delete source');
    }
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleString();
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: { xs: 2, sm: 3, md: 4 }, maxWidth: '100%', overflow: 'hidden' }}>
      <Box display="flex" justifyContent="between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1" gutterBottom>
          Knowledge Sources
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => handleOpenDialog()}
          sx={{ ml: 'auto' }}
        >
          Add Source
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6">Total Sources</Typography>
              <Typography variant="h4">{sources.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6">Active</Typography>
              <Typography variant="h4">
                {Array.isArray(sources) ? sources.filter(s => s.status?.toLowerCase() === 'completed').length : 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6">Scraping</Typography>
              <Typography variant="h4">
                {Array.isArray(sources) ? sources.filter(s => s.scraping_status?.job_status === 'processing').length : 0}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                + {Array.isArray(sources) ? sources.filter(s => s.scraping_status?.job_status === 'queued').length : 0} queued
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6">Total Documents</Typography>
              <Typography variant="h4">
                {Array.isArray(sources) ? sources.reduce((sum, s) => sum + (s.stats?.documents || 0), 0) : 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Card>
        <CardContent>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>URL</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Documents</TableCell>
                  <TableCell>Last Scraped</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Array.isArray(sources) && sources.map((source) => (
                  <TableRow key={source.id}>
                    <TableCell>
                      <Box display="flex" alignItems="center" gap={1}>
                        {sourceTypeIcons[source.type]}
                        {source.name}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip label={source.type} size="small" />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {source.url}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      {source.scraping_status ? (
                        <Box>
                          <Chip 
                            label={`${source.scraping_status.job_status === 'processing' ? 'Scraping' : 'Queued'} (${source.scraping_status.priority})`}
                            color="warning"
                            size="small"
                            sx={{ mb: 0.5 }}
                          />
                          {source.scraping_status.position && (
                            <Typography variant="caption" display="block">
                              Position: {source.scraping_status.position}
                            </Typography>
                          )}
                        </Box>
                      ) : (
                        <Chip 
                          label={source.status} 
                          color={statusColors[source.status]}
                          size="small"
                        />
                      )}
                    </TableCell>
                    <TableCell>{source.stats?.documents || 0}</TableCell>
                    <TableCell>{formatDate(source.last_scraped_at)}</TableCell>
                    <TableCell>
                      <Box display="flex" gap={1}>
                        <Tooltip title="Refresh">
                          <IconButton
                            size="small"
                            onClick={() => handleRefresh(source.id)}
                            disabled={source.status === 'crawling'}
                          >
                            <RefreshIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Edit">
                          <IconButton
                            size="small"
                            onClick={() => handleOpenDialog(source)}
                          >
                            <EditIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete">
                          <IconButton
                            size="small"
                            onClick={() => handleDelete(source.id)}
                            color="error"
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Add/Edit Source Dialog */}
      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="md" fullWidth>
        <DialogTitle>
          {editingSource ? 'Edit Source' : 'Add New Source'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="URL"
                value={formData.url}
                onChange={(e) => setFormData({ ...formData, url: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Type</InputLabel>
                <Select
                  value={formData.type}
                  label="Type"
                  onChange={(e) => setFormData({ ...formData, type: e.target.value as Source['type'] })}
                >
                  <MenuItem value="website">Website</MenuItem>
                  <MenuItem value="documentation">Documentation</MenuItem>
                  <MenuItem value="repository">Repository</MenuItem>
                  <MenuItem value="api">API</MenuItem>
                  <MenuItem value="wiki">Wiki</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Advanced Configuration</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="Max Depth"
                        type="number"
                        value={formData.max_depth}
                        onChange={(e) => setFormData({ ...formData, max_depth: parseInt(e.target.value) || 3 })}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="Max Pages"
                        type="number"
                        value={formData.max_pages}
                        onChange={(e) => setFormData({ ...formData, max_pages: parseInt(e.target.value) || 500 })}
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Crawl Delay (seconds)"
                        type="number"
                        inputProps={{ step: 0.1 }}
                        value={formData.crawl_delay}
                        onChange={(e) => setFormData({ ...formData, crawl_delay: parseFloat(e.target.value) || 1.0 })}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="Follow Patterns (one per line)"
                        multiline
                        rows={4}
                        value={formData.follow_patterns}
                        onChange={(e) => setFormData({ ...formData, follow_patterns: e.target.value })}
                        placeholder=".*\.html$&#10;.*\.md$"
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="Exclude Patterns (one per line)"
                        multiline
                        rows={4}
                        value={formData.exclude_patterns}
                        onChange={(e) => setFormData({ ...formData, exclude_patterns: e.target.value })}
                        placeholder=".*/_static/.*&#10;.*/assets/.*"
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button onClick={handleSubmit} variant="contained">
            {editingSource ? 'Update' : 'Add'} Source
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}