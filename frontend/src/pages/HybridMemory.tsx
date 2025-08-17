import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Alert,
  CircularProgress,
  Grid,
  Paper,
  Divider,
  IconButton,
  Tooltip,
  LinearProgress,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Sync as SyncIcon,
  Storage as StorageIcon,
  Analytics as AnalyticsIcon,
  Refresh as RefreshIcon,
  CloudUpload as CloudUploadIcon,
  CloudDone as CloudDoneIcon,
  CloudOff as CloudOffIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';
import { api } from '../services/api';
import PageWrapper from '../components/PageWrapper';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const HybridMemory: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Quick Store state
  const [content, setContent] = useState('');
  const [memoryType, setMemoryType] = useState('general');
  const [project, setProject] = useState('');
  const [tags, setTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState('');

  // Quick Recall state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);

  // Stats state
  const [cacheStats, setCacheStats] = useState<any>(null);
  const [syncStatus, setSyncStatus] = useState<any>(null);

  // Optimization state
  const [optimizeContent, setOptimizeContent] = useState('');
  const [optimizationResult, setOptimizationResult] = useState<any>(null);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5000); // Refresh every 5s
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const [cache, sync] = await Promise.all([
        api.get('/api/hybrid/cache/stats'),
        api.get('/api/hybrid/sync/status'),
      ]);
      setCacheStats(cache.data);
      setSyncStatus(sync.data);
    } catch (err) {
    }
  };

  const handleQuickStore = async () => {
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await api.post('/api/hybrid/quick-store', {
        content,
        type: memoryType,
        project: project || undefined,
        tags: tags.length > 0 ? tags : undefined,
      });

      setSuccess(
        `Memory stored! ID: ${response.data.memory_id} | Saved ${response.data.tokens_saved} tokens`
      );
      setContent('');
      setTags([]);
      fetchStats();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to store memory');
    } finally {
      setLoading(false);
    }
  };

  const handleQuickRecall = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await api.get('/api/hybrid/quick-recall', {
        params: {
          query: searchQuery,
          type: memoryType !== 'general' ? memoryType : undefined,
          project: project || undefined,
          limit: 20,
        },
      });

      setSearchResults(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to search memories');
    } finally {
      setLoading(false);
    }
  };

  const handleOptimize = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await api.post('/api/hybrid/optimize', {
        content: optimizeContent,
        target_reduction: 60,
        preserve_code: true,
      });

      setOptimizationResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to optimize content');
    } finally {
      setLoading(false);
    }
  };

  const handleAddTag = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && tagInput.trim()) {
      setTags([...tags, tagInput.trim()]);
      setTagInput('');
    }
  };

  const handleDeleteTag = (tagToDelete: string) => {
    setTags(tags.filter((tag) => tag !== tagToDelete));
  };

  const handleForceSync = async () => {
    setLoading(true);
    try {
      await api.post('/api/hybrid/sync/force');
      setSuccess('Sync initiated');
      fetchStats();
    } catch (err: any) {
      setError('Failed to force sync');
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageWrapper>
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" sx={{ mb: 3, display: 'flex', alignItems: 'center' }}>
          <SpeedIcon sx={{ mr: 1, fontSize: 40 }} />
          Hybrid Memory System
        </Typography>

        {/* Performance Stats Cards */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <TrendingUpIcon color="primary" />
                  <Typography variant="h6" sx={{ ml: 1 }}>
                    Cache Hit Rate
                  </Typography>
                </Box>
                <Typography variant="h4">
                  {cacheStats?.cache_hit_rate || '0%'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Local: {cacheStats?.local_hit_rate || '0%'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <StorageIcon color="success" />
                  <Typography variant="h6" sx={{ ml: 1 }}>
                    Token Savings
                  </Typography>
                </Box>
                <Typography variant="h4">
                  {cacheStats?.token_savings?.toLocaleString() || '0'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total saved
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <SyncIcon color={syncStatus?.is_syncing ? 'warning' : 'info'} />
                  <Typography variant="h6" sx={{ ml: 1 }}>
                    Sync Status
                  </Typography>
                </Box>
                <Typography variant="h4">
                  {syncStatus?.is_syncing ? 'Syncing...' : 'Idle'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Pending: {syncStatus?.pending_count || 0}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <AnalyticsIcon color="secondary" />
                  <Typography variant="h6" sx={{ ml: 1 }}>
                    Total Queries
                  </Typography>
                </Box>
                <Typography variant="h4">
                  {cacheStats?.total_queries?.toLocaleString() || '0'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  All time
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Alerts */}
        {error && (
          <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        {success && (
          <Alert severity="success" onClose={() => setSuccess(null)} sx={{ mb: 2 }}>
            {success}
          </Alert>
        )}

        {/* Main Tabs */}
        <Paper sx={{ width: '100%' }}>
          <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)}>
            <Tab label="Quick Store" icon={<CloudUploadIcon />} />
            <Tab label="Quick Recall" icon={<MemoryIcon />} />
            <Tab label="Token Optimizer" icon={<SpeedIcon />} />
            <Tab label="Sync Control" icon={<SyncIcon />} />
          </Tabs>

          <TabPanel value={tabValue} index={0}>
            <Box sx={{ maxWidth: 800 }}>
              <TextField
                fullWidth
                multiline
                rows={4}
                label="Content"
                value={content}
                onChange={(e) => setContent(e.target.value)}
                sx={{ mb: 2 }}
              />

              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={12} sm={4}>
                  <FormControl fullWidth>
                    <InputLabel>Memory Type</InputLabel>
                    <Select
                      value={memoryType}
                      onChange={(e) => setMemoryType(e.target.value)}
                      label="Memory Type"
                    >
                      <MenuItem value="general">General</MenuItem>
                      <MenuItem value="code">Code</MenuItem>
                      <MenuItem value="error">Error</MenuItem>
                      <MenuItem value="decision">Decision</MenuItem>
                      <MenuItem value="documentation">Documentation</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={4}>
                  <TextField
                    fullWidth
                    label="Project"
                    value={project}
                    onChange={(e) => setProject(e.target.value)}
                  />
                </Grid>

                <Grid item xs={12} sm={4}>
                  <TextField
                    fullWidth
                    label="Add Tag (Enter)"
                    value={tagInput}
                    onChange={(e) => setTagInput(e.target.value)}
                    onKeyPress={handleAddTag}
                  />
                </Grid>
              </Grid>

              <Box sx={{ mb: 2 }}>
                {tags.map((tag) => (
                  <Chip
                    key={tag}
                    label={tag}
                    onDelete={() => handleDeleteTag(tag)}
                    sx={{ mr: 1, mb: 1 }}
                  />
                ))}
              </Box>

              <Button
                variant="contained"
                onClick={handleQuickStore}
                disabled={!content || loading}
                startIcon={<CloudUploadIcon />}
              >
                Store Memory (&lt; 100ms)
              </Button>
            </Box>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <Box sx={{ maxWidth: 1200 }}>
              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <TextField
                  fullWidth
                  label="Search Query"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleQuickRecall()}
                />
                <Button
                  variant="contained"
                  onClick={handleQuickRecall}
                  disabled={!searchQuery || loading}
                  startIcon={<MemoryIcon />}
                >
                  Search
                </Button>
              </Box>

              {searchResults.length > 0 && (
                <TableContainer component={Paper}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Content</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Created</TableCell>
                        <TableCell>Accessed</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {searchResults.map((result) => (
                        <TableRow key={result.id}>
                          <TableCell>{result.content.substring(0, 100)}...</TableCell>
                          <TableCell>
                            <Chip label={result.type} size="small" />
                          </TableCell>
                          <TableCell>
                            {new Date(result.created_at).toLocaleDateString()}
                          </TableCell>
                          <TableCell>{result.access_count}x</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Box>
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            <Box sx={{ maxWidth: 800 }}>
              <TextField
                fullWidth
                multiline
                rows={6}
                label="Content to Optimize"
                value={optimizeContent}
                onChange={(e) => setOptimizeContent(e.target.value)}
                sx={{ mb: 2 }}
              />

              <Button
                variant="contained"
                onClick={handleOptimize}
                disabled={!optimizeContent || loading}
                startIcon={<SpeedIcon />}
              >
                Optimize Tokens
              </Button>

              {optimizationResult && (
                <Box sx={{ mt: 3 }}>
                  <Alert severity="success" sx={{ mb: 2 }}>
                    Saved {optimizationResult.savings_percentage.toFixed(1)}% (
                    {optimizationResult.original_tokens -
                      optimizationResult.optimized_tokens}{' '}
                    tokens)
                  </Alert>

                  <Typography variant="h6" sx={{ mb: 1 }}>
                    Optimized Content:
                  </Typography>
                  <Paper sx={{ p: 2, bgcolor: 'grey.100' }}>
                    <Typography
                      variant="body2"
                      sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}
                    >
                      {optimizationResult.optimized_content}
                    </Typography>
                  </Paper>

                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Strategies: {optimizationResult.strategies_applied.join(', ')}
                  </Typography>
                </Box>
              )}
            </Box>
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            <Box sx={{ maxWidth: 600 }}>
              <Card sx={{ mb: 2 }}>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2 }}>
                    Sync Status
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    {syncStatus?.is_syncing ? (
                      <>
                        <CircularProgress size={20} sx={{ mr: 1 }} />
                        <Typography>Syncing in progress...</Typography>
                      </>
                    ) : (
                      <>
                        <CloudDoneIcon color="success" sx={{ mr: 1 }} />
                        <Typography>Sync idle</Typography>
                      </>
                    )}
                  </Box>

                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Last Sync
                      </Typography>
                      <Typography>
                        {syncStatus?.last_sync
                          ? new Date(syncStatus.last_sync).toLocaleString()
                          : 'Never'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Pending Items
                      </Typography>
                      <Typography>{syncStatus?.pending_count || 0}</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Failed Items
                      </Typography>
                      <Typography color="error">
                        {syncStatus?.failed_count || 0}
                      </Typography>
                    </Grid>
                  </Grid>

                  <Button
                    variant="outlined"
                    onClick={handleForceSync}
                    disabled={loading || syncStatus?.is_syncing}
                    startIcon={<SyncIcon />}
                    sx={{ mt: 2 }}
                  >
                    Force Sync
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2 }}>
                    Performance Metrics
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Local Hits
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={
                        cacheStats?.performance?.local_hits
                          ? (cacheStats.performance.local_hits /
                              cacheStats.total_queries) *
                            100
                          : 0
                      }
                    />
                  </Box>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Cache Hits
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={
                        cacheStats?.performance?.cache_hits
                          ? (cacheStats.performance.cache_hits /
                              cacheStats.total_queries) *
                            100
                          : 0
                      }
                      color="secondary"
                    />
                  </Box>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Remote Hits
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={
                        cacheStats?.performance?.remote_hits
                          ? (cacheStats.performance.remote_hits /
                              cacheStats.total_queries) *
                            100
                          : 0
                      }
                      color="warning"
                    />
                  </Box>
                </CardContent>
              </Card>
            </Box>
          </TabPanel>
        </Paper>
      </Box>
    </PageWrapper>
  );
};

export default HybridMemory;