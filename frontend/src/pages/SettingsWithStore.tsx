import React from 'react';
import {
  Box,
  Grid,
  Typography,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  TextField,
  Button,
  Paper,
  Divider,
  Card,
  CardContent,
  CardHeader,
  Alert,
} from '@mui/material';
import {
  Palette,
  Notifications,
  Security,
  Storage,
  Speed,
  Language,
} from '@mui/icons-material';
import { Layout } from '../components/StoreLayout';
import { 
  useTheme as useStoreTheme, 
  useUI, 
  useAuth, 
  useApp,
  useAuthActions,
  useNotifications 
} from '../store';

export default function SettingsWithStore() {
  const theme = useStoreTheme();
  const { setTheme } = useUI();
  const { user, apiKey } = useAuth();
  const { features, toggleFeature } = useApp();
  const { setApiKey } = useAuthActions();
  const { success, error } = useNotifications();

  const [localApiKey, setLocalApiKey] = React.useState(apiKey || '');

  const handleThemeChange = (newTheme: 'light' | 'dark' | 'system') => {
    setTheme(newTheme);
    success('Theme Updated', `Theme changed to ${newTheme} mode`);
  };

  const handleApiKeyUpdate = () => {
    if (!localApiKey.trim()) {
      error('Invalid API Key', 'API key cannot be empty');
      return;
    }

    try {
      setApiKey(localApiKey);
      success('API Key Updated', 'Your API key has been saved successfully');
    } catch (err) {
      error('Update Failed', 'Failed to update API key');
    }
  };

  const handleFeatureToggle = (feature: string, enabled: boolean) => {
    toggleFeature(feature, enabled);
    success(
      'Feature Updated', 
      `${feature} has been ${enabled ? 'enabled' : 'disabled'}`
    );
  };

  return (
    <Layout 
      title="Settings" 
      breadcrumbs={[
        { label: 'Home', path: '/' },
        { label: 'Settings', path: '/settings' }
      ]}
    >
      <Box sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
        <Typography variant="h4" sx={{ mb: 4, fontWeight: 600 }}>
          Settings
        </Typography>

        <Grid container spacing={3}>
          {/* Theme Settings */}
          <Grid item xs={12}>
            <Card>
              <CardHeader
                avatar={<Palette color="primary" />}
                title="Appearance"
                subheader="Customize the look and feel of the application"
              />
              <CardContent>
                <FormControlLabel
                  control={
                    <Select
                      value={theme}
                      onChange={(e) => handleThemeChange(e.target.value as any)}
                      size="small"
                      sx={{ minWidth: 120 }}
                    >
                      <MenuItem value="light">Light</MenuItem>
                      <MenuItem value="dark">Dark</MenuItem>
                      <MenuItem value="system">System</MenuItem>
                    </Select>
                  }
                  label="Theme Mode"
                  labelPlacement="start"
                  sx={{ width: '100%', justifyContent: 'space-between', m: 0 }}
                />
              </CardContent>
            </Card>
          </Grid>

          {/* API Configuration */}
          <Grid item xs={12}>
            <Card>
              <CardHeader
                avatar={<Security color="primary" />}
                title="API Configuration"
                subheader="Configure your API access and security settings"
              />
              <CardContent>
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                  <TextField
                    label="API Key"
                    value={localApiKey}
                    onChange={(e) => setLocalApiKey(e.target.value)}
                    type="password"
                    fullWidth
                    size="small"
                    placeholder="Enter your API key"
                  />
                  <Button 
                    variant="contained" 
                    onClick={handleApiKeyUpdate}
                    disabled={!localApiKey.trim() || localApiKey === apiKey}
                  >
                    Update
                  </Button>
                </Box>
                {apiKey && (
                  <Alert severity="success" sx={{ mt: 2 }}>
                    API key is configured and active
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Feature Toggles */}
          <Grid item xs={12}>
            <Card>
              <CardHeader
                avatar={<Speed color="primary" />}
                title="Features"
                subheader="Enable or disable application features"
              />
              <CardContent>
                <Grid container spacing={2}>
                  {Object.entries(features).map(([feature, enabled]) => (
                    <Grid item xs={12} sm={6} key={feature}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={enabled}
                            onChange={(e) => handleFeatureToggle(feature, e.target.checked)}
                            color="primary"
                          />
                        }
                        label={
                          <Box>
                            <Typography variant="body1">
                              {feature.replace(/([A-Z])/g, ' ').replace(/^./, str => str.toUpperCase())}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {getFeatureDescription(feature)}
                            </Typography>
                          </Box>
                        }
                        sx={{ alignItems: 'flex-start' }}
                      />
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* User Information */}
          {user && (
            <Grid item xs={12}>
              <Card>
                <CardHeader
                  title="User Information"
                  subheader="Your account details and preferences"
                />
                <CardContent>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        label="Name"
                        value={user.name}
                        disabled
                        fullWidth
                        size="small"
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        label="Email"
                        value={user.email}
                        disabled
                        fullWidth
                        size="small"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <Typography variant="body2" color="text.secondary">
                        To update your profile information, please contact support.
                      </Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Performance Information */}
          <Grid item xs={12}>
            <Card>
              <CardHeader
                avatar={<Storage color="primary" />}
                title="Performance"
                subheader="Application performance metrics"
              />
              <CardContent>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Performance monitoring helps improve the application experience.
                </Typography>
                <FormControlLabel
                  control={
                    <Switch
                      checked={features.performanceMonitoring}
                      onChange={(e) => handleFeatureToggle('performanceMonitoring', e.target.checked)}
                      color="primary"
                    />
                  }
                  label="Enable Performance Monitoring"
                />
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Layout>
  );
}

function getFeatureDescription(feature: string): string {
  const descriptions: Record<string, string> = {
    realTimeUpdates: 'Enable live data updates via WebSocket',
    advancedSearch: 'Enhanced search with AI-powered suggestions',
    contextualMemory: 'Smart memory organization and retrieval',
    aiInsights: 'AI-generated insights and recommendations',
    knowledgeGraph: 'Visual knowledge relationships',
    performanceMonitoring: 'Track application performance metrics',
  };

  return descriptions[feature] || 'Feature configuration';
}
