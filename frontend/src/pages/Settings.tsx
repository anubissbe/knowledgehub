import { useState } from 'react'
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
  Grid,
} from '@mui/material'
import { Save } from '@mui/icons-material'

export default function Settings() {
  const [settings, setSettings] = useState({
    apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:3000',
    apiKey: '',
    enableNotifications: true,
    autoRefresh: true,
    refreshInterval: 30,
    darkMode: true,
  })
  const [saved, setSaved] = useState(false)

  const handleSave = () => {
    // In a real app, this would save to localStorage or backend
    localStorage.setItem('knowledgehub_settings', JSON.stringify(settings))
    setSaved(true)
    setTimeout(() => setSaved(false), 3000)
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>

      {saved && (
        <Alert severity="success" sx={{ mb: 3 }}>
          Settings saved successfully!
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              API Configuration
            </Typography>
            <TextField
              fullWidth
              label="API URL"
              value={settings.apiUrl}
              onChange={(e) => setSettings({ ...settings, apiUrl: e.target.value })}
              margin="normal"
            />
            <TextField
              fullWidth
              label="API Key"
              type="password"
              value={settings.apiKey}
              onChange={(e) => setSettings({ ...settings, apiKey: e.target.value })}
              margin="normal"
              helperText="Your API key for authenticated requests"
            />
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Display Preferences
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.darkMode}
                  onChange={(e) => setSettings({ ...settings, darkMode: e.target.checked })}
                />
              }
              label="Dark Mode"
            />
            <Divider sx={{ my: 2 }} />
            <FormControlLabel
              control={
                <Switch
                  checked={settings.enableNotifications}
                  onChange={(e) => setSettings({ ...settings, enableNotifications: e.target.checked })}
                />
              }
              label="Enable Notifications"
            />
            <Divider sx={{ my: 2 }} />
            <FormControlLabel
              control={
                <Switch
                  checked={settings.autoRefresh}
                  onChange={(e) => setSettings({ ...settings, autoRefresh: e.target.checked })}
                />
              }
              label="Auto Refresh Data"
            />
            {settings.autoRefresh && (
              <TextField
                fullWidth
                type="number"
                label="Refresh Interval (seconds)"
                value={settings.refreshInterval}
                onChange={(e) => setSettings({ ...settings, refreshInterval: parseInt(e.target.value) || 30 })}
                margin="normal"
                inputProps={{ min: 10, max: 300 }}
              />
            )}
          </Paper>
        </Grid>
      </Grid>

      <Box sx={{ mt: 3 }}>
        <Button
          variant="contained"
          startIcon={<Save />}
          onClick={handleSave}
          size="large"
        >
          Save Settings
        </Button>
      </Box>
    </Box>
  )
}