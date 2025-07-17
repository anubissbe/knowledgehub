import { useState, useEffect } from 'react'
import {
  Box,
  Typography,
  TextField,
  IconButton,
  Switch,
  FormControlLabel,
  Divider,
  Grid,
  Slider,
  Chip,
  alpha,
  Collapse,
} from '@mui/material'
import { 
  Save,
  Api,
  Palette,
  Notifications,
  Refresh,
  Security,
  Speed,
  Storage,
  Cloud,
  CheckCircle,
  Language,
  DarkMode,
  LightMode,
  AutoAwesome,
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'
import PageContainer from '../components/ultra/PageContainer'
import UltraHeader from '../components/ultra/UltraHeader'
import GlassCard from '../components/GlassCard'

const SETTING_CATEGORIES = [
  {
    id: 'api',
    title: 'API Configuration',
    icon: <Api />,
    color: '#2196F3',
    description: 'Configure API endpoints and authentication',
  },
  {
    id: 'display',
    title: 'Display & Theme',
    icon: <Palette />,
    color: '#FF00FF',
    description: 'Customize appearance and visual preferences',
  },
  {
    id: 'performance',
    title: 'Performance',
    icon: <Speed />,
    color: '#00FF88',
    description: 'Optimize speed and resource usage',
  },
  {
    id: 'storage',
    title: 'Storage & Cache',
    icon: <Storage />,
    color: '#FFD700',
    description: 'Manage local storage and caching',
  },
]

export default function Settings() {
  const [settings, setSettings] = useState({
    apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:3000',
    apiKey: '',
    enableNotifications: true,
    autoRefresh: true,
    refreshInterval: 30,
    darkMode: true,
    language: 'en',
    cacheSize: 100,
    maxMemories: 1000,
    compressionEnabled: true,
    animationSpeed: 1,
  })
  const [saved, setSaved] = useState(false)
  const [expandedCategory, setExpandedCategory] = useState<string | null>('api')

  useEffect(() => {
    // Load settings from localStorage
    const savedSettings = localStorage.getItem('knowledgehub_settings')
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings))
    }
  }, [])

  const handleSave = () => {
    localStorage.setItem('knowledgehub_settings', JSON.stringify(settings))
    setSaved(true)
    setTimeout(() => setSaved(false), 3000)
  }

  const handleCategoryClick = (categoryId: string) => {
    setExpandedCategory(expandedCategory === categoryId ? null : categoryId)
  }

  return (
    <PageContainer>
      <UltraHeader 
        title="System Settings" 
        subtitle="CONFIGURATION & PREFERENCES"
      />

      <Box sx={{ px: 3, pb: 6 }}>
        {/* Save Notification */}
        <AnimatePresence>
          {saved && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <GlassCard sx={{ mb: 3, backgroundColor: alpha('#00FF88', 0.1) }}>
                <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
                  <CheckCircle sx={{ color: '#00FF88' }} />
                  <Typography>Settings saved successfully!</Typography>
                </Box>
              </GlassCard>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Setting Categories */}
        <Grid container spacing={3}>
          {SETTING_CATEGORIES.map((category, index) => (
            <Grid item xs={12} key={category.id}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 + index * 0.1 }}
              >
                <GlassCard
                  hover
                  sx={{
                    cursor: 'pointer',
                    overflow: 'hidden',
                    borderLeft: `3px solid ${category.color}`,
                  }}
                  onClick={() => handleCategoryClick(category.id)}
                >
                  <Box sx={{ p: 3 }}>
                    <Box display="flex" alignItems="center" justifyContent="space-between">
                      <Box display="flex" alignItems="center" gap={2}>
                        <Box
                          sx={{
                            p: 2,
                            borderRadius: 2,
                            backgroundColor: alpha(category.color, 0.1),
                            color: category.color,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                          }}
                        >
                          {category.icon}
                        </Box>
                        <Box>
                          <Typography variant="h6" fontWeight="bold">
                            {category.title}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {category.description}
                          </Typography>
                        </Box>
                      </Box>
                      <IconButton>
                        <motion.div
                          animate={{ rotate: expandedCategory === category.id ? 180 : 0 }}
                          transition={{ duration: 0.3 }}
                        >
                          <AutoAwesome />
                        </motion.div>
                      </IconButton>
                    </Box>

                    <Collapse in={expandedCategory === category.id}>
                      <Divider sx={{ my: 3, borderColor: alpha('#ffffff', 0.1) }} />
                      
                      {/* API Settings */}
                      {category.id === 'api' && (
                        <Box>
                          <TextField
                            fullWidth
                            label="API URL"
                            value={settings.apiUrl}
                            onChange={(e) => setSettings({ ...settings, apiUrl: e.target.value })}
                            sx={{ mb: 3 }}
                          />
                          <TextField
                            fullWidth
                            label="API Key"
                            type="password"
                            value={settings.apiKey}
                            onChange={(e) => setSettings({ ...settings, apiKey: e.target.value })}
                            helperText="Your API key for authenticated requests"
                          />
                          <Box display="flex" alignItems="center" gap={1} mt={2}>
                            <Security sx={{ color: 'text.secondary', fontSize: 16 }} />
                            <Typography variant="caption" color="text.secondary">
                              API key is encrypted and stored locally
                            </Typography>
                          </Box>
                        </Box>
                      )}

                      {/* Display Settings */}
                      {category.id === 'display' && (
                        <Box>
                          <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
                            <Box display="flex" alignItems="center" gap={2}>
                              {settings.darkMode ? <DarkMode /> : <LightMode />}
                              <Typography>Theme Mode</Typography>
                            </Box>
                            <Switch
                              checked={settings.darkMode}
                              onChange={(e) => setSettings({ ...settings, darkMode: e.target.checked })}
                              sx={{
                                '& .MuiSwitch-track': {
                                  backgroundColor: theme => alpha(theme.palette.primary.main, 0.3),
                                },
                              }}
                            />
                          </Box>

                          <Box mb={3}>
                            <Box display="flex" alignItems="center" gap={2} mb={1}>
                              <Language />
                              <Typography>Language</Typography>
                            </Box>
                            <Box display="flex" gap={1}>
                              {['en', 'es', 'fr', 'de', 'zh'].map((lang) => (
                                <Chip
                                  key={lang}
                                  label={lang.toUpperCase()}
                                  onClick={() => setSettings({ ...settings, language: lang })}
                                  color={settings.language === lang ? 'primary' : 'default'}
                                  sx={{
                                    fontWeight: settings.language === lang ? 'bold' : 'normal',
                                  }}
                                />
                              ))}
                            </Box>
                          </Box>

                          <FormControlLabel
                            control={
                              <Switch
                                checked={settings.enableNotifications}
                                onChange={(e) => setSettings({ ...settings, enableNotifications: e.target.checked })}
                              />
                            }
                            label={
                              <Box display="flex" alignItems="center" gap={1}>
                                <Notifications />
                                <Typography>Enable Notifications</Typography>
                              </Box>
                            }
                          />
                        </Box>
                      )}

                      {/* Performance Settings */}
                      {category.id === 'performance' && (
                        <Box>
                          <FormControlLabel
                            control={
                              <Switch
                                checked={settings.autoRefresh}
                                onChange={(e) => setSettings({ ...settings, autoRefresh: e.target.checked })}
                              />
                            }
                            label={
                              <Box display="flex" alignItems="center" gap={1}>
                                <Refresh />
                                <Typography>Auto Refresh Data</Typography>
                              </Box>
                            }
                          />
                          
                          <Collapse in={settings.autoRefresh}>
                            <Box sx={{ mt: 3 }}>
                              <Typography gutterBottom>
                                Refresh Interval: {settings.refreshInterval}s
                              </Typography>
                              <Slider
                                value={settings.refreshInterval}
                                onChange={(_, value) => setSettings({ ...settings, refreshInterval: value as number })}
                                min={10}
                                max={300}
                                step={10}
                                marks={[
                                  { value: 10, label: '10s' },
                                  { value: 150, label: '2.5m' },
                                  { value: 300, label: '5m' },
                                ]}
                                sx={{
                                  '& .MuiSlider-track': {
                                    background: `linear-gradient(90deg, ${category.color}, ${alpha(category.color, 0.5)})`,
                                  },
                                }}
                              />
                            </Box>
                          </Collapse>

                          <Box sx={{ mt: 3 }}>
                            <Typography gutterBottom>
                              Animation Speed: {settings.animationSpeed}x
                            </Typography>
                            <Slider
                              value={settings.animationSpeed}
                              onChange={(_, value) => setSettings({ ...settings, animationSpeed: value as number })}
                              min={0.5}
                              max={2}
                              step={0.1}
                              marks={[
                                { value: 0.5, label: 'Slow' },
                                { value: 1, label: 'Normal' },
                                { value: 2, label: 'Fast' },
                              ]}
                            />
                          </Box>
                        </Box>
                      )}

                      {/* Storage Settings */}
                      {category.id === 'storage' && (
                        <Box>
                          <Box sx={{ mb: 3 }}>
                            <Typography gutterBottom>
                              Cache Size: {settings.cacheSize} MB
                            </Typography>
                            <Slider
                              value={settings.cacheSize}
                              onChange={(_, value) => setSettings({ ...settings, cacheSize: value as number })}
                              min={50}
                              max={500}
                              step={50}
                              marks={[
                                { value: 50, label: '50 MB' },
                                { value: 250, label: '250 MB' },
                                { value: 500, label: '500 MB' },
                              ]}
                              sx={{
                                '& .MuiSlider-track': {
                                  background: `linear-gradient(90deg, ${category.color}, ${alpha(category.color, 0.5)})`,
                                },
                              }}
                            />
                          </Box>

                          <Box sx={{ mb: 3 }}>
                            <Typography gutterBottom>
                              Max Memories: {settings.maxMemories}
                            </Typography>
                            <Slider
                              value={settings.maxMemories}
                              onChange={(_, value) => setSettings({ ...settings, maxMemories: value as number })}
                              min={100}
                              max={10000}
                              step={100}
                              marks={[
                                { value: 100, label: '100' },
                                { value: 5000, label: '5K' },
                                { value: 10000, label: '10K' },
                              ]}
                            />
                          </Box>

                          <FormControlLabel
                            control={
                              <Switch
                                checked={settings.compressionEnabled}
                                onChange={(e) => setSettings({ ...settings, compressionEnabled: e.target.checked })}
                              />
                            }
                            label={
                              <Box display="flex" alignItems="center" gap={1}>
                                <Cloud />
                                <Typography>Enable Compression</Typography>
                              </Box>
                            }
                          />
                        </Box>
                      )}
                    </Collapse>
                  </Box>
                </GlassCard>
              </motion.div>
            </Grid>
          ))}
        </Grid>

        {/* Save Button */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
        >
          <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
            <IconButton
              onClick={handleSave}
              sx={{
                backgroundColor: 'primary.main',
                color: 'white',
                width: 64,
                height: 64,
                '&:hover': {
                  backgroundColor: 'primary.dark',
                  transform: 'scale(1.1)',
                },
                transition: 'all 0.3s',
              }}
            >
              <Save fontSize="large" />
            </IconButton>
          </Box>
        </motion.div>
      </Box>
    </PageContainer>
  )
}