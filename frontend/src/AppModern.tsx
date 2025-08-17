import React, { useState, useEffect, Suspense } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { ThemeProvider, CssBaseline } from '@mui/material'
import { motion, AnimatePresence } from 'framer-motion'

// Theme and design system
import { lightTheme, darkTheme } from './theme/designSystem'
import ModernLayout from './components/modern/ModernLayout'

// Icons
import {
  Dashboard as DashboardIcon,
  Psychology as AIIcon,
  Memory as MemoryIcon,
  Search as SearchIcon,
  Source as SourceIcon,
  Settings as SettingsIcon,
  GraphicEq as GraphIcon,
  Description as DocsIcon,
  TrendingUp as AnalyticsIcon,
  Business as EnterpriseIcon,
} from '@mui/icons-material'

// Services and utilities
import { pwaManager, isPWACapable } from './utils/pwaUtils'
import webSocketService from './services/websocketService'
import { apiService } from './services/apiService'

// Lazy load pages for better performance
const ModernDashboardV2 = React.lazy(() => import('./pages/ModernDashboardV2'))
const AiIntelligenceModern = React.lazy(() => import('./pages/AiIntelligenceModern'))
const MemorySystemModern = React.lazy(() => import('./pages/MemorySystemModern'))
const SearchKnowledgeModern = React.lazy(() => import('./pages/SearchKnowledgeModern'))
const SourcesModern = React.lazy(() => import('./pages/SourcesModern'))
const SettingsModern = React.lazy(() => import('./pages/SettingsModern'))
const KnowledgeGraphModern = React.lazy(() => import('./pages/KnowledgeGraphModern'))
const ApiDocsModern = React.lazy(() => import('./pages/ApiDocsModern'))
const AnalyticsModern = React.lazy(() => import('./pages/AnalyticsModern'))
const EnterpriseModern = React.lazy(() => import('./pages/EnterpriseModern'))

// Loading component
const PageLoader = () => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100vh',
      flexDirection: 'column',
      gap: '1rem',
    }}
  >
    <motion.div
      animate={{ rotate: 360 }}
      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
      style={{
        width: 40,
        height: 40,
        border: '3px solid rgba(33, 150, 243, 0.3)',
        borderTop: '3px solid #2196F3',
        borderRadius: '50%',
      }}
    />
    <motion.p
      animate={{ opacity: [0.5, 1, 0.5] }}
      transition={{ duration: 1.5, repeat: Infinity }}
      style={{ color: '#666', fontSize: '0.875rem' }}
    >
      Loading KnowledgeHub...
    </motion.p>
  </motion.div>
)

interface AppState {
  themeMode: 'light' | 'dark'
  isOnline: boolean
  pwaInstallPrompt: boolean
  systemHealth: 'healthy' | 'degraded' | 'critical'
  notifications: number
}

export default function AppModern() {
  const [state, setState] = useState<AppState>({
    themeMode: 'dark',
    isOnline: navigator.onLine,
    pwaInstallPrompt: false,
    systemHealth: 'healthy',
    notifications: 0,
  })

  // Navigation configuration
  const navigationItems = [
    {
      label: 'Dashboard',
      path: '/dashboard',
      icon: <DashboardIcon />,
    },
    {
      label: 'AI Intelligence',
      path: '/ai',
      icon: <AIIcon />,
      badge: state.notifications > 0 ? state.notifications : undefined,
    },
    {
      label: 'Memory System',
      path: '/memory',
      icon: <MemoryIcon />,
    },
    {
      label: 'Knowledge Graph',
      path: '/knowledge-graph',
      icon: <GraphIcon />,
    },
    {
      label: 'Search',
      path: '/search',
      icon: <SearchIcon />,
    },
    {
      label: 'Analytics',
      path: '/analytics',
      icon: <AnalyticsIcon />,
    },
    {
      label: 'Enterprise',
      path: '/enterprise',
      icon: <EnterpriseIcon />,
    },
    {
      label: 'Sources',
      path: '/sources',
      icon: <SourceIcon />,
    },
    {
      label: 'API Docs',
      path: '/api-docs',
      icon: <DocsIcon />,
    },
    {
      label: 'Settings',
      path: '/settings',
      icon: <SettingsIcon />,
    },
  ]


  // Initialize services and event handlers
  useEffect(() => {
    const init = async () => {
      try {
        // Initialize PWA features if supported
        if (isPWACapable()) {
          
          // Check for install prompt
          const unsubscribeInstall = pwaManager.onInstallStateChange((canInstall) => {
            setState(prev => ({ ...prev, pwaInstallPrompt: canInstall }))
          })

          // Monitor online status
          const unsubscribeOnline = pwaManager.onOnlineStatusChange((isOnline) => {
            setState(prev => ({ ...prev, isOnline }))
          })

          // Request notification permission
          await pwaManager.requestNotificationPermission()

          return () => {
            unsubscribeInstall()
            unsubscribeOnline()
          }
        }

        // Initialize WebSocket connection
        try {
          await webSocketService.connect()
        } catch (error) {
        }

        // Load initial system health
        try {
          const healthResponse = await apiService.getSystemHealth()
          setState(prev => ({ 
            ...prev, 
            systemHealth: healthResponse.data.status || 'healthy' 
          }))
        } catch (error) {
        }

      } catch (error) {
      }
    }

    init()

    // Load theme preference
    const savedTheme = localStorage.getItem('knowledgehub_theme') as 'light' | 'dark'
    if (savedTheme) {
      setState(prev => ({ ...prev, themeMode: savedTheme }))
    }

    return () => {
      webSocketService.disconnect()
    }
  }, [])

  // WebSocket event handlers
  useEffect(() => {
    if (!webSocketService) return

    const unsubscribers = [
      // System alerts
      webSocketService.on('system_alert', (alert) => {
        if (alert.severity === 'error' || alert.severity === 'warning') {
          setState(prev => ({ ...prev, notifications: prev.notifications + 1 }))
          
          // Show PWA notification if available
          pwaManager.showNotification('System Alert', {
            body: alert.message,
            icon: '/icons/icon-192x192.png',
            badge: '/icons/icon-96x96.png',
            tag: 'system-alert',
          })
        }
      }),

      // Health updates
      webSocketService.on('health_update', (health) => {
        setState(prev => ({ ...prev, systemHealth: health.status }))
      }),
    ]

    return () => {
      unsubscribers.forEach(unsub => unsub())
    }
  }, [])

  const handleThemeToggle = () => {
    const newTheme = state.themeMode === 'dark' ? 'light' : 'dark'
    setState(prev => ({ ...prev, themeMode: newTheme }))
    localStorage.setItem('knowledgehub_theme', newTheme)
  }

  const theme = state.themeMode === 'dark' ? darkTheme : lightTheme

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      
      <AnimatePresence mode="wait">
        <motion.div
          key={state.themeMode}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          style={{ minHeight: '100vh' }}
        >
          <Routes>
            <Route 
              path="/" 
              element={
                <ModernLayout
                  navigationItems={navigationItems}
                  title="KnowledgeHub"
                  
                  onThemeToggle={handleThemeToggle}
                  notifications={state.notifications}
                />
              }
            >
              <Route index element={<Navigate to="/dashboard" replace />} />
              
              <Route 
                path="dashboard" 
                element={
                  <Suspense fallback={<PageLoader />}>
                    <ModernDashboardV2 />
                  </Suspense>
                } 
              />
              
              <Route 
                path="ai" 
                element={
                  <Suspense fallback={<PageLoader />}>
                    <AiIntelligenceModern />
                  </Suspense>
                } 
              />
              
              <Route 
                path="memory" 
                element={
                  <Suspense fallback={<PageLoader />}>
                    <MemorySystemModern />
                  </Suspense>
                } 
              />
              
              <Route 
                path="knowledge-graph" 
                element={
                  <Suspense fallback={<PageLoader />}>
                    <KnowledgeGraphModern />
                  </Suspense>
                } 
              />
              
              <Route 
                path="search" 
                element={
                  <Suspense fallback={<PageLoader />}>
                    <SearchKnowledgeModern />
                  </Suspense>
                } 
              />
              
              <Route 
                path="analytics" 
                element={
                  <Suspense fallback={<PageLoader />}>
                    <AnalyticsModern />
                  </Suspense>
                } 
              />
              
              <Route 
                path="enterprise" 
                element={
                  <Suspense fallback={<PageLoader />}>
                    <EnterpriseModern />
                  </Suspense>
                } 
              />
              
              <Route 
                path="sources" 
                element={
                  <Suspense fallback={<PageLoader />}>
                    <SourcesModern />
                  </Suspense>
                } 
              />
              
              <Route 
                path="api-docs" 
                element={
                  <Suspense fallback={<PageLoader />}>
                    <ApiDocsModern />
                  </Suspense>
                } 
              />
              
              <Route 
                path="settings" 
                element={
                  <Suspense fallback={<PageLoader />}>
                    <SettingsModern />
                  </Suspense>
                } 
              />
            </Route>
          </Routes>
        </motion.div>
      </AnimatePresence>

      {/* PWA Install Prompt */}
      {state.pwaInstallPrompt && (
        <motion.div
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          style={{
            position: 'fixed',
            bottom: 16,
            right: 16,
            zIndex: 1000,
            background: theme.palette.primary.main,
            color: theme.palette.primary.contrastText,
            padding: '12px 16px',
            borderRadius: 8,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            cursor: 'pointer',
          }}
          onClick={() => pwaManager.showInstallPrompt()}
        >
          <div style={{ fontSize: '0.875rem', fontWeight: 500 }}>
            📱 Install KnowledgeHub
          </div>
          <div style={{ fontSize: '0.75rem', opacity: 0.9, marginTop: 4 }}>
            Get the native app experience
          </div>
        </motion.div>
      )}

      {/* Offline Indicator */}
      {!state.isOnline && (
        <motion.div
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            background: '#FF9800',
            color: 'white',
            padding: '8px 16px',
            textAlign: 'center',
            fontSize: '0.875rem',
            fontWeight: 500,
            zIndex: 2000,
          }}
        >
          📶 You're offline - Some features may be limited
        </motion.div>
      )}
    </ThemeProvider>
  )
}