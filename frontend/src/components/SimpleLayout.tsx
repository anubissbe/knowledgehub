import React, { useState, useMemo, useCallback } from 'react'
import { Outlet, Link, useLocation } from 'react-router-dom'
import {
  Box,
  AppBar,
  Toolbar,
  IconButton,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  useTheme,
  useMediaQuery,
  Badge,
  Chip,
} from '@mui/material'
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Psychology as PsychologyIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Hub as HubIcon,
  Search as SearchIcon,
  Source as SourceIcon,
  Api as ApiIcon,
  Settings as SettingsIcon,
  AutoFixHigh as HybridRAGIcon,
  SmartToy as AgentIcon,
  Analytics as AnalyticsIcon,
  CloudDownload as WebIngestionIcon,
  AccountTree as ClusterIcon,
  Visibility as ObservabilityIcon,
} from '@mui/icons-material'

const menuItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
  { text: 'AI Intelligence', icon: <PsychologyIcon />, path: '/ai' },
  { text: 'Memory System', icon: <MemoryIcon />, path: '/memory' },
  { text: 'Hybrid Memory', icon: <SpeedIcon />, path: '/hybrid-memory' },
  { text: 'Knowledge Graph', icon: <HubIcon />, path: '/knowledge-graph' },
  { text: 'Search', icon: <SearchIcon />, path: '/search' },
  { text: 'Sources', icon: <SourceIcon />, path: '/sources' },
  { 
    text: 'Enhanced RAG', 
    icon: <HybridRAGIcon />, 
    path: '/hybrid-rag',
    isNew: true 
  },
  { 
    text: 'Agent Workflows', 
    icon: <AgentIcon />, 
    path: '/agent-workflows',
    isNew: true 
  },
  { 
    text: 'Retrieval Analytics', 
    icon: <AnalyticsIcon />, 
    path: '/retrieval-analytics',
    isNew: true 
  },
  { 
    text: 'Web Ingestion', 
    icon: <WebIngestionIcon />, 
    path: '/web-ingestion',
    isNew: true 
  },
  { 
    text: 'Memory Clusters', 
    icon: <ClusterIcon />, 
    path: '/memory-clusters',
    isNew: true 
  },
  { 
    text: 'Observability', 
    icon: <ObservabilityIcon />, 
    path: '/observability',
    isNew: true 
  },
  { text: 'API Docs', icon: <ApiIcon />, path: '/api-docs' },
  { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
]

const drawerWidth = 280

const SimpleLayout = React.memo(() => {
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()

  const handleDrawerToggle = useCallback(() => {
    setMobileOpen(!mobileOpen)
  }, [mobileOpen])

  const drawerContent = (
    <List>
      {menuItems.map((item) => (
        <ListItem key={item.path} disablePadding>
          <ListItemButton
            component={Link}
            to={item.path}
            selected={location.pathname === item.path}
            onClick={() => isMobile && setMobileOpen(false)}
          >
            <ListItemIcon>
              {item.isNew ? (
                <Badge color="error" variant="dot">
                  {item.icon}
                </Badge>
              ) : (
                item.icon
              )}
            </ListItemIcon>
            <ListItemText 
              primary={
                <Box display="flex" alignItems="center" gap={1}>
                  {item.text}
                  {item.isNew && (
                    <Chip 
                      label="NEW" 
                      color="primary" 
                      size="small" 
                      sx={{ 
                        height: 16, 
                        fontSize: '0.6rem',
                        fontWeight: 'bold'
                      }} 
                    />
                  )}
                </Box>
              }
            />
          </ListItemButton>
        </ListItem>
      ))}
    </List>
  )

  return (
    <Box sx={{ display: 'flex', width: '100%', minHeight: '100vh' }}>
      <AppBar position="fixed">
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap>
            KnowledgeHub
          </Typography>
        </Toolbar>
      </AppBar>

      {/* Mobile drawer */}
      {isMobile && (
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{
            '& .MuiDrawer-paper': { width: drawerWidth },
          }}
        >
          <Toolbar />
          {drawerContent}
        </Drawer>
      )}

      {/* Desktop sidebar */}
      {!isMobile && (
        <Box
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            '& > *': { position: 'fixed', width: drawerWidth },
          }}
        >
          <Box sx={{ pt: 8, height: '100vh', bgcolor: 'background.paper' }}>
            {drawerContent}
          </Box>
        </Box>
      )}

      {/* Main content */}
      <Box component="main" sx={{ flexGrow: 1, width: 0 }}>
        <Toolbar />
        <Box sx={{ p: 3 }}>
          <Outlet />
        </Box>
      </Box>
    </Box>
  )
})

SimpleLayout.displayName = 'SimpleLayout'

export default SimpleLayout
