import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { Outlet, Link, useLocation } from 'react-router-dom'
import { api } from '../services/api'
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  List,
  Typography,
  Divider,
  IconButton,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  useTheme as useMuiTheme,
  alpha,
  Chip,
  Avatar,
  Tooltip,
  Badge,
  LinearProgress,
  useMediaQuery,
} from '@mui/material'
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Psychology as PsychologyIcon,
  Memory as MemoryIcon,
  Hub as HubIcon,
  Search as SearchIcon,
  Source as SourceIcon,
  Api as ApiIcon,
  Settings as SettingsIcon,
  ChevronLeft as ChevronLeftIcon,
  Brightness4 as DarkModeIcon,
  Brightness7 as LightModeIcon,
  AutoAwesome as AutoAwesomeIcon,
  NotificationsOutlined,
  ElectricBolt as ElectricBoltIcon,
  CloudSync,
  TrendingUp,
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'
import { useTheme } from '../context/ThemeContext'

const drawerWidth = 280

const menuItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard', badge: null, color: '#2196F3' },
  { text: 'Ultra Modern', icon: <ElectricBoltIcon />, path: '/ultra', badge: 'NEW', color: '#00FFFF' },
  { text: 'AI Intelligence', icon: <PsychologyIcon />, path: '/ai', badge: '8 Features', color: '#FF00FF' },
  { text: 'Memory System', icon: <MemoryIcon />, path: '/memory', badge: null, color: '#00FF88' },
  { text: 'Knowledge Graph', icon: <HubIcon />, path: '/knowledge-graph', badge: 'Live', color: '#FFD700' },
  { text: 'Search', icon: <SearchIcon />, path: '/search', badge: null, color: '#8B5CF6' },
  { text: 'Sources', icon: <SourceIcon />, path: '/sources', badge: 'NEW', color: '#FF6B35' },
  { text: 'API Docs', icon: <ApiIcon />, path: '/api-docs', badge: null, color: '#EC4899' },
  { text: 'Settings', icon: <SettingsIcon />, path: '/settings', badge: null, color: '#FF3366' },
]

const Layout = React.memo(() => {
  const location = useLocation()
  const muiTheme = useMuiTheme()
  const { darkMode, toggleDarkMode } = useTheme()
  
  // Responsive breakpoints
  const isMobile = useMediaQuery(muiTheme.breakpoints.down('sm'))
  
  // Drawer state - default closed on mobile, open on desktop
  const [open, setOpen] = useState(!isMobile)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [aiStatus, setAiStatus] = useState({ status: 'active', performance: 98 })
  
  // Drawer content component
  const DrawerContent = ({ isMobileDrawer = false }) => (
    <>
      {isMobileDrawer && (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            padding: muiTheme.spacing(0, 1),
            ...muiTheme.mixins.toolbar,
            justifyContent: 'space-between',
            borderBottom: `1px solid ${alpha(muiTheme.palette.divider, 0.1)}`,
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', ml: 2 }}>
            <Box
              sx={{
                width: 40,
                height: 40,
                borderRadius: 2,
                background: `linear-gradient(135deg, ${muiTheme.palette.primary.main} 0%, ${muiTheme.palette.secondary.main} 100%)`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mr: 2,
                boxShadow: '0 4px 12px rgba(33, 150, 243, 0.3)',
              }}
            >
              <AutoAwesomeIcon sx={{ color: '#fff' }} />
            </Box>
            <Box>
              <Typography variant="h6" fontWeight="700">
                KnowledgeHub
              </Typography>
              <Typography variant="caption" color="text.secondary">
                AI Intelligence System
              </Typography>
            </Box>
          </Box>
          <IconButton onClick={handleDrawerToggle}>
            <ChevronLeftIcon />
          </IconButton>
        </Box>
      )}
      
      <Box sx={{ p: 2 }}>
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Box
            sx={{
              p: 2,
              borderRadius: 2,
              background: darkMode
                ? `linear-gradient(135deg, ${alpha(muiTheme.palette.primary.main, 0.1)} 0%, ${alpha(muiTheme.palette.primary.main, 0.05)} 100%)`
                : `linear-gradient(135deg, ${alpha(muiTheme.palette.primary.main, 0.08)} 0%, ${alpha(muiTheme.palette.primary.main, 0.03)} 100%)`,
              border: `1px solid ${alpha(muiTheme.palette.primary.main, 0.2)}`,
              mb: 2,
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
              <Box display="flex" alignItems="center" gap={1}>
                <CloudSync 
                  sx={{ 
                    fontSize: 16, 
                    color: 'primary.main',
                    animation: 'pulse 2s infinite',
                  }} 
                />
                <Typography variant="body2" fontWeight="600" color="primary">
                  AI Status: Active
                </Typography>
              </Box>
              <Chip
                icon={<TrendingUp sx={{ fontSize: 14 }} />}
                label={`${aiStatus.performance}%`}
                size="small"
                color="success"
                sx={{ height: 20, fontSize: '0.75rem' }}
              />
            </Box>
            <Typography variant="caption" color="text.secondary">
              All systems operational
            </Typography>
            <LinearProgress
              variant="determinate"
              value={aiStatus.performance}
              sx={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                right: 0,
                height: 2,
                backgroundColor: 'transparent',
                '& .MuiLinearProgress-bar': {
                  background: `linear-gradient(90deg, ${muiTheme.palette.primary.main}, ${muiTheme.palette.secondary.main})`,
                },
              }}
            />
          </Box>
        </motion.div>
      </Box>
      
      <List sx={{ px: 2 }}>
        {menuItems.map((item, index) => (
          <motion.div
            key={item.text}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 + index * 0.05 }}
          >
            <ListItem disablePadding sx={{ mb: 0.5 }}>
              <ListItemButton
                component={Link}
                to={item.path}
                selected={location.pathname === item.path}
                onClick={() => isMobile && setMobileOpen(false)}
                sx={{
                  borderRadius: 2,
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                  position: 'relative',
                  overflow: 'hidden',
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    left: 0,
                    top: 0,
                    bottom: 0,
                    width: 3,
                    backgroundColor: item.color,
                    transform: location.pathname === item.path ? 'scaleY(1)' : 'scaleY(0)',
                    transition: 'transform 0.3s',
                  },
                  '&.Mui-selected': {
                    bgcolor: alpha(item.color, 0.1),
                    '&:hover': {
                      bgcolor: alpha(item.color, 0.15),
                    },
                    '& .MuiListItemIcon-root': {
                      color: item.color,
                    },
                    '& .MuiListItemText-primary': {
                      fontWeight: 600,
                      color: item.color,
                    },
                  },
                  '&:hover': {
                    transform: 'translateX(4px)',
                    bgcolor: alpha(item.color, 0.05),
                  },
                }}
              >
                <ListItemIcon
                  sx={{
                    minWidth: 40,
                    color:
                      location.pathname === item.path
                        ? item.color
                        : muiTheme.palette.text.secondary,
                    transition: 'color 0.3s',
                  }}
                >
                  {item.icon}
                </ListItemIcon>
                <ListItemText 
                  primary={item.text}
                  primaryTypographyProps={{
                    fontSize: '0.9375rem',
                    fontWeight: location.pathname === item.path ? 600 : 500,
                  }}
                />
                {item.badge && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: 'spring', stiffness: 300 }}
                  >
                    <Chip
                      label={item.badge}
                      size="small"
                      sx={{ 
                        height: 20,
                        fontSize: '0.75rem',
                        fontWeight: 600,
                        backgroundColor: item.badge === 'NEW' 
                          ? alpha('#00FFFF', 0.2)
                          : item.badge === 'Live'
                          ? alpha('#00FF88', 0.2)
                          : alpha(item.color, 0.2),
                        color: item.badge === 'NEW' 
                          ? '#00FFFF'
                          : item.badge === 'Live'
                          ? '#00FF88'
                          : item.color,
                        border: `1px solid ${alpha(item.color, 0.3)}`,
                      }}
                    />
                  </motion.div>
                )}
              </ListItemButton>
            </ListItem>
          </motion.div>
        ))}
      </List>
      
      <Box sx={{ flexGrow: 1 }} />
      
      <Box sx={{ p: 2 }}>
        <Divider sx={{ mb: 2 }} />
        <Box
          sx={{
            p: 2,
            borderRadius: 2,
            background: alpha(muiTheme.palette.background.default, 0.5),
            textAlign: 'center',
          }}
        >
          <Typography variant="caption" color="text.secondary">
            Powered by Claude AI
          </Typography>
          <Typography variant="caption" display="block" color="text.secondary">
            v2.0.0 | Connected to LAN
          </Typography>
        </Box>
      </Box>
    </>
  )

  useEffect(() => {
    // Simulate AI status updates
    const interval = setInterval(async () => {
      try {
        const response = await api.get('/api/ai-features/status')
        setAiStatus({
          status: response.data.status || 'active',
          performance: response.data.performance || 95,
        })
      } catch (error) {
      }
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  // Update drawer state when screen size changes
  useEffect(() => {
    setOpen(!isMobile)
  }, [isMobile])

  const handleDrawerToggle = useCallback(() => {
    if (isMobile) {
      setMobileOpen(!mobileOpen)
    } else {
      setOpen(!open)
    }
  }, [isMobile, mobileOpen, open])

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <AppBar
        position="fixed"
        elevation={0}
        sx={{
          zIndex: (theme) => theme.zIndex.drawer + 1,
          transition: muiTheme.transitions.create(['margin', 'width'], {
            easing: muiTheme.transitions.easing.sharp,
            duration: muiTheme.transitions.duration.leavingScreen,
          }),
          background: darkMode 
            ? `linear-gradient(135deg, ${alpha(muiTheme.palette.background.paper, 0.8)} 0%, ${alpha(muiTheme.palette.background.paper, 0.95)} 100%)`
            : alpha('#ffffff', 0.8),
          backdropFilter: 'blur(20px)',
          borderBottom: `1px solid ${alpha(muiTheme.palette.divider, 0.1)}`,
          boxShadow: `0 4px 30px ${alpha(muiTheme.palette.common.black, 0.1)}`,
        }}
      >
        <Toolbar sx={{ justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              onClick={handleDrawerToggle}
              edge="start"
              sx={{ mr: 2, display: { xs: 'block', md: 'none' } }}
            >
              <MenuIcon />
            </IconButton>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Typography 
                variant="h6" 
                noWrap 
                component="div" 
                sx={{
                  fontWeight: 700,
                  background: `linear-gradient(135deg, ${muiTheme.palette.primary.main} 0%, ${muiTheme.palette.secondary.main} 100%)`,
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                KnowledgeHub
              </Typography>
            </motion.div>
            <AnimatePresence>
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', stiffness: 200 }}
              >
                <Chip
                  icon={<AutoAwesomeIcon />}
                  label="AI Enhanced"
                  size="small"
                  sx={{ 
                    ml: 2, 
                    fontWeight: 600,
                    background: `linear-gradient(135deg, ${alpha(muiTheme.palette.primary.main, 0.2)} 0%, ${alpha(muiTheme.palette.secondary.main, 0.2)} 100%)`,
                    border: `1px solid ${alpha(muiTheme.palette.primary.main, 0.3)}`,
                  }}
                />
              </motion.div>
            </AnimatePresence>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tooltip title="3 new notifications">
              <IconButton color="inherit">
                <Badge badgeContent={3} color="error" variant="dot">
                  <NotificationsOutlined />
                </Badge>
              </IconButton>
            </Tooltip>
            
            <Tooltip title={`Switch to ${darkMode ? 'light' : 'dark'} mode`}>
              <IconButton onClick={toggleDarkMode} color="inherit">
                {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
              </IconButton>
            </Tooltip>
            
            <Avatar 
              sx={{ 
                ml: 2,
                width: 36,
                height: 36,
                background: `linear-gradient(135deg, ${muiTheme.palette.primary.main} 0%, ${muiTheme.palette.secondary.main} 100%)`,
                fontSize: '0.875rem',
                fontWeight: 600,
              }}
            >
              AI
            </Avatar>
          </Box>
        </Toolbar>
      </AppBar>
      
      {/* Mobile drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={handleDrawerToggle}
        ModalProps={{
          keepMounted: true, // Better open performance on mobile.
        }}
        sx={{
          display: { xs: 'block', md: 'none' },
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: drawerWidth,
            bgcolor: muiTheme.palette.background.paper,
            borderRight: 'none',
            boxShadow: darkMode
              ? '4px 0 24px rgba(0,0,0,0.4)'
              : '2px 0 12px rgba(0,0,0,0.08)',
          },
        }}
      >
        <DrawerContent isMobileDrawer={true} />
      </Drawer>
      
      {/* Desktop drawer */}
      <Drawer
        sx={{
          display: { xs: 'none', md: 'block' },
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            bgcolor: muiTheme.palette.background.paper,
            borderRight: 'none',
            boxShadow: darkMode
              ? '4px 0 24px rgba(0,0,0,0.4)'
              : '2px 0 12px rgba(0,0,0,0.08)',
            top: '64px', // Height of AppBar
          },
        }}
        variant="permanent"
        anchor="left"
      >
        <DrawerContent />
      </Drawer>
      
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: { xs: 2, sm: 3 },
          transition: muiTheme.transitions.create(['margin', 'width'], {
            easing: muiTheme.transitions.easing.sharp,
            duration: muiTheme.transitions.duration.leavingScreen,
          }),
          background: darkMode
            ? `radial-gradient(ellipse at top left, ${alpha(muiTheme.palette.primary.main, 0.05)} 0%, transparent 50%),
               radial-gradient(ellipse at bottom right, ${alpha(muiTheme.palette.secondary.main, 0.05)} 0%, transparent 50%)`
            : muiTheme.palette.background.default,
          minHeight: '100vh',
        }}
      >
        <Toolbar />
        <Box
          sx={{
            animation: 'fadeIn 0.5s ease-in-out',
            '@keyframes fadeIn': {
              from: { opacity: 0, transform: 'translateY(20px)' },
              to: { opacity: 1, transform: 'translateY(0)' },
            },
          }}
        >
          <Outlet />
        </Box>
      </Box>
    </Box>
  )
})

Layout.displayName = 'Layout'

export default Layout
