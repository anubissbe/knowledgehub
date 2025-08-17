import React, { ReactNode, useState } from 'react'
import { 
  Box, 
  Drawer, 
  AppBar, 
  Toolbar, 
  Typography, 
  IconButton, 
  useTheme, 
  useMediaQuery,
  Badge,
  Avatar,
  Menu,
  MenuItem,
  Divider,
  alpha,
} from '@mui/material'
import { 
  Menu as MenuIcon, 
  Close as CloseIcon,
  Notifications,
  Settings,
  Person,
  Logout,
} from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'
import { Outlet, useLocation, Link } from 'react-router-dom'
import { designTokens } from '../../theme/designSystem'
import ModernCard from './ModernCard'

interface NavigationItem {
  label: string
  path: string
  icon: ReactNode
  badge?: number
  subItems?: NavigationItem[]
}

interface ModernLayoutProps {
  navigationItems: NavigationItem[]
  title?: string
  user?: {
    name: string
    email: string
    avatar?: string
  }
  onThemeToggle?: () => void
  notifications?: number
}

const sidebarVariants = {
  open: {
    x: 0,
    transition: {
      type: "spring",
      stiffness: 100,
      damping: 15,
    }
  },
  closed: {
    x: "-100%",
    transition: {
      type: "spring",
      stiffness: 100,
      damping: 15,
    }
  }
}

const NavigationLink = ({ 
  item, 
  isActive, 
  onClick 
}: { 
  item: NavigationItem
  isActive: boolean
  onClick: () => void 
}) => {
  const theme = useTheme()

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <Box
        component={Link}
        to={item.path}
        onClick={onClick}
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 2,
          p: 2,
          mx: 1,
          borderRadius: designTokens.borderRadius.lg,
          color: isActive ? theme.palette.primary.main : theme.palette.text.primary,
          backgroundColor: isActive 
            ? alpha(theme.palette.primary.main, 0.1)
            : 'transparent',
          textDecoration: 'none',
          transition: designTokens.transitions.normal,
          position: 'relative',
          overflow: 'hidden',
          '&:hover': {
            backgroundColor: alpha(theme.palette.primary.main, 0.05),
            transform: 'translateX(4px)',
          },
          '&::before': isActive ? {
            content: '""',
            position: 'absolute',
            left: 0,
            top: '50%',
            transform: 'translateY(-50%)',
            width: 3,
            height: '60%',
            backgroundColor: theme.palette.primary.main,
            borderRadius: '0 2px 2px 0',
          } : {},
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', color: 'inherit' }}>
          {item.icon}
        </Box>
        
        <Typography 
          variant="body2" 
          sx={{ 
            fontWeight: isActive ? 600 : 400,
            flex: 1,
          }}
        >
          {item.label}
        </Typography>

        {item.badge && (
          <Badge
            badgeContent={item.badge}
            color="error"
            sx={{
              '& .MuiBadge-badge': {
                fontSize: '0.75rem',
                minWidth: 18,
                height: 18,
              },
            }}
          />
        )}
      </Box>
    </motion.div>
  )
}

export default function ModernLayout({
  navigationItems,
  title = "KnowledgeHub",
  user,
  onThemeToggle,
  notifications = 0,
}: ModernLayoutProps) {
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))
  const location = useLocation()
  
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile)
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null)

  const handleSidebarToggle = () => {
    setSidebarOpen(!sidebarOpen)
  }

  const handleUserMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setUserMenuAnchor(event.currentTarget)
  }

  const handleUserMenuClose = () => {
    setUserMenuAnchor(null)
  }

  const handleNavigationClick = () => {
    if (isMobile) {
      setSidebarOpen(false)
    }
  }

  const sidebarWidth = 280

  const sidebarContent = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box
        sx={{
          p: 3,
          background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.primary.dark})`,
          color: theme.palette.primary.contrastText,
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            right: 0,
            width: 100,
            height: 100,
            background: 'radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%)',
            borderRadius: '50%',
            transform: 'translate(30px, -30px)',
          },
        }}
      >
        <Typography variant="h6" sx={{ fontWeight: 700, position: 'relative', zIndex: 1 }}>
          {title}
        </Typography>
        <Typography variant="caption" sx={{ opacity: 0.8, position: 'relative', zIndex: 1 }}>
          AI Intelligence Platform
        </Typography>

        {isMobile && (
          <IconButton
            onClick={handleSidebarToggle}
            sx={{ 
              position: 'absolute', 
              top: 16, 
              right: 16, 
              color: 'inherit',
              zIndex: 2,
            }}
          >
            <CloseIcon />
          </IconButton>
        )}
      </Box>

      {/* Navigation */}
      <Box sx={{ flex: 1, py: 2, overflowY: 'auto' }}>
        {navigationItems.map((item) => (
          <NavigationLink
            key={item.path}
            item={item}
            isActive={location.pathname === item.path}
            onClick={handleNavigationClick}
          />
        ))}
      </Box>

      {/* User section */}
      {user && (
        <Box sx={{ p: 2, borderTop: `1px solid ${theme.palette.divider}` }}>
          <ModernCard variant="glass" noPadding>
            <Box sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Avatar
                  src={user.avatar}
                  sx={{ width: 40, height: 40 }}
                >
                  {user.name.charAt(0).toUpperCase()}
                </Avatar>
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Typography variant="body2" sx={{ fontWeight: 600 }} noWrap>
                    {user.name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" noWrap>
                    {user.email}
                  </Typography>
                </Box>
              </Box>
            </Box>
          </ModernCard>
        </Box>
      )}
    </Box>
  )

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      {/* Mobile backdrop */}
      <AnimatePresence>
        {isMobile && sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleSidebarToggle}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: 'rgba(0, 0, 0, 0.5)',
              zIndex: theme.zIndex.drawer - 1,
            }}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      {isMobile ? (
        <Drawer
          variant="temporary"
          anchor="left"
          open={sidebarOpen}
          onClose={handleSidebarToggle}
          sx={{
            '& .MuiDrawer-paper': {
              width: sidebarWidth,
              boxSizing: 'border-box',
              border: 'none',
              background: theme.palette.background.paper,
            },
          }}
        >
          {sidebarContent}
        </Drawer>
      ) : (
        <motion.div
          variants={sidebarVariants}
          animate={sidebarOpen ? "open" : "closed"}
          style={{
            width: sidebarOpen ? sidebarWidth : 0,
            overflow: 'hidden',
            borderRight: `1px solid ${theme.palette.divider}`,
            background: theme.palette.background.paper,
          }}
        >
          <Box sx={{ width: sidebarWidth }}>
            {sidebarContent}
          </Box>
        </motion.div>
      )}

      {/* Main content */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* App Bar */}
        <AppBar
          position="sticky"
          elevation={0}
          sx={{
            backgroundColor: alpha(theme.palette.background.default, 0.8),
            backdropFilter: 'blur(10px)',
            borderBottom: `1px solid ${theme.palette.divider}`,
            color: theme.palette.text.primary,
          }}
        >
          <Toolbar>
            <IconButton
              edge="start"
              color="inherit"
              aria-label="menu"
              onClick={handleSidebarToggle}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>

            <Box sx={{ flex: 1 }} />

            {/* Actions */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {onThemeToggle && (
                <IconButton color="inherit" onClick={onThemeToggle}>
                  <Settings />
                </IconButton>
              )}

              <IconButton color="inherit">
                <Badge badgeContent={notifications} color="error">
                  <Notifications />
                </Badge>
              </IconButton>

              {user && (
                <>
                  <IconButton
                    color="inherit"
                    onClick={handleUserMenuClick}
                    sx={{ ml: 1 }}
                  >
                    <Avatar
                      src={user.avatar}
                      sx={{ width: 32, height: 32 }}
                    >
                      {user.name.charAt(0).toUpperCase()}
                    </Avatar>
                  </IconButton>

                  <Menu
                    anchorEl={userMenuAnchor}
                    open={Boolean(userMenuAnchor)}
                    onClose={handleUserMenuClose}
                    transformOrigin={{ horizontal: 'right', vertical: 'top' }}
                    anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
                    PaperProps={{
                      sx: {
                        background: alpha(theme.palette.background.paper, 0.9),
                        backdropFilter: 'blur(10px)',
                        border: `1px solid ${theme.palette.divider}`,
                        boxShadow: designTokens.shadows.lg,
                        borderRadius: designTokens.borderRadius.lg,
                        mt: 1,
                        minWidth: 200,
                      },
                    }}
                  >
                    <MenuItem onClick={handleUserMenuClose}>
                      <Person sx={{ mr: 2 }} />
                      Profile
                    </MenuItem>
                    <MenuItem onClick={handleUserMenuClose}>
                      <Settings sx={{ mr: 2 }} />
                      Settings
                    </MenuItem>
                    <Divider />
                    <MenuItem onClick={handleUserMenuClose}>
                      <Logout sx={{ mr: 2 }} />
                      Logout
                    </MenuItem>
                  </Menu>
                </>
              )}
            </Box>
          </Toolbar>
        </AppBar>

        {/* Page content */}
        <Box
          component="main"
          sx={{
            flex: 1,
            overflow: 'auto',
            background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.02)} 0%, ${alpha(theme.palette.secondary.main, 0.02)} 100%)`,
            position: 'relative',
          }}
        >
          <Outlet />
        </Box>
      </Box>
    </Box>
  )
}