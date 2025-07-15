import { useState } from 'react'
import { Outlet, Link, useLocation } from 'react-router-dom'
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
  useTheme,
  alpha,
} from '@mui/material'
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Psychology as PsychologyIcon,
  Memory as MemoryIcon,
  Hub as HubIcon,
  Search as SearchIcon,
  Api as ApiIcon,
  Settings as SettingsIcon,
  ChevronLeft as ChevronLeftIcon,
} from '@mui/icons-material'

const drawerWidth = 240

const menuItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
  { text: 'AI Intelligence', icon: <PsychologyIcon />, path: '/ai' },
  { text: 'Memory System', icon: <MemoryIcon />, path: '/memory' },
  { text: 'Knowledge Graph', icon: <HubIcon />, path: '/knowledge-graph' },
  { text: 'Search', icon: <SearchIcon />, path: '/search' },
  { text: 'API Docs', icon: <ApiIcon />, path: '/api-docs' },
  { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
]

export default function Layout() {
  const [open, setOpen] = useState(true)
  const location = useLocation()
  const theme = useTheme()

  const handleDrawerToggle = () => {
    setOpen(!open)
  }

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{
          width: `calc(100% - ${open ? drawerWidth : 0}px)`,
          ml: `${open ? drawerWidth : 0}px`,
          transition: theme.transitions.create(['margin', 'width'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            onClick={handleDrawerToggle}
            edge="start"
            sx={{ mr: 2, ...(open && { display: 'none' }) }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div">
            KnowledgeHub - AI Intelligence System
          </Typography>
        </Toolbar>
      </AppBar>
      <Drawer
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            bgcolor: theme.palette.background.paper,
            borderRight: `1px solid ${theme.palette.divider}`,
          },
        }}
        variant="persistent"
        anchor="left"
        open={open}
      >
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            padding: theme.spacing(0, 1),
            ...theme.mixins.toolbar,
            justifyContent: 'space-between',
          }}
        >
          <Typography variant="h6" sx={{ ml: 2 }}>
            KnowledgeHub
          </Typography>
          <IconButton onClick={handleDrawerToggle}>
            <ChevronLeftIcon />
          </IconButton>
        </Box>
        <Divider />
        <List>
          {menuItems.map((item) => (
            <ListItem key={item.text} disablePadding>
              <ListItemButton
                component={Link}
                to={item.path}
                selected={location.pathname === item.path}
                sx={{
                  '&.Mui-selected': {
                    bgcolor: alpha(theme.palette.primary.main, 0.12),
                    '&:hover': {
                      bgcolor: alpha(theme.palette.primary.main, 0.18),
                    },
                  },
                }}
              >
                <ListItemIcon
                  sx={{
                    color:
                      location.pathname === item.path
                        ? theme.palette.primary.main
                        : 'inherit',
                  }}
                >
                  {item.icon}
                </ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Drawer>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          bgcolor: theme.palette.background.default,
          p: 3,
          width: `calc(100% - ${open ? drawerWidth : 0}px)`,
          ml: open ? 0 : `-${drawerWidth}px`,
          transition: theme.transitions.create(['margin', 'width'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        <Toolbar />
        <Outlet />
      </Box>
    </Box>
  )
}