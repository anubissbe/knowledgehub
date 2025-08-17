import React, { ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { Box, CircularProgress, Typography } from '@mui/material'

interface RouteGuardProps {
  children: ReactNode
  requireAuth?: boolean
  redirectTo?: string
}

// For now, we'll use a simple auth check - this can be enhanced later
const useAuth = () => {
  const isAuthenticated = true // For now, always allow access
  const isLoading = false
  
  return { isAuthenticated, isLoading }
}

export const RouteGuard: React.FC<RouteGuardProps> = ({
  children,
  requireAuth = false,
  redirectTo = '/dashboard'
}) => {
  const { isAuthenticated, isLoading } = useAuth()
  const location = useLocation()

  if (isLoading) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '100vh',
          gap: 2,
        }}
      >
        <CircularProgress />
        <Typography>Loading...</Typography>
      </Box>
    )
  }

  if (requireAuth && !isAuthenticated) {
    // Redirect to login page with return URL  
    const returnUrl = location.pathname
    return <Navigate to={`/login?return=${returnUrl}`} replace />
  }

  return <>{children}</>
}

export default RouteGuard
