import React from 'react'
import { useRouteError, isRouteErrorResponse, Link } from 'react-router-dom'
import { Box, Typography, Button, Alert, Paper } from '@mui/material'
import { Home as HomeIcon, Refresh as RefreshIcon } from '@mui/icons-material'

export const RouterErrorBoundary: React.FC = () => {
  const error = useRouteError()

  let errorMessage = 'An unexpected error occurred'
  let errorCode = '500'

  if (isRouteErrorResponse(error)) {
    errorMessage = error.statusText || error.data?.message
    errorCode = error.status.toString()
  } else if (error instanceof Error) {
    errorMessage = error.message
  }

  const handleRefresh = () => {
    window.location.reload()
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        p: 3,
        bgcolor: 'background.default',
      }}
    >
      <Paper
        elevation={3}
        sx={{
          p: 4,
          maxWidth: 600,
          width: '100%',
          textAlign: 'center',
        }}
      >
        <Typography variant="h1" color="error" gutterBottom>
          {errorCode}
        </Typography>
        
        <Typography variant="h4" gutterBottom>
          Oops! Something went wrong
        </Typography>
        
        <Alert severity="error" sx={{ mb: 3 }}>
          {errorMessage}
        </Alert>
        
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            startIcon={<HomeIcon />}
            component={Link}
            to="/dashboard"
            size="large"
          >
            Go to Dashboard
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={handleRefresh}
            size="large"
          >
            Refresh Page
          </Button>
        </Box>
      </Paper>
    </Box>
  )
}

export default RouterErrorBoundary
