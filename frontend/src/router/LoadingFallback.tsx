import React from 'react'
import { Box, CircularProgress, Typography, LinearProgress } from '@mui/material'

interface LoadingFallbackProps {
  text?: string
  variant?: 'circular' | 'linear'
}

export const LoadingFallback: React.FC<LoadingFallbackProps> = ({
  text = 'Loading...',
  variant = 'circular'
}) => {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 'calc(100vh - 120px)',
        gap: 2,
        p: 3,
      }}
    >
      {variant === 'circular' ? (
        <CircularProgress size={60} />
      ) : (
        <Box sx={{ width: '100%', maxWidth: 400 }}>
          <LinearProgress />
        </Box>
      )}
      <Typography variant="h6" color="textSecondary">
        {text}
      </Typography>
    </Box>
  )
}

export default LoadingFallback
