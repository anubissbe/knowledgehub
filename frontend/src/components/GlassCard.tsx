import React from 'react'
import { Card, CardProps, useTheme, alpha } from '@mui/material'
import { motion } from 'framer-motion'

// Enhanced interface with better typing
interface GlassCardProps extends Omit<CardProps, 'component'> {
  blur?: number
  opacity?: number
  gradient?: boolean
  hover?: boolean
  children: React.ReactNode
  className?: string
  'data-testid'?: string
}

const GlassCard = React.memo<GlassCardProps>(({ 
  children, 
  blur = 10, 
  opacity = 0.1,
  gradient = false,
  hover = true,
  sx,
  className,
  'data-testid': dataTestId = 'glass-card',
  ...props 
}) => {
  const theme = useTheme()
  
  // Memoized motion variants
  const hoverVariants = React.useMemo(() => ({
    hover: hover ? {
      y: -5,
      scale: 1.02,
      transition: { 
        duration: 0.3,
        ease: [0.4, 0, 0.2, 1]
      }
    } : {},
  }), [hover])

  // Memoized background styles
  const backgroundStyles = React.useMemo(() => {
    const baseBackground = alpha(theme.palette.background.paper, opacity)
    
    return gradient
      ? `linear-gradient(135deg, 
          ${baseBackground} 0%, 
          ${alpha(theme.palette.primary.main, 0.05)} 50%,
          ${alpha(theme.palette.secondary.main, 0.05)} 100%)`
      : baseBackground
  }, [theme, opacity, gradient])

  // Memoized card styles
  const cardStyles = React.useMemo(() => ({
    backdropFilter: `blur(${blur}px)`,
    backgroundColor: 'transparent',
    background: backgroundStyles,
    border: `1px solid ${alpha(theme.palette.divider, 0.2)}`,
    borderRadius: 3,
    boxShadow: `0 8px 32px ${alpha(theme.palette.common.black, 0.1)}`,
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    position: 'relative' as const,
    overflow: 'hidden' as const,
    '&:hover': hover ? {
      boxShadow: `0 12px 48px ${alpha(theme.palette.primary.main, 0.15)}`,
      borderColor: alpha(theme.palette.primary.main, 0.3),
      transform: 'translateY(-2px)',
    } : undefined,
    '&::before': {
      content: '""',
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: `linear-gradient(135deg, 
        ${alpha(theme.palette.common.white, 0.1)} 0%, 
        transparent 100%)`,
      opacity: 0.5,
      pointerEvents: 'none',
    },
    ...sx,
  }), [blur, backgroundStyles, theme, hover, sx])

  return (
    <Card
      component={motion.div}
      className={className}
      data-testid={dataTestId}
      variants={hoverVariants}
      whileHover="hover"
      sx={cardStyles}
      {...props}
    >
      {children}
    </Card>
  )
})

GlassCard.displayName = 'GlassCard'

export default GlassCard
