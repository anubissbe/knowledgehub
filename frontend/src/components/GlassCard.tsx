import { Card, CardProps, useTheme, alpha } from '@mui/material'
import { motion } from 'framer-motion'

interface GlassCardProps extends CardProps {
  blur?: number
  opacity?: number
  gradient?: boolean
  hover?: boolean
}

export default function GlassCard({ 
  children, 
  blur = 10, 
  opacity = 0.1,
  gradient = false,
  hover = true,
  sx,
  ...props 
}: GlassCardProps) {
  const theme = useTheme()
  
  return (
    <Card
      component={motion.div}
      whileHover={hover ? { 
        y: -5, 
        scale: 1.02,
        transition: { duration: 0.3 }
      } : undefined}
      sx={{
        backdropFilter: `blur(${blur}px)`,
        backgroundColor: alpha(theme.palette.background.paper, opacity),
        background: gradient
          ? `linear-gradient(135deg, 
              ${alpha(theme.palette.background.paper, opacity)} 0%, 
              ${alpha(theme.palette.primary.main, 0.05)} 50%,
              ${alpha(theme.palette.secondary.main, 0.05)} 100%)`
          : alpha(theme.palette.background.paper, opacity),
        border: `1px solid ${alpha(theme.palette.divider, 0.2)}`,
        borderRadius: 3,
        boxShadow: `0 8px 32px ${alpha(theme.palette.common.black, 0.1)}`,
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        '&:hover': hover ? {
          boxShadow: `0 12px 48px ${alpha(theme.palette.primary.main, 0.2)}`,
          borderColor: alpha(theme.palette.primary.main, 0.3),
        } : undefined,
        ...sx,
      }}
      {...props}
    >
      {children}
    </Card>
  )
}