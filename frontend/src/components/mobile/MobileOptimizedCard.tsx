import React, { ReactNode } from 'react'
import { Box, useTheme, alpha, Typography } from '@mui/material'
import { motion } from 'framer-motion'
import { designTokens } from '../../theme/designSystem'
import useResponsive from '../../hooks/useResponsive'

export interface MobileOptimizedCardProps {
  children: ReactNode
  title?: string
  subtitle?: string
  icon?: ReactNode
  variant?: 'default' | 'compact' | 'expanded'
  touchOptimized?: boolean
  swipeEnabled?: boolean
  onSwipe?: (direction: 'left' | 'right') => void
  onClick?: () => void
}

const cardVariants = {
  hidden: { opacity: 0, scale: 0.95 },
  visible: { 
    opacity: 1, 
    scale: 1,
    transition: {
      type: "spring",
      stiffness: 100,
      damping: 15,
    }
  },
}

const swipeVariants = {
  center: { x: 0, rotate: 0 },
  left: { x: -100, rotate: -5 },
  right: { x: 100, rotate: 5 },
}

export default function MobileOptimizedCard({
  children,
  title,
  subtitle,
  icon,
  variant = 'default',
  touchOptimized = true,
  swipeEnabled = false,
  onSwipe,
  onClick,
}: MobileOptimizedCardProps) {
  const theme = useTheme()
  const { isMobile, screenSize } = useResponsive()

  const getCardStyles = () => {
    const baseStyles = {
      borderRadius: isMobile ? designTokens.borderRadius.lg : designTokens.borderRadius.xl,
      background: theme.palette.mode === 'dark'
        ? 'rgba(255, 255, 255, 0.05)'
        : 'rgba(255, 255, 255, 0.8)',
      backdropFilter: 'blur(10px)',
      border: theme.palette.mode === 'dark'
        ? '1px solid rgba(255, 255, 255, 0.1)'
        : '1px solid rgba(255, 255, 255, 0.3)',
      boxShadow: designTokens.shadows.lg,
      transition: designTokens.transitions.normal,
      cursor: onClick ? 'pointer' : 'default',
      // Touch-friendly sizing
      minHeight: touchOptimized ? 44 : 'auto',
    }

    switch (variant) {
      case 'compact':
        return {
          ...baseStyles,
          p: isMobile ? 2 : 3,
          minHeight: touchOptimized ? 64 : 'auto',
        }
      case 'expanded':
        return {
          ...baseStyles,
          p: isMobile ? 3 : 4,
          minHeight: touchOptimized ? 120 : 'auto',
        }
      default:
        return {
          ...baseStyles,
          p: isMobile ? 2.5 : 3,
          minHeight: touchOptimized ? 80 : 'auto',
        }
    }
  }

  const handleSwipe = (event: any, info: any) => {
    if (!swipeEnabled || !onSwipe) return

    const threshold = 100
    if (Math.abs(info.offset.x) > threshold) {
      const direction = info.offset.x > 0 ? 'right' : 'left'
      onSwipe(direction)
    }
  }

  return (
    <motion.div
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      drag={swipeEnabled ? "x" : false}
      dragConstraints={{ left: 0, right: 0 }}
      dragElastic={0.2}
      onDragEnd={handleSwipe}
      whileTap={touchOptimized ? { scale: 0.98 } : undefined}
      whileHover={!isMobile ? { y: -2, scale: 1.02 } : undefined}
      onClick={onClick}
    >
      <Box sx={getCardStyles()}>
        {/* Header */}
        {(title || icon) && (
          <Box 
            sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: isMobile ? 1.5 : 2,
              mb: children ? 2 : 0,
            }}
          >
            {icon && (
              <Box 
                sx={{ 
                  color: 'primary.main',
                  fontSize: isMobile ? 20 : 24,
                  display: 'flex',
                  alignItems: 'center',
                  // Touch-friendly icon size
                  minWidth: touchOptimized ? 24 : 'auto',
                  minHeight: touchOptimized ? 24 : 'auto',
                }}
              >
                {icon}
              </Box>
            )}
            
            <Box sx={{ flex: 1, minWidth: 0 }}>
              {title && (
                <Typography 
                  variant={isMobile ? "body1" : "h6"}
                  sx={{ 
                    fontWeight: 600,
                    // Responsive font sizing
                    fontSize: {
                      xs: '1rem',
                      sm: '1.1rem',
                      md: '1.25rem',
                    },
                    // Text truncation for mobile
                    ...(isMobile && {
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }),
                  }}
                >
                  {title}
                </Typography>
              )}
              
              {subtitle && (
                <Typography 
                  variant="body2" 
                  color="text.secondary"
                  sx={{
                    fontSize: {
                      xs: '0.75rem',
                      sm: '0.875rem',
                    },
                    ...(isMobile && {
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }),
                  }}
                >
                  {subtitle}
                </Typography>
              )}
            </Box>
          </Box>
        )}

        {/* Content */}
        <Box 
          sx={{
            // Responsive content spacing
            '& > *:not(:last-child)': {
              mb: isMobile ? 1.5 : 2,
            },
            // Mobile-optimized text
            '& .MuiTypography-root': {
              fontSize: {
                xs: '0.875rem',
                sm: '1rem',
              },
            },
          }}
        >
          {children}
        </Box>

        {/* Swipe indicator */}
        {swipeEnabled && (
          <Box
            sx={{
              position: 'absolute',
              bottom: 8,
              right: 8,
              width: 20,
              height: 2,
              borderRadius: 1,
              background: alpha(theme.palette.text.secondary, 0.3),
              '&::after': {
                content: '""',
                position: 'absolute',
                top: -4,
                left: 0,
                right: 0,
                height: 2,
                borderRadius: 1,
                background: alpha(theme.palette.text.secondary, 0.2),
              },
            }}
          />
        )}
      </Box>
    </motion.div>
  )
}