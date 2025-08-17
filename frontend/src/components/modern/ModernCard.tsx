import React, { ReactNode } from 'react'
import { Box, Card, CardContent, useTheme, alpha, SxProps, Theme } from '@mui/material'
import { motion } from 'framer-motion'
import { designTokens } from '../../theme/designSystem'

export interface ModernCardProps {
  children: ReactNode
  variant?: 'default' | 'glass' | 'elevated' | 'neon'
  hover?: boolean
  animated?: boolean
  noPadding?: boolean
  sx?: SxProps<Theme>
  delay?: number
}

const cardVariants = {
  hidden: { 
    opacity: 0, 
    y: 20,
    scale: 0.95,
  },
  visible: { 
    opacity: 1, 
    y: 0,
    scale: 1,
    transition: {
      type: "spring",
      stiffness: 100,
      damping: 15,
    }
  },
  hover: {
    y: -5,
    scale: 1.02,
    transition: {
      type: "spring",
      stiffness: 400,
      damping: 10,
    }
  }
}

export default function ModernCard({
  children,
  variant = 'default',
  hover = true,
  animated = true,
  noPadding = false,
  sx = {},
  delay = 0,
}: ModernCardProps) {
  const theme = useTheme()
  const isDark = theme.palette.mode === 'dark'

  const getCardStyles = () => {
    const baseStyles = {
      position: 'relative',
      overflow: 'hidden',
      cursor: hover ? 'pointer' : 'default',
      transition: designTokens.transitions.normal,
    }

    switch (variant) {
      case 'glass':
        return {
          ...baseStyles,
          background: isDark
            ? 'rgba(255, 255, 255, 0.05)'
            : 'rgba(255, 255, 255, 0.7)',
          backdropFilter: 'blur(20px)',
          border: isDark
            ? '1px solid rgba(255, 255, 255, 0.1)'
            : '1px solid rgba(255, 255, 255, 0.3)',
          borderRadius: designTokens.borderRadius.xl,
          boxShadow: designTokens.shadows.glass,
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '1px',
            background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)',
          },
        }

      case 'elevated':
        return {
          ...baseStyles,
          background: isDark 
            ? alpha(theme.palette.background.paper, 0.8)
            : theme.palette.background.paper,
          borderRadius: designTokens.borderRadius.xl,
          boxShadow: designTokens.shadows.xl,
          border: 'none',
          '&:hover': hover ? {
            boxShadow: designTokens.shadows['2xl'],
          } : {},
        }

      case 'neon':
        return {
          ...baseStyles,
          background: 'transparent',
          border: `2px solid ${designTokens.colors.accent.cyan}`,
          borderRadius: designTokens.borderRadius.lg,
          boxShadow: `0 0 20px ${alpha(designTokens.colors.accent.cyan, 0.3)}`,
          '&::before': {
            content: '""',
            position: 'absolute',
            inset: 0,
            background: `linear-gradient(45deg, transparent, ${alpha(designTokens.colors.accent.cyan, 0.1)}, transparent)`,
            borderRadius: 'inherit',
          },
          '&:hover': hover ? {
            boxShadow: `0 0 40px ${alpha(designTokens.colors.accent.cyan, 0.5)}`,
          } : {},
        }

      default:
        return {
          ...baseStyles,
          background: theme.palette.background.paper,
          border: `1px solid ${theme.palette.divider}`,
          borderRadius: designTokens.borderRadius.lg,
          boxShadow: designTokens.shadows.md,
          '&:hover': hover ? {
            boxShadow: designTokens.shadows.lg,
            borderColor: theme.palette.primary.main,
          } : {},
        }
    }
  }

  const cardContent = (
    <Card
      sx={{
        ...getCardStyles(),
        ...sx,
      }}
      elevation={0}
    >
      {noPadding ? (
        children
      ) : (
        <CardContent sx={{ p: { xs: 2, sm: 3 } }}>
          {children}
        </CardContent>
      )}
    </Card>
  )

  if (!animated) {
    return cardContent
  }

  return (
    <motion.div
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      whileHover={hover ? "hover" : undefined}
      transition={{ delay }}
    >
      {cardContent}
    </motion.div>
  )
}