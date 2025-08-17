import React, { ReactNode } from 'react'
import { Button, ButtonProps, useTheme, alpha } from '@mui/material'
import { motion } from 'framer-motion'
import { designTokens } from '../../theme/designSystem'

export interface ModernButtonProps extends Omit<ButtonProps, 'variant'> {
  variant?: 'primary' | 'secondary' | 'glass' | 'neon' | 'gradient' | 'outline'
  icon?: ReactNode
  loading?: boolean
  glowing?: boolean
}

const buttonVariants = {
  hover: { scale: 1.02 },
  tap: { scale: 0.98 }
}

export default function ModernButton({
  children,
  variant = 'primary',
  icon,
  loading = false,
  glowing = false,
  sx = {},
  ...props
}: ModernButtonProps) {
  const theme = useTheme()
  const isDark = theme.palette.mode === 'dark'

  const getButtonStyles = () => {
    const baseStyles = {
      borderRadius: designTokens.borderRadius.lg,
      textTransform: 'none' as const,
      fontWeight: designTokens.typography.fontWeight.semibold,
      padding: '12px 24px',
      minHeight: 48,
      position: 'relative',
      overflow: 'hidden',
      transition: designTokens.transitions.normal,
      '&::before': glowing ? {
        content: '""',
        position: 'absolute',
        inset: 0,
        background: 'inherit',
        filter: 'blur(10px)',
        opacity: 0.7,
        zIndex: -1,
      } : {},
    }

    switch (variant) {
      case 'glass':
        return {
          ...baseStyles,
          background: isDark 
            ? 'rgba(255, 255, 255, 0.1)'
            : 'rgba(255, 255, 255, 0.2)',
          backdropFilter: 'blur(10px)',
          border: isDark
            ? '1px solid rgba(255, 255, 255, 0.2)'
            : '1px solid rgba(255, 255, 255, 0.3)',
          color: isDark ? theme.palette.text.primary : theme.palette.text.secondary,
          '&:hover': {
            background: isDark 
              ? 'rgba(255, 255, 255, 0.15)'
              : 'rgba(255, 255, 255, 0.3)',
            transform: 'translateY(-2px)',
          },
        }

      case 'neon':
        return {
          ...baseStyles,
          background: 'transparent',
          border: `2px solid ${designTokens.colors.accent.cyan}`,
          color: designTokens.colors.accent.cyan,
          boxShadow: glowing 
            ? `0 0 20px ${alpha(designTokens.colors.accent.cyan, 0.5)}`
            : `0 0 10px ${alpha(designTokens.colors.accent.cyan, 0.3)}`,
          '&:hover': {
            background: alpha(designTokens.colors.accent.cyan, 0.1),
            boxShadow: `0 0 30px ${alpha(designTokens.colors.accent.cyan, 0.7)}`,
            transform: 'translateY(-2px)',
          },
        }

      case 'gradient':
        return {
          ...baseStyles,
          background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
          color: theme.palette.primary.contrastText,
          boxShadow: designTokens.shadows.md,
          '&:hover': {
            background: `linear-gradient(45deg, ${theme.palette.primary.dark}, ${theme.palette.secondary.dark})`,
            boxShadow: designTokens.shadows.lg,
            transform: 'translateY(-2px)',
          },
        }

      case 'outline':
        return {
          ...baseStyles,
          background: 'transparent',
          border: `2px solid ${theme.palette.primary.main}`,
          color: theme.palette.primary.main,
          '&:hover': {
            background: alpha(theme.palette.primary.main, 0.1),
            transform: 'translateY(-2px)',
          },
        }

      case 'secondary':
        return {
          ...baseStyles,
          background: theme.palette.secondary.main,
          color: theme.palette.secondary.contrastText,
          boxShadow: designTokens.shadows.md,
          '&:hover': {
            background: theme.palette.secondary.dark,
            boxShadow: designTokens.shadows.lg,
            transform: 'translateY(-2px)',
          },
        }

      default: // primary
        return {
          ...baseStyles,
          background: theme.palette.primary.main,
          color: theme.palette.primary.contrastText,
          boxShadow: designTokens.shadows.md,
          '&:hover': {
            background: theme.palette.primary.dark,
            boxShadow: designTokens.shadows.lg,
            transform: 'translateY(-2px)',
          },
        }
    }
  }

  return (
    <motion.div
      variants={buttonVariants}
      whileHover="hover"
      whileTap="tap"
      style={{ display: 'inline-block' }}
    >
      <Button
        {...props}
        sx={{
          ...getButtonStyles(),
          ...sx,
        }}
        disabled={loading || props.disabled}
        startIcon={loading ? (
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            style={{ width: 20, height: 20, border: '2px solid currentColor', borderTop: '2px solid transparent', borderRadius: '50%' }}
          />
        ) : icon}
      >
        {children}
      </Button>
    </motion.div>
  )
}