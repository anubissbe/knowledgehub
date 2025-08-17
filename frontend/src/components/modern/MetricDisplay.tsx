import React, { ReactNode, useEffect, useState } from 'react'
import { Box, Typography, useTheme, alpha } from '@mui/material'
import { motion, AnimatePresence } from 'framer-motion'
import { TrendingUp, TrendingDown, TrendingFlat } from '@mui/icons-material'
import { designTokens } from '../../theme/designSystem'
import ModernCard from './ModernCard'

export interface MetricDisplayProps {
  label: string
  value: number | string
  icon?: ReactNode
  trend?: number
  unit?: string
  format?: 'number' | 'currency' | 'percentage' | 'duration'
  color?: string
  variant?: 'default' | 'compact' | 'large'
  animated?: boolean
  realTime?: boolean
  sparkline?: number[]
  target?: number
}

const formatValue = (value: number | string, format: string, unit?: string) => {
  if (typeof value === 'string') return value

  const formatted = (() => {
    switch (format) {
      case 'currency':
        return new Intl.NumberFormat('en-US', { 
          style: 'currency', 
          currency: 'USD' 
        }).format(value)
      case 'percentage':
        return `${value.toFixed(1)}%`
      case 'duration':
        return `${value}ms`
      default:
        return new Intl.NumberFormat().format(value)
    }
  })()

  return unit ? `${formatted} ${unit}` : formatted
}

const getTrendColor = (trend: number, theme: any) => {
  if (trend > 0) return designTokens.colors.semantic.success
  if (trend < 0) return designTokens.colors.semantic.error
  return theme.palette.text.secondary
}

const getTrendIcon = (trend: number) => {
  if (trend > 0) return <TrendingUp />
  if (trend < 0) return <TrendingDown />
  return <TrendingFlat />
}

const Sparkline = ({ data, color }: { data: number[], color: string }) => {
  if (!data || data.length === 0) return null

  const max = Math.max(...data)
  const min = Math.min(...data)
  const range = max - min
  
  const points = data.map((value, index) => {
    const x = (index / (data.length - 1)) * 100
    const y = range === 0 ? 50 : 100 - ((value - min) / range) * 100
    return `${x},${y}`
  }).join(' ')

  return (
    <Box sx={{ width: 60, height: 20, opacity: 0.7 }}>
      <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
        <motion.polyline
          fill="none"
          stroke={color}
          strokeWidth="2"
          points={points}
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1, ease: "easeInOut" }}
        />
      </svg>
    </Box>
  )
}

const AnimatedNumber = ({ 
  value, 
  format, 
  unit 
}: { 
  value: number | string, 
  format: string, 
  unit?: string 
}) => {
  const [displayValue, setDisplayValue] = useState<number | string>(0)

  useEffect(() => {
    if (typeof value === 'string') {
      setDisplayValue(value)
      return
    }

    let start = 0
    const end = value
    const duration = 1000
    const startTime = Date.now()

    const animate = () => {
      const elapsed = Date.now() - startTime
      const progress = Math.min(elapsed / duration, 1)
      
      const easedProgress = 1 - Math.pow(1 - progress, 3) // easeOutCubic
      const current = start + (end - start) * easedProgress
      
      setDisplayValue(current)

      if (progress < 1) {
        requestAnimationFrame(animate)
      } else {
        setDisplayValue(end)
      }
    }

    animate()
  }, [value])

  return <>{formatValue(displayValue, format, unit)}</>
}

export default function MetricDisplay({
  label,
  value,
  icon,
  trend,
  unit,
  format = 'number',
  color,
  variant = 'default',
  animated = true,
  realTime = false,
  sparkline,
  target,
}: MetricDisplayProps) {
  const theme = useTheme()
  const trendColor = trend !== undefined ? getTrendColor(trend, theme) : undefined
  const mainColor = color || theme.palette.primary.main

  const [isRealTimeUpdating, setIsRealTimeUpdating] = useState(false)

  useEffect(() => {
    if (realTime) {
      setIsRealTimeUpdating(true)
      const timer = setTimeout(() => setIsRealTimeUpdating(false), 300)
      return () => clearTimeout(timer)
    }
  }, [value, realTime])

  const getVariantStyles = () => {
    switch (variant) {
      case 'compact':
        return { p: 2, minHeight: 'auto' }
      case 'large':
        return { p: { xs: 3, sm: 4 }, minHeight: 200 }
      default:
        return { p: 3, minHeight: 140 }
    }
  }

  return (
    <ModernCard
      variant="glass"
      animated={animated}
      sx={{
        height: '100%',
        position: 'relative',
        background: isRealTimeUpdating 
          ? `linear-gradient(45deg, ${alpha(mainColor, 0.1)}, transparent)`
          : undefined,
        transition: designTokens.transitions.normal,
        ...getVariantStyles(),
      }}
    >
      {/* Pulse indicator for real-time updates */}
      <AnimatePresence>
        {isRealTimeUpdating && (
          <motion.div
            initial={{ scale: 0, opacity: 1 }}
            animate={{ scale: 2, opacity: 0 }}
            exit={{ opacity: 0 }}
            style={{
              position: 'absolute',
              top: 12,
              right: 12,
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: designTokens.colors.semantic.success,
            }}
          />
        )}
      </AnimatePresence>

      <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <Box 
          sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between',
            mb: variant === 'compact' ? 1 : 2 
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {icon && (
              <Box sx={{ 
                color: mainColor, 
                display: 'flex',
                fontSize: variant === 'large' ? 28 : 20 
              }}>
                {icon}
              </Box>
            )}
            <Typography 
              variant={variant === 'large' ? 'h6' : 'body2'} 
              color="text.secondary"
              sx={{ fontWeight: 500 }}
            >
              {label}
            </Typography>
          </Box>
          
          {sparkline && (
            <Sparkline data={sparkline} color={mainColor} />
          )}
        </Box>

        {/* Value */}
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <Typography
            variant={variant === 'large' ? 'h3' : variant === 'compact' ? 'h6' : 'h5'}
            sx={{
              fontWeight: 700,
              color: mainColor,
              lineHeight: 1,
              mb: trend !== undefined ? 1 : 0,
            }}
          >
            {animated ? (
              <AnimatedNumber value={value} format={format} unit={unit} />
            ) : (
              formatValue(value, format, unit)
            )}
          </Typography>

          {/* Trend */}
          {trend !== undefined && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Box sx={{ 
                  color: trendColor, 
                  display: 'flex', 
                  alignItems: 'center',
                  fontSize: 16 
                }}>
                  {getTrendIcon(trend)}
                </Box>
                <Typography 
                  variant="caption" 
                  sx={{ 
                    color: trendColor,
                    fontWeight: 600,
                    fontSize: variant === 'large' ? '0.875rem' : '0.75rem'
                  }}
                >
                  {trend > 0 ? '+' : ''}{trend.toFixed(1)}%
                </Typography>
              </Box>
            </motion.div>
          )}

          {/* Target indicator */}
          {target !== undefined && typeof value === 'number' && (
            <Box sx={{ mt: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography variant="caption" color="text.secondary">
                  Target: {formatValue(target, format, unit)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {((value / target) * 100).toFixed(0)}%
                </Typography>
              </Box>
              <Box
                sx={{
                  width: '100%',
                  height: 4,
                  backgroundColor: alpha(theme.palette.text.secondary, 0.2),
                  borderRadius: 2,
                  overflow: 'hidden',
                }}
              >
                <motion.div
                  style={{
                    height: '100%',
                    background: `linear-gradient(90deg, ${mainColor}, ${alpha(mainColor, 0.7)})`,
                    borderRadius: 2,
                  }}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min((value / target) * 100, 100)}%` }}
                  transition={{ duration: 1, ease: "easeOut" }}
                />
              </Box>
            </Box>
          )}
        </Box>
      </Box>
    </ModernCard>
  )
}