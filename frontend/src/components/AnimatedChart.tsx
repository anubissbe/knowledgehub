import { useEffect, useState } from 'react'
import { Box, useTheme, alpha } from '@mui/material'
import { motion } from 'framer-motion'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  LineChart,
  Line,
  BarChart,
  Bar,
} from 'recharts'

interface AnimatedChartProps {
  data: any[]
  type: 'area' | 'radar' | 'line' | 'bar'
  dataKeys: string[]
  colors?: string[]
  height?: number
  animated?: boolean
}

const defaultColors = [
  '#2196F3', // Blue
  '#FF00FF', // Magenta
  '#00FF88', // Green
  '#FFD700', // Gold
  '#00FFFF', // Cyan
  '#8B5CF6', // Violet
]

export default function AnimatedChart({
  data,
  type,
  dataKeys,
  colors = defaultColors,
  height = 300,
  animated = true,
}: AnimatedChartProps) {
  const theme = useTheme()
  const [animatedData, setAnimatedData] = useState(animated ? [] : data)

  useEffect(() => {
    if (animated && data.length > 0) {
      // Animate data points appearing
      const timer = setTimeout(() => {
        setAnimatedData(data)
      }, 100)
      return () => clearTimeout(timer)
    } else {
      setAnimatedData(data)
    }
  }, [data, animated])

  const chartVariants = {
    hidden: { opacity: 0, scale: 0.9 },
    visible: {
      opacity: 1,
      scale: 1,
      transition: {
        duration: 0.5,
        ease: 'easeOut' as const,
      },
    },
  }

  const tooltipStyle = {
    backgroundColor: alpha(theme.palette.background.paper, 0.95),
    backdropFilter: 'blur(10px)',
    border: `1px solid ${alpha(theme.palette.divider, 0.2)}`,
    borderRadius: 8,
    padding: '8px 12px',
  }

  const renderChart = () => {
    switch (type) {
      case 'area':
        return (
          <AreaChart data={animatedData}>
            <defs>
              {dataKeys.map((key, index) => (
                <linearGradient key={key} id={`color${key}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={colors[index % colors.length]} stopOpacity={0.8} />
                  <stop offset="95%" stopColor={colors[index % colors.length]} stopOpacity={0.1} />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={alpha(theme.palette.divider, 0.3)} />
            <XAxis stroke={theme.palette.text.secondary} />
            <YAxis stroke={theme.palette.text.secondary} />
            <Tooltip contentStyle={tooltipStyle} />
            {dataKeys.map((key, index) => (
              <Area
                key={key}
                type="monotone"
                dataKey={key}
                stroke={colors[index % colors.length]}
                fillOpacity={1}
                fill={`url(#color${key})`}
                strokeWidth={2}
                animationDuration={animated ? 1500 : 0}
              />
            ))}
          </AreaChart>
        )

      case 'radar':
        return (
          <RadarChart data={animatedData}>
            <PolarGrid stroke={alpha(theme.palette.divider, 0.3)} />
            <PolarAngleAxis dataKey="name" stroke={theme.palette.text.secondary} />
            <PolarRadiusAxis stroke={theme.palette.text.secondary} />
            <Tooltip contentStyle={tooltipStyle} />
            {dataKeys.map((key, index) => (
              <Radar
                key={key}
                name={key}
                dataKey={key}
                stroke={colors[index % colors.length]}
                fill={colors[index % colors.length]}
                fillOpacity={0.3}
                animationDuration={animated ? 1500 : 0}
              />
            ))}
          </RadarChart>
        )

      case 'line':
        return (
          <LineChart data={animatedData}>
            <CartesianGrid strokeDasharray="3 3" stroke={alpha(theme.palette.divider, 0.3)} />
            <XAxis stroke={theme.palette.text.secondary} />
            <YAxis stroke={theme.palette.text.secondary} />
            <Tooltip contentStyle={tooltipStyle} />
            {dataKeys.map((key, index) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={colors[index % colors.length]}
                strokeWidth={3}
                dot={{ fill: colors[index % colors.length], strokeWidth: 2, r: 4 }}
                activeDot={{ r: 6 }}
                animationDuration={animated ? 1500 : 0}
              />
            ))}
          </LineChart>
        )

      case 'bar':
        return (
          <BarChart data={animatedData}>
            <CartesianGrid strokeDasharray="3 3" stroke={alpha(theme.palette.divider, 0.3)} />
            <XAxis stroke={theme.palette.text.secondary} />
            <YAxis stroke={theme.palette.text.secondary} />
            <Tooltip contentStyle={tooltipStyle} />
            {dataKeys.map((key, index) => (
              <Bar
                key={key}
                dataKey={key}
                fill={colors[index % colors.length]}
                animationDuration={animated ? 1500 : 0}
                radius={[8, 8, 0, 0]}
              />
            ))}
          </BarChart>
        )

      default:
        return <div>No chart type selected</div>
    }
  }

  return (
    <motion.div
      variants={chartVariants}
      initial="hidden"
      animate="visible"
      style={{ width: '100%', height }}
    >
      <Box
        sx={{
          position: 'relative',
          width: '100%',
          height: '100%',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `linear-gradient(45deg, 
              ${alpha(theme.palette.primary.main, 0.05)} 0%, 
              transparent 50%, 
              ${alpha(theme.palette.secondary.main, 0.05)} 100%)`,
            pointerEvents: 'none',
            zIndex: 0,
          },
        }}
      >
        <ResponsiveContainer width="100%" height="100%">
          {renderChart()}
        </ResponsiveContainer>
      </Box>
    </motion.div>
  )
}