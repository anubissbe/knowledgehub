import React, { useMemo } from 'react'
import { Box, useTheme } from '@mui/material'
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend,
  AreaChart,
  Area,
  BarChart,
  Bar,
  TooltipProps,
} from 'recharts'
import { motion } from 'framer-motion'
import { designTokens } from '../../theme/designSystem'

export interface ChartDataPoint {
  timestamp: string | number
  [key: string]: string | number
}

export interface RealTimeChartProps {
  data: ChartDataPoint[]
  type?: 'line' | 'area' | 'bar'
  dataKeys: string[]
  colors?: string[]
  height?: number
  showGrid?: boolean
  showLegend?: boolean
  showTooltip?: boolean
  animated?: boolean
  gradient?: boolean
  strokeWidth?: number
  fillOpacity?: number
  xAxisKey?: string
  formatTooltip?: (value: any, name: string) => [string, string]
}

const CustomTooltip = ({ 
  active, 
  payload, 
  label, 
  formatTooltip 
}: TooltipProps<any, any> & { formatTooltip?: (value: any, name: string) => [string, string] }) => {
  const theme = useTheme()
  
  if (!active || !payload || !payload.length) return null

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      style={{
        background: theme.palette.mode === 'dark'
          ? 'rgba(33, 33, 33, 0.95)'
          : 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(10px)',
        border: theme.palette.mode === 'dark'
          ? '1px solid rgba(255, 255, 255, 0.1)'
          : '1px solid rgba(0, 0, 0, 0.1)',
        borderRadius: designTokens.borderRadius.lg,
        padding: '12px 16px',
        boxShadow: designTokens.shadows.lg,
      }}
    >
      <Box sx={{ mb: 1, fontSize: '0.875rem', fontWeight: 600, color: 'text.secondary' }}>
        {typeof label === 'string' 
          ? new Date(label).toLocaleTimeString()
          : label
        }
      </Box>
      {payload.map((item: any, index: number) => {
        const [formattedValue, formattedName] = formatTooltip 
          ? formatTooltip(item.value, item.name)
          : [item.value, item.name]
        
        return (
          <Box
            key={index}
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 2,
              fontSize: '0.875rem',
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  backgroundColor: item.color,
                }}
              />
              <span>{formattedName}</span>
            </Box>
            <Box sx={{ fontWeight: 600, color: item.color }}>
              {formattedValue}
            </Box>
          </Box>
        )
      })}
    </motion.div>
  )
}

export default function RealTimeChart({
  data,
  type = 'line',
  dataKeys,
  colors,
  height = 300,
  showGrid = true,
  showLegend = true,
  showTooltip = true,
  animated = true,
  gradient = false,
  strokeWidth = 2,
  fillOpacity = 0.1,
  xAxisKey = 'timestamp',
  formatTooltip,
}: RealTimeChartProps) {
  const theme = useTheme()
  const isDark = theme.palette.mode === 'dark'

  const defaultColors = [
    designTokens.colors.primary[500],
    designTokens.colors.secondary[500],
    designTokens.colors.accent.green,
    designTokens.colors.accent.purple,
    designTokens.colors.accent.orange,
    designTokens.colors.accent.cyan,
  ]

  const chartColors = colors || defaultColors.slice(0, dataKeys.length)

  const gradientDefinitions = useMemo(() => {
    if (!gradient) return null
    
    return (
      <defs>
        {dataKeys.map((key, index) => (
          <linearGradient 
            key={key} 
            id={`gradient-${key}`} 
            x1="0" 
            y1="0" 
            x2="0" 
            y2="1"
          >
            <stop 
              offset="5%" 
              stopColor={chartColors[index]} 
              stopOpacity={0.8} 
            />
            <stop 
              offset="95%" 
              stopColor={chartColors[index]} 
              stopOpacity={0.1} 
            />
          </linearGradient>
        ))}
      </defs>
    )
  }, [dataKeys, chartColors, gradient])

  const commonProps = {
    data,
    margin: { top: 5, right: 30, left: 20, bottom: 5 },
  }

  const xAxisProps = {
    dataKey: xAxisKey,
    axisLine: false,
    tickLine: false,
    tick: { 
      fontSize: 12, 
      fill: theme.palette.text.secondary 
    },
    tickFormatter: (value: any) => {
      if (typeof value === 'string' || typeof value === 'number') {
        const date = new Date(value)
        return isNaN(date.getTime()) ? value : date.toLocaleTimeString()
      }
      return value
    },
  }

  const yAxisProps = {
    axisLine: false,
    tickLine: false,
    tick: { 
      fontSize: 12, 
      fill: theme.palette.text.secondary 
    },
  }

  const gridProps = showGrid ? {
    strokeDasharray: "3 3",
    stroke: isDark 
      ? 'rgba(255, 255, 255, 0.1)' 
      : 'rgba(0, 0, 0, 0.1)',
  } : false

  const tooltipProps = showTooltip ? {
    content: <CustomTooltip formatTooltip={formatTooltip} />,
  } : false

  const legendProps = showLegend ? {
    wrapperStyle: {
      paddingTop: '20px',
      fontSize: '12px',
    },
  } : false

  const renderChart = () => {
    const animationProps = animated ? {
      isAnimationActive: true,
      animationDuration: 1000,
      animationBegin: 0,
    } : {
      isAnimationActive: false,
    }

    switch (type) {
      case 'area':
        return (
          <AreaChart {...commonProps}>
            {gradientDefinitions}
            <CartesianGrid {...gridProps} />
            <XAxis {...xAxisProps} />
            <YAxis {...yAxisProps} />
            {tooltipProps && <Tooltip {...tooltipProps} />}
            {legendProps && <Legend {...legendProps} />}
            {dataKeys.map((key, index) => (
              <Area
                key={key}
                type="monotone"
                dataKey={key}
                stroke={chartColors[index]}
                strokeWidth={strokeWidth}
                fill={gradient ? `url(#gradient-${key})` : chartColors[index]}
                fillOpacity={fillOpacity}
                {...animationProps}
              />
            ))}
          </AreaChart>
        )

      case 'bar':
        return (
          <BarChart {...commonProps}>
            <CartesianGrid {...gridProps} />
            <XAxis {...xAxisProps} />
            <YAxis {...yAxisProps} />
            {tooltipProps && <Tooltip {...tooltipProps} />}
            {legendProps && <Legend {...legendProps} />}
            {dataKeys.map((key, index) => (
              <Bar
                key={key}
                dataKey={key}
                fill={chartColors[index]}
                radius={[4, 4, 0, 0]}
                {...animationProps}
              />
            ))}
          </BarChart>
        )

      default: // line
        return (
          <LineChart {...commonProps}>
            <CartesianGrid {...gridProps} />
            <XAxis {...xAxisProps} />
            <YAxis {...yAxisProps} />
            {tooltipProps && <Tooltip {...tooltipProps} />}
            {legendProps && <Legend {...legendProps} />}
            {dataKeys.map((key, index) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={chartColors[index]}
                strokeWidth={strokeWidth}
                dot={{
                  fill: chartColors[index],
                  strokeWidth: 0,
                  r: 4,
                }}
                activeDot={{
                  r: 6,
                  fill: chartColors[index],
                  strokeWidth: 0,
                  style: { filter: `drop-shadow(0 0 6px ${chartColors[index]})` },
                }}
                {...animationProps}
              />
            ))}
          </LineChart>
        )
    }
  }

  return (
    <Box sx={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        {renderChart()}
      </ResponsiveContainer>
    </Box>
  )
}