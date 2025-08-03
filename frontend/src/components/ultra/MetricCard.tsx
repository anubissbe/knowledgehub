import { Box, Typography, Avatar, IconButton, alpha, useTheme } from '@mui/material'
import { TrendingUp, TrendingDown, MoreVert } from '@mui/icons-material'
import { motion } from 'framer-motion'
import GlassCard from '../GlassCard'
import { ResponsiveContainer, LineChart, Line } from 'recharts'

interface MetricCardProps {
  icon: React.ReactNode
  label: string
  value: string | number
  trend?: number
  unit?: string
  color: string
  sparkline?: number[]
  delay?: number
}

export default function MetricCard({
  icon,
  label,
  value,
  trend = 0,
  unit = '',
  color,
  sparkline = [],
  delay = 0,
}: MetricCardProps) {
  const theme = useTheme()
  
  const sparklineData = sparkline.map((v, i) => ({ value: v, index: i }))

  return (
    <motion.div
      initial={{ y: 50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{
        delay,
        type: 'spring',
        stiffness: 100,
        damping: 15,
      }}
      whileHover={{ y: -5, transition: { duration: 0.2 } }}
    >
      <GlassCard gradient hover sx={{ height: '100%' }}>
        <Box sx={{ 
          p: { xs: 2, sm: 3 }, 
          height: '100%', 
          display: 'flex', 
          flexDirection: 'column' 
        }}>
          <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
            <Avatar
              sx={{
                bgcolor: alpha(color, 0.1),
                color: color,
                width: 48,
                height: 48,
                boxShadow: `0 0 20px ${alpha(color, 0.4)}`,
              }}
            >
              {icon}
            </Avatar>
            <IconButton size="small">
              <MoreVert fontSize="small" />
            </IconButton>
          </Box>
          
          <Typography variant="caption" color="text.secondary">
            {label}
          </Typography>
          
          <Box display="flex" alignItems="baseline" gap={1}>
            <Typography variant="h4" fontWeight="bold">
              {typeof value === 'number' ? value.toLocaleString() : value}
            </Typography>
            {unit && (
              <Typography variant="body2" color="text.secondary">
                {unit}
              </Typography>
            )}
          </Box>
          
          {trend !== 0 && (
            <Box display="flex" alignItems="center" gap={0.5} mt={1}>
              {trend > 0 ? (
                <TrendingUp sx={{ fontSize: 16, color: theme.palette.success.main }} />
              ) : (
                <TrendingDown sx={{ fontSize: 16, color: theme.palette.error.main }} />
              )}
              <Typography
                variant="caption"
                color={trend > 0 ? 'success.main' : 'error.main'}
              >
                {Math.abs(trend)}%
              </Typography>
            </Box>
          )}
          
          <Box sx={{ flexGrow: 1 }} />
          
          {/* Always render sparkline container for consistent height */}
          <Box sx={{ height: 40, mt: 2 }}>
            {sparkline.length > 0 && (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sparklineData}>
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke={color}
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </Box>
        </Box>
      </GlassCard>
    </motion.div>
  )
}