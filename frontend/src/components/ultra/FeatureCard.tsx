import { Box, Typography, IconButton, LinearProgress, alpha } from '@mui/material'
import { ArrowForward } from '@mui/icons-material'
import { motion } from 'framer-motion'
import GlassCard from '../GlassCard'

interface FeatureCardProps {
  icon: React.ReactNode
  title: string
  description: string
  progress?: number
  color: string
  onClick?: () => void
  delay?: number
  stats?: {
    label: string
    value: string | number
  }[]
}

export default function FeatureCard({
  icon,
  title,
  description,
  progress,
  color,
  onClick,
  delay = 0,
  stats = [],
}: FeatureCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{
        delay,
        type: 'spring',
        stiffness: 100,
        damping: 20,
      }}
      whileHover={{ scale: 1.02, transition: { duration: 0.2 } }}
    >
      <GlassCard
        gradient
        hover
        sx={{
          height: '100%',
          cursor: onClick ? 'pointer' : 'default',
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: -2,
            left: -2,
            right: -2,
            bottom: -2,
            background: `linear-gradient(45deg, ${color}, transparent)`,
            borderRadius: 'inherit',
            opacity: 0,
            transition: 'opacity 0.3s',
            zIndex: -1,
          },
          '&:hover::before': {
            opacity: 0.1,
          },
        }}
        onClick={onClick}
      >
        <Box sx={{ p: 3 }}>
          {/* Header */}
          <Box display="flex" alignItems="flex-start" mb={3}>
            <Box
              sx={{
                p: 2,
                borderRadius: 2,
                background: alpha(color, 0.1),
                color: color,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: `0 0 30px ${alpha(color, 0.3)}`,
              }}
            >
              {icon}
            </Box>
            
            <Box flex={1} ml={2}>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                {title}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {description}
              </Typography>
            </Box>
            
            {onClick && (
              <IconButton
                size="small"
                sx={{
                  color: color,
                  backgroundColor: alpha(color, 0.1),
                  '&:hover': {
                    backgroundColor: alpha(color, 0.2),
                  },
                }}
                onClick={(e) => {
                  e.stopPropagation()
                  onClick()
                }}
              >
                <ArrowForward />
              </IconButton>
            )}
          </Box>

          {/* Progress */}
          {progress !== undefined && (
            <Box mb={2}>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="caption" color="text.secondary">
                  Progress
                </Typography>
                <Typography variant="caption" fontWeight="bold">
                  {progress}%
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={progress}
                sx={{
                  height: 6,
                  borderRadius: 3,
                  backgroundColor: alpha(color, 0.1),
                  '& .MuiLinearProgress-bar': {
                    borderRadius: 3,
                    background: `linear-gradient(90deg, ${color}, ${alpha(color, 0.6)})`,
                  },
                }}
              />
            </Box>
          )}

          {/* Stats */}
          {stats.length > 0 && (
            <Box display="flex" gap={2} mt={2}>
              {stats.map((stat, index) => (
                <Box
                  key={index}
                  flex={1}
                  sx={{
                    p: 1.5,
                    borderRadius: 2,
                    backgroundColor: theme => alpha(theme.palette.background.default, 0.5),
                    textAlign: 'center',
                  }}
                >
                  <Typography variant="h6" fontWeight="bold" color={color}>
                    {stat.value}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {stat.label}
                  </Typography>
                </Box>
              ))}
            </Box>
          )}

          {/* Decorative element */}
          <Box
            sx={{
              position: 'absolute',
              bottom: -20,
              right: -20,
              width: 80,
              height: 80,
              borderRadius: '50%',
              background: `radial-gradient(circle, ${alpha(color, 0.2)}, transparent)`,
              filter: 'blur(20px)',
            }}
          />
        </Box>
      </GlassCard>
    </motion.div>
  )
}