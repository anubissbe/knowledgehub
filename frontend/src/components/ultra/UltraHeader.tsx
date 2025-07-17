import { Box, Typography, Chip, alpha } from '@mui/material'
import { motion } from 'framer-motion'
import { AutoAwesome } from '@mui/icons-material'

interface UltraHeaderProps {
  title: string
  subtitle?: string
  gradient?: boolean
}

export default function UltraHeader({ title, subtitle, gradient = true }: UltraHeaderProps) {
  return (
    <Box sx={{ textAlign: 'center', py: 6 }}>
      <motion.div
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, type: 'spring', stiffness: 100 }}
      >
        <Typography
          variant="h1"
          sx={{
            fontSize: { xs: '3rem', md: '5rem' },
            fontWeight: 900,
            background: gradient
              ? theme => `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main} 100%)`
              : 'none',
            backgroundClip: gradient ? 'text' : 'unset',
            WebkitBackgroundClip: gradient ? 'text' : 'unset',
            WebkitTextFillColor: gradient ? 'transparent' : 'inherit',
            mb: 2,
            textShadow: theme => `0 0 80px ${alpha(theme.palette.primary.main, 0.5)}`,
          }}
        >
          {title}
        </Typography>
        
        {subtitle && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.5 }}
          >
            <Typography
              variant="h5"
              color="text.secondary"
              sx={{ mb: 2, letterSpacing: 2 }}
            >
              {subtitle}
            </Typography>
          </motion.div>
        )}
        
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.5, type: 'spring', stiffness: 200 }}
        >
          <Chip
            icon={<AutoAwesome />}
            label="AI ENHANCED"
            color="primary"
            size="medium"
            sx={{
              fontWeight: 600,
              fontSize: '0.875rem',
              animation: 'pulse 2s infinite',
              '@keyframes pulse': {
                '0%': { transform: 'scale(1)', opacity: 1 },
                '50%': { transform: 'scale(1.05)', opacity: 0.8 },
                '100%': { transform: 'scale(1)', opacity: 1 },
              },
            }}
          />
        </motion.div>
      </motion.div>
    </Box>
  )
}