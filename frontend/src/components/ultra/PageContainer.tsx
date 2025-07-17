import { Box } from '@mui/material'
import { motion } from 'framer-motion'
import ParticlesBackground from '../ParticlesBackground'

interface PageContainerProps {
  children: React.ReactNode
}

export default function PageContainer({ children }: PageContainerProps) {
  return (
    <Box sx={{ position: 'relative', minHeight: '100vh', overflow: 'hidden' }}>
      <ParticlesBackground />
      
      {/* Animated gradient background */}
      <Box
        sx={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: theme => `
            radial-gradient(circle at 20% 50%, ${theme.palette.primary.main}20 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, ${theme.palette.secondary.main}20 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, ${theme.palette.info.main}20 0%, transparent 50%)
          `,
          animation: 'gradient 15s ease infinite',
          '@keyframes gradient': {
            '0%, 100%': { transform: 'scale(1) rotate(0deg)' },
            '50%': { transform: 'scale(1.1) rotate(180deg)' },
          },
          zIndex: -1,
        }}
      />
      
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        {children}
      </motion.div>
    </Box>
  )
}