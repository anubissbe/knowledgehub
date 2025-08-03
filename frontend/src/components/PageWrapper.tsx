import { Box } from '@mui/material'

interface PageWrapperProps {
  children: React.ReactNode
}

export default function PageWrapper({ children }: PageWrapperProps) {
  return (
    <Box sx={{ width: '100%', height: '100%' }}>
      {children}
    </Box>
  )
}