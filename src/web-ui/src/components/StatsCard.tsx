import { Card, CardContent, Typography, Box } from '@mui/material'
import { ReactNode } from 'react'

interface StatsCardProps {
  title: string
  value: number | string
  icon: ReactNode
  color?: 'primary' | 'secondary' | 'success' | 'error' | 'warning' | 'info'
}

function StatsCard({ title, value, icon, color = 'primary' }: StatsCardProps) {
  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Box
            sx={{
              p: 1.5,
              borderRadius: 2,
              bgcolor: `${color}.light`,
              color: `${color}.main`,
              mr: 2,
            }}
          >
            {icon}
          </Box>
          <Typography color="text.secondary" variant="subtitle2">
            {title}
          </Typography>
        </Box>
        <Typography variant="h4" component="div">
          {value}
        </Typography>
      </CardContent>
    </Card>
  )
}

export default StatsCard