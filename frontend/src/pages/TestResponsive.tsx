import { Box, Typography, Paper } from '@mui/material'

export default function TestResponsive() {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Responsive Test Page
      </Typography>
      
      <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
        <Paper sx={{ p: 2, flex: 1 }}>
          <Typography>Box 1 - Should expand</Typography>
        </Paper>
        <Paper sx={{ p: 2, flex: 1 }}>
          <Typography>Box 2 - Should expand</Typography>
        </Paper>
        <Paper sx={{ p: 2, flex: 1 }}>
          <Typography>Box 3 - Should expand</Typography>
        </Paper>
      </Box>
      
      <Paper sx={{ p: 2, width: '100%', bgcolor: 'primary.light' }}>
        <Typography>Full width test - this should span the entire content area</Typography>
      </Paper>
    </Box>
  )
}