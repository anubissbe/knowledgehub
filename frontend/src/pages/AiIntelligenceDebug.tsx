import { Box, Typography, Grid, Card, CardContent } from '@mui/material'
import PageContainer from '../components/ultra/PageContainer'
import UltraHeader from '../components/ultra/UltraHeader'

export default function AiIntelligenceDebug() {
  const features = [
    { id: 1, title: 'Session Continuity', color: '#2196F3' },
    { id: 2, title: 'Mistake Learning', color: '#FF00FF' },
    { id: 3, title: 'Proactive Assistant', color: '#00FF88' },
    { id: 4, title: 'Decision Reasoning', color: '#FFD700' },
    { id: 5, title: 'Code Evolution', color: '#00FFFF' },
    { id: 6, title: 'Performance Optimization', color: '#8B5CF6' },
    { id: 7, title: 'Pattern Recognition', color: '#FF3366' },
    { id: 8, title: 'Workflow Integration', color: '#EC4899' },
  ]

  return (
    <PageContainer>
      <UltraHeader 
        title="AI Intelligence Debug" 
        subtitle="TESTING COMPONENT RENDERING"
      />

      <Box sx={{ px: 3 }}>
        <Typography variant="h5" sx={{ mb: 3 }}>
          Debug: Testing if components render
        </Typography>

        <Grid container spacing={3}>
          {features.map((feature) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={feature.id}>
              <Card sx={{ 
                backgroundColor: feature.color + '20',
                border: `2px solid ${feature.color}`
              }}>
                <CardContent>
                  <Typography variant="h6">{feature.title}</Typography>
                  <Typography variant="body2">Feature #{feature.id}</Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    </PageContainer>
  )
}