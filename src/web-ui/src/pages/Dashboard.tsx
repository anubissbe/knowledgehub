import { Grid, Card, CardContent, Typography, Box } from '@mui/material'
import {
  Storage as StorageIcon,
  Work as WorkIcon,
  Search as SearchIcon,
  Memory as MemoryIcon,
} from '@mui/icons-material'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/services/api'
import StatsCard from '@/components/StatsCard'
import RecentActivity from '@/components/RecentActivity'
import SystemHealth from '@/components/SystemHealth'

function Dashboard() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: api.getDashboardStats,
  })

  if (isLoading) {
    return <Typography>Loading...</Typography>
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        {/* Stats Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <StatsCard
            title="Sources"
            value={stats?.total_sources || 0}
            icon={<StorageIcon />}
            color="primary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatsCard
            title="Running Jobs"
            value={stats?.running_jobs || 0}
            icon={<WorkIcon />}
            color="warning"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatsCard
            title="Total Chunks"
            value={stats?.total_chunks || 0}
            icon={<SearchIcon />}
            color="success"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatsCard
            title="Documents"
            value={stats?.total_documents || 0}
            icon={<MemoryIcon />}
            color="info"
          />
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Activity
              </Typography>
              <RecentActivity />
            </CardContent>
          </Card>
        </Grid>

        {/* System Health */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Health
              </Typography>
              <SystemHealth />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

export default Dashboard