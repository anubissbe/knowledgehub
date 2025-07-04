import { Grid, Card, CardContent, Typography, Box, IconButton, Tooltip } from '@mui/material'
import {
  Storage as StorageIcon,
  Work as WorkIcon,
  Search as SearchIcon,
  Memory as MemoryIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/services/api'
import StatsCard from '@/components/StatsCard'
import RecentActivity from '@/components/RecentActivity'
import SystemHealth from '@/components/SystemHealth'

function Dashboard() {
  const { data: stats, isLoading, refetch, isRefetching } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: api.getDashboardStats,
    refetchInterval: 30000, // Refresh every 30 seconds
    refetchIntervalInBackground: true, // Continue refreshing even when tab is not active
  })

  if (isLoading) {
    return <Typography>Loading...</Typography>
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" sx={{ flexGrow: 1 }}>
          Dashboard
        </Typography>
        <Tooltip title="Refresh stats">
          <IconButton 
            onClick={() => refetch()} 
            disabled={isRefetching}
            color="primary"
          >
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Box>
      
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