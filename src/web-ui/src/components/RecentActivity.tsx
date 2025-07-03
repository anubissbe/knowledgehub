import {
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Typography,
  Chip,
  Box,
} from '@mui/material'
import {
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Schedule as ScheduleIcon,
  Storage as StorageIcon,
} from '@mui/icons-material'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/services/api'

// Activity interface removed as it's not used

function RecentActivity() {
  const { data: activities } = useQuery({
    queryKey: ['recent-activity'],
    queryFn: async () => {
      // Get recent jobs
      const response = await api.getJobs({ limit: 10 })
      const jobs = Array.isArray(response?.jobs) ? response.jobs : []
      
      return jobs.map(job => ({
        id: job.id,
        type: job.type,
        title: `${job.type} job`,
        description: `Source: ${job.source_id.slice(0, 8)}...`,
        status: job.status,
        timestamp: job.created_at,
      }))
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const getIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckIcon />
      case 'failed':
        return <ErrorIcon />
      case 'running':
        return <ScheduleIcon />
      default:
        return <StorageIcon />
    }
  }

  const getColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success'
      case 'failed':
        return 'error'
      case 'running':
        return 'warning'
      default:
        return 'default'
    }
  }

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    
    if (diff < 60000) return 'Just now'
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
    return date.toLocaleDateString()
  }

  return (
    <List>
      {activities?.map((activity) => (
        <ListItem key={activity.id} alignItems="flex-start">
          <ListItemAvatar>
            <Avatar sx={{ bgcolor: `${getColor(activity.status)}.light` }}>
              {getIcon(activity.status)}
            </Avatar>
          </ListItemAvatar>
          <ListItemText
            primary={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="subtitle2">{activity.title}</Typography>
                <Chip
                  label={activity.status}
                  size="small"
                  color={getColor(activity.status)}
                />
              </Box>
            }
            secondary={
              <Box component="span">
                <Typography component="span" variant="body2" color="text.secondary">
                  {activity.description}
                </Typography>
                <Typography component="span" variant="caption" color="text.secondary" display="block">
                  {formatTime(activity.timestamp)}
                </Typography>
              </Box>
            }
          />
        </ListItem>
      ))}
      
      {(!activities || activities.length === 0) && (
        <ListItem>
          <ListItemText
            primary="No recent activity"
            secondary="Jobs and activities will appear here"
          />
        </ListItem>
      )}
    </List>
  )
}

export default RecentActivity