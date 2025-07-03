import { List, ListItem, ListItemText, Chip } from '@mui/material'
import { useQuery } from '@tanstack/react-query'

interface ServiceHealth {
  name: string
  status: 'healthy' | 'unhealthy' | 'unknown'
  latency?: number
}

function SystemHealth() {
  const { data: health } = useQuery({
    queryKey: ['system-health'],
    queryFn: async () => {
      // This would normally call a health check endpoint
      // For now, return mock data
      return [
        { name: 'API Server', status: 'healthy', latency: 12 },
        { name: 'Database', status: 'healthy', latency: 5 },
        { name: 'Redis Cache', status: 'healthy', latency: 2 },
        { name: 'Weaviate', status: 'healthy', latency: 15 },
        { name: 'MCP Server', status: 'healthy', latency: 8 },
      ] as ServiceHealth[]
    },
    refetchInterval: 60000, // Check every minute
  })

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'success'
      case 'unhealthy':
        return 'error'
      default:
        return 'warning'
    }
  }

  return (
    <List dense>
      {health?.map((service) => (
        <ListItem key={service.name}>
          <ListItemText
            primary={service.name}
            secondary={service.latency ? `${service.latency}ms` : undefined}
          />
          <Chip
            label={service.status}
            size="small"
            color={getStatusColor(service.status)}
          />
        </ListItem>
      ))}
    </List>
  )
}

export default SystemHealth