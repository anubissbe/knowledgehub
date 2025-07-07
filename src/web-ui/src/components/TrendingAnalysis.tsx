import React from 'react'
import {
  Card,
  CardContent,
  Typography,
  Box,
  Tabs,
  Tab,
  Grid,
  Chip
} from '@mui/material'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/services/api'

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

function TabPanel({ children, value, index }: TabPanelProps) {
  return (
    <div hidden={value !== index}>
      {value === index && (
        <Box sx={{ pt: 3 }}>
          {children}
        </Box>
      )}
    </div>
  )
}

export default function TrendingAnalysis() {
  const [tabValue, setTabValue] = React.useState(0)
  
  const { data: trends, isLoading } = useQuery({
    queryKey: ['trending-analysis'],
    queryFn: api.getTrendingAnalysis,
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue)
  }

  if (isLoading || !trends) {
    return (
      <Card>
        <CardContent>
          <Typography>Loading trending analysis...</Typography>
        </CardContent>
      </Card>
    )
  }

  const formatTooltipValue = (value: number, name: string) => {
    if (name.includes('Time') || name.includes('Duration')) {
      return [`${value}ms`, name]
    }
    return [value.toLocaleString(), name]
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Trending Analysis
        </Typography>
        
        <Tabs value={tabValue} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tab label="Usage Trends" />
          <Tab label="Performance" />
          <Tab label="Popular Content" />
        </Tabs>
        
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Daily Activity
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={trends.daily_activity}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip formatter={formatTooltipValue} />
                  <Area 
                    type="monotone" 
                    dataKey="searches" 
                    stackId="1"
                    stroke="#8884d8" 
                    fill="#8884d8" 
                    name="Searches"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="documents_added" 
                    stackId="1"
                    stroke="#82ca9d" 
                    fill="#82ca9d" 
                    name="Documents Added"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="jobs_completed" 
                    stackId="1"
                    stroke="#ffc658" 
                    fill="#ffc658" 
                    name="Jobs Completed"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>
                Response Times
              </Typography>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={trends.response_times}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip formatter={formatTooltipValue} />
                  <Line 
                    type="monotone" 
                    dataKey="search_time" 
                    stroke="#8884d8" 
                    name="Search Time"
                    strokeWidth={2}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="api_time" 
                    stroke="#82ca9d" 
                    name="API Time"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>
                System Load
              </Typography>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={trends.system_load}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${value}%`, 'Usage']} />
                  <Area 
                    type="monotone" 
                    dataKey="cpu" 
                    stroke="#ff7300" 
                    fill="#ff7300" 
                    name="CPU"
                    fillOpacity={0.6}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="memory" 
                    stroke="#00ff00" 
                    fill="#00ff00" 
                    name="Memory"
                    fillOpacity={0.4}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>
                Top Search Queries
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={trends.top_queries} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="query" type="category" width={100} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8884d8" name="Search Count" />
                </BarChart>
              </ResponsiveContainer>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>
                Popular Sources
              </Typography>
              <Box sx={{ mt: 2 }}>
                {trends.popular_sources?.map((source: any) => (
                  <Box key={source.name} sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">{source.name}</Typography>
                      <Chip 
                        label={`${source.searches} searches`} 
                        size="small" 
                        color="primary"
                        variant="outlined"
                      />
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="caption" color="text.secondary">
                        {source.documents} documents
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Last updated: {source.last_updated}
                      </Typography>
                    </Box>
                  </Box>
                ))}
              </Box>
            </Grid>
          </Grid>
        </TabPanel>
      </CardContent>
    </Card>
  )
}