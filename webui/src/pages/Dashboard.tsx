import React, { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  Psychology as PsychologyIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';
import { apiService } from '../services/api';

interface DashboardStats {
  mistakes: {
    total: number;
    solved: number;
    patterns: number;
  };
  decisions: {
    total: number;
    avg_confidence: number;
  };
  performance: {
    total_tracked: number;
    avg_duration: number;
  };
  sessions: {
    active: number;
    total: number;
  };
}

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await apiService.getAiFeaturesSummary();
        setStats(response);
      } catch (err) {
        setError('Failed to load dashboard statistics');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  const statCards = [
    {
      title: 'Learning Progress',
      value: stats?.mistakes.solved || 0,
      total: stats?.mistakes.total || 0,
      icon: <PsychologyIcon />,
      color: '#4caf50',
      subtitle: 'Mistakes Resolved',
    },
    {
      title: 'Active Sessions',
      value: stats?.sessions.active || 0,
      total: stats?.sessions.total || 0,
      icon: <MemoryIcon />,
      color: '#2196f3',
      subtitle: 'Memory Sessions',
    },
    {
      title: 'Performance',
      value: `${Math.round(stats?.performance.avg_duration || 0)}ms`,
      icon: <SpeedIcon />,
      color: '#ff9800',
      subtitle: 'Avg Response Time',
    },
    {
      title: 'Decision Confidence',
      value: `${Math.round((stats?.decisions.avg_confidence || 0) * 100)}%`,
      icon: <TrendingUpIcon />,
      color: '#9c27b0',
      subtitle: 'Average Score',
    },
  ];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        KnowledgeHub Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Real-time overview of your AI intelligence systems
      </Typography>

      <Grid container spacing={3}>
        {statCards.map((card, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <Box
                    sx={{
                      backgroundColor: card.color,
                      color: 'white',
                      p: 1,
                      borderRadius: 1,
                      display: 'flex',
                      mr: 2,
                    }}
                  >
                    {card.icon}
                  </Box>
                  <Typography variant="h6" component="div">
                    {card.title}
                  </Typography>
                </Box>
                <Typography variant="h3" component="div" gutterBottom>
                  {card.value}
                  {card.total && (
                    <Typography variant="body2" component="span" color="text.secondary">
                      {' / '}{card.total}
                    </Typography>
                  )}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {card.subtitle}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Grid container spacing={3} mt={2}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Health
              </Typography>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  All systems operational
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Activity
              </Typography>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  {stats?.performance.total_tracked || 0} commands tracked
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {stats?.mistakes.patterns || 0} patterns identified
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {stats?.decisions.total || 0} decisions recorded
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;