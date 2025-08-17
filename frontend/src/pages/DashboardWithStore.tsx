import React, { useEffect } from 'react';
import { Grid, Box, Typography, Chip, LinearProgress, alpha } from '@mui/material';
import { 
  Memory, 
  Psychology, 
  Speed, 
  Storage,
  Security,
  Cloud,
  AutoAwesome,
  Timeline,
  BubbleChart,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import PageWrapper from '../components/PageWrapper';
import UltraHeader from '../components/ultra/UltraHeader';
import MetricCard from '../components/ultra/MetricCard';
import GlassCard from '../components/GlassCard';
import AnimatedChart from '../components/AnimatedChart';
import Network3D from '../components/Network3D';
import { Layout } from '../components/StoreLayout';
import { useAI, useApp, useNotifications, useRealTimeConnection } from '../store';

const METRIC_COLORS = {
  primary: '#2196F3',
  secondary: '#FF00FF',
  success: '#00FF88',
  warning: '#FFD700',
  error: '#FF3366',
  info: '#00FFFF',
  violet: '#8B5CF6',
  pink: '#EC4899',
};

// 3D network data
const generate3DNetwork = () => {
  const nodes = [
    { id: '1', label: 'Core', position: [0, 0, 0] as [number, number, number], color: METRIC_COLORS.primary, size: 0.8 },
    { id: '2', label: 'API', position: [3, 1, 0] as [number, number, number], color: METRIC_COLORS.success, size: 0.6 },
    { id: '3', label: 'Cache', position: [-3, 1, 0] as [number, number, number], color: METRIC_COLORS.secondary, size: 0.6 },
    { id: '4', label: 'DB', position: [0, 1, 3] as [number, number, number], color: METRIC_COLORS.warning, size: 0.5 },
    { id: '5', label: 'AI', position: [0, 1, -3] as [number, number, number], color: METRIC_COLORS.info, size: 0.5 },
  ];
  
  const edges = [
    { from: '1', to: '2' },
    { from: '1', to: '3' },
    { from: '1', to: '4' },
    { from: '1', to: '5' },
    { from: '2', to: '4' },
  ];

  return { nodes, edges };
};

export default function Dashboard() {
  // Use store hooks
  const { metrics, features, fetchFeatures, isLoading } = useAI();
  const { isInitialized } = useApp();
  const { success } = useNotifications();
  const { isConnected, enabled: realtimeEnabled } = useRealTimeConnection();

  useEffect(() => {
    if (isInitialized && !features.length) {
      fetchFeatures();
    }
  }, [isInitialized, features.length, fetchFeatures]);

  // Show connection status notification
  useEffect(() => {
    if (isConnected) {
      success('Real-time Connected', 'Live data updates are now active');
    }
  }, [isConnected, success]);

  const chartData = [
    { name: 'Memory Usage', value: metrics.memoriesCreated, color: METRIC_COLORS.primary },
    { name: 'Queries', value: metrics.queriesProcessed, color: METRIC_COLORS.success },
    { name: 'Accuracy', value: Math.round(metrics.accuracy * 100), color: METRIC_COLORS.warning },
    { name: 'Response Time', value: Math.round(metrics.responseTime), color: METRIC_COLORS.info },
  ];

  return (
    <Layout title="Dashboard" breadcrumbs={[{ label: 'Home', path: '/' }]}>
      <PageWrapper>
        <UltraHeader 
          title="AI Intelligence Dashboard" 
          subtitle="Real-time insights and system analytics"
          gradient="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        />

        <Box sx={{ mb: 4 }}>
          <Grid container spacing={3}>
            {/* Status indicators */}
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                <Chip 
                  icon={<Cloud />} 
                  label={isConnected ? 'Connected' : 'Disconnected'} 
                  color={isConnected ? 'success' : 'error'} 
                  variant="filled" 
                />
                <Chip 
                  icon={<AutoAwesome />} 
                  label={`AI Features: ${features.length}`} 
                  color="primary" 
                  variant="outlined" 
                />
                <Chip 
                  icon={<Timeline />} 
                  label={`${metrics.uptime}% Uptime`} 
                  color="info" 
                  variant="outlined" 
                />
                {realtimeEnabled && (
                  <Chip 
                    icon={<Speed />} 
                    label="Real-time Enabled" 
                    color="secondary" 
                    variant="filled" 
                  />
                )}
              </Box>
            </Grid>

            {/* Metric Cards */}
            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                title="Active Sessions"
                value={metrics.sessionsActive}
                change="+12%"
                changeType="positive"
                icon={<Psychology />}
                color={METRIC_COLORS.primary}
                loading={isLoading}
              />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                title="Memories Created"
                value={metrics.memoriesCreated}
                change="+24%"
                changeType="positive"
                icon={<Memory />}
                color={METRIC_COLORS.success}
                loading={isLoading}
              />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                title="Queries Processed"
                value={metrics.queriesProcessed}
                change="+8%"
                changeType="positive"
                icon={<Speed />}
                color={METRIC_COLORS.warning}
                loading={isLoading}
              />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                title="Response Time"
                value={`${Math.round(metrics.responseTime)}ms`}
                change="-5%"
                changeType="positive"
                icon={<Timeline />}
                color={METRIC_COLORS.info}
                loading={isLoading}
              />
            </Grid>

            {/* Charts and visualizations */}
            <Grid item xs={12} md={6}>
              <GlassCard
                sx={{
                  background: `linear-gradient(135deg, ${alpha(METRIC_COLORS.primary, 0.1)} 0%, ${alpha(METRIC_COLORS.secondary, 0.1)} 100%)`,
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  height: 400,
                  display: 'flex',
                  flexDirection: 'column',
                }}
              >
                <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                  Performance Metrics
                </Typography>
                <Box sx={{ flex: 1 }}>
                  <AnimatedChart data={chartData} />
                </Box>
              </GlassCard>
            </Grid>

            <Grid item xs={12} md={6}>
              <GlassCard
                sx={{
                  background: `linear-gradient(135deg, ${alpha(METRIC_COLORS.success, 0.1)} 0%, ${alpha(METRIC_COLORS.warning, 0.1)} 100%)`,
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  height: 400,
                  display: 'flex',
                  flexDirection: 'column',
                }}
              >
                <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                  System Architecture
                </Typography>
                <Box sx={{ flex: 1, position: 'relative' }}>
                  <Network3D data={generate3DNetwork()} />
                </Box>
              </GlassCard>
            </Grid>

            {/* AI Features */}
            <Grid item xs={12}>
              <GlassCard
                sx={{
                  background: `linear-gradient(135deg, ${alpha(METRIC_COLORS.violet, 0.1)} 0%, ${alpha(METRIC_COLORS.pink, 0.1)} 100%)`,
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(255,255,255,0.1)',
                }}
              >
                <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
                  AI Features Status
                </Typography>
                <Grid container spacing={2}>
                  {features.length > 0 ? (
                    features.map((feature, index) => (
                      <Grid item xs={12} sm={6} md={4} key={feature.id}>
                        <motion.div
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: index * 0.1 }}
                        >
                          <Box
                            sx={{
                              p: 2,
                              borderRadius: 2,
                              background: alpha(METRIC_COLORS.primary, 0.05),
                              border: '1px solid rgba(255,255,255,0.05)',
                            }}
                          >
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                              <Typography variant="subtitle1" sx={{ fontWeight: 600, flex: 1 }}>
                                {feature.name}
                              </Typography>
                              <Chip 
                                label={feature.status} 
                                color={feature.status === 'active' ? 'success' : 'warning'}
                                size="small"
                              />
                            </Box>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                              {feature.description}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              Updated: {new Date(feature.lastUpdated).toLocaleDateString()}
                            </Typography>
                          </Box>
                        </motion.div>
                      </Grid>
                    ))
                  ) : (
                    <Grid item xs={12}>
                      <Box sx={{ textAlign: 'center', py: 4 }}>
                        {isLoading ? (
                          <LinearProgress sx={{ mb: 2 }} />
                        ) : null}
                        <Typography color="text.secondary">
                          {isLoading ? 'Loading AI features...' : 'No AI features available'}
                        </Typography>
                      </Box>
                    </Grid>
                  )}
                </Grid>
              </GlassCard>
            </Grid>
          </Grid>
        </Box>
      </PageWrapper>
    </Layout>
  );
}
