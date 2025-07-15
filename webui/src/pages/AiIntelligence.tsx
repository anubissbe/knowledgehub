import React, { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  List,
  ListItem,
  ListItemText,
  Chip,
  Button,
  LinearProgress,
} from '@mui/material';
import { apiService } from '../services/api';

const AiIntelligence: React.FC = () => {
  const [mistakes, setMistakes] = useState<any[]>([]);
  const [decisions, setDecisions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [mistakesData, decisionsData] = await Promise.all([
          apiService.getMistakes(),
          apiService.getDecisions(),
        ]);
        setMistakes(mistakesData.mistakes || []);
        setDecisions(decisionsData.decisions || []);
      } catch (error) {
        console.error('Failed to fetch AI intelligence data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return <LinearProgress />;
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        AI Intelligence Systems
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Mistake Learning
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Patterns learned from errors to prevent future issues
              </Typography>
              
              <List>
                {mistakes.slice(0, 5).map((mistake, index) => (
                  <ListItem key={index}>
                    <ListItemText
                      primary={mistake.error_type}
                      secondary={mistake.solution}
                    />
                    <Chip 
                      label={`${mistake.frequency}x`} 
                      size="small" 
                      color={mistake.frequency > 2 ? "error" : "default"}
                    />
                  </ListItem>
                ))}
              </List>
              
              {mistakes.length === 0 && (
                <Typography variant="body2" color="text.secondary">
                  No mistakes tracked yet
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Decision Tracking
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Technical decisions with reasoning and confidence scores
              </Typography>
              
              <List>
                {decisions.slice(0, 5).map((decision, index) => (
                  <ListItem key={index}>
                    <ListItemText
                      primary={decision.chosen_option}
                      secondary={decision.reasoning}
                    />
                    <Chip 
                      label={`${Math.round(decision.confidence_score * 100)}%`} 
                      size="small"
                      color={decision.confidence_score > 0.8 ? "success" : "warning"}
                    />
                  </ListItem>
                ))}
              </List>
              
              {decisions.length === 0 && (
                <Typography variant="body2" color="text.secondary">
                  No decisions recorded yet
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                AI Features Status
              </Typography>
              
              <Grid container spacing={2}>
                {[
                  { name: 'Session Continuity', status: 'active' },
                  { name: 'Mistake Learning', status: 'active' },
                  { name: 'Decision Reasoning', status: 'active' },
                  { name: 'Performance Tracking', status: 'active' },
                  { name: 'Code Evolution', status: 'active' },
                  { name: 'Proactive Assistance', status: 'active' },
                  { name: 'Project Context', status: 'active' },
                  { name: 'Workflow Integration', status: 'active' },
                ].map((feature) => (
                  <Grid item xs={12} sm={6} md={3} key={feature.name}>
                    <Box display="flex" alignItems="center" justifyContent="space-between">
                      <Typography variant="body2">{feature.name}</Typography>
                      <Chip 
                        label={feature.status} 
                        size="small" 
                        color="success"
                      />
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AiIntelligence;