import {
  Box,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  Grid,
  Switch,
  FormControlLabel,
} from '@mui/material'
import { useState } from 'react'
import { useForm } from 'react-hook-form'

interface SettingsForm {
  maxCrawlDepth: number
  maxPagesPerCrawl: number
  chunkSize: number
  chunkOverlap: number
  searchResultLimit: number
  vectorSearchWeight: number
  keywordSearchWeight: number
  enableMetrics: boolean
  logLevel: string
}

function Settings() {
  const [saved, setSaved] = useState(false)

  const { register, handleSubmit } = useForm<SettingsForm>({
    defaultValues: {
      maxCrawlDepth: 3,
      maxPagesPerCrawl: 100,
      chunkSize: 800,
      chunkOverlap: 200,
      searchResultLimit: 10,
      vectorSearchWeight: 0.7,
      keywordSearchWeight: 0.3,
      enableMetrics: true,
      logLevel: 'INFO',
    },
  })

  const onSubmit = (data: SettingsForm) => {
    // In a real app, this would save to backend
    console.log('Saving settings:', data)
    setSaved(true)
    setTimeout(() => setSaved(false), 3000)
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>

      <form onSubmit={handleSubmit(onSubmit)}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Crawling Configuration
                </Typography>
                <TextField
                  {...register('maxCrawlDepth', { valueAsNumber: true })}
                  label="Max Crawl Depth"
                  type="number"
                  fullWidth
                  margin="normal"
                />
                <TextField
                  {...register('maxPagesPerCrawl', { valueAsNumber: true })}
                  label="Max Pages Per Crawl"
                  type="number"
                  fullWidth
                  margin="normal"
                />
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Chunking Configuration
                </Typography>
                <TextField
                  {...register('chunkSize', { valueAsNumber: true })}
                  label="Chunk Size (tokens)"
                  type="number"
                  fullWidth
                  margin="normal"
                />
                <TextField
                  {...register('chunkOverlap', { valueAsNumber: true })}
                  label="Chunk Overlap (tokens)"
                  type="number"
                  fullWidth
                  margin="normal"
                />
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Search Configuration
                </Typography>
                <TextField
                  {...register('searchResultLimit', { valueAsNumber: true })}
                  label="Search Result Limit"
                  type="number"
                  fullWidth
                  margin="normal"
                />
                <TextField
                  {...register('vectorSearchWeight', { valueAsNumber: true })}
                  label="Vector Search Weight"
                  type="number"
                  inputProps={{ step: 0.1, min: 0, max: 1 }}
                  fullWidth
                  margin="normal"
                />
                <TextField
                  {...register('keywordSearchWeight', { valueAsNumber: true })}
                  label="Keyword Search Weight"
                  type="number"
                  inputProps={{ step: 0.1, min: 0, max: 1 }}
                  fullWidth
                  margin="normal"
                />
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  System Configuration
                </Typography>
                <FormControlLabel
                  control={
                    <Switch {...register('enableMetrics')} defaultChecked />
                  }
                  label="Enable Metrics"
                  sx={{ mt: 2, mb: 1 }}
                />
                <TextField
                  {...register('logLevel')}
                  label="Log Level"
                  select
                  fullWidth
                  margin="normal"
                  SelectProps={{
                    native: true,
                  }}
                >
                  <option value="DEBUG">DEBUG</option>
                  <option value="INFO">INFO</option>
                  <option value="WARNING">WARNING</option>
                  <option value="ERROR">ERROR</option>
                </TextField>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
              <Button variant="outlined" type="button">
                Reset to Defaults
              </Button>
              <Button variant="contained" type="submit">
                Save Settings
              </Button>
            </Box>
            {saved && (
              <Typography color="success.main" sx={{ mt: 2, textAlign: 'right' }}>
                Settings saved successfully!
              </Typography>
            )}
          </Grid>
        </Grid>
      </form>
    </Box>
  )
}

export default Settings