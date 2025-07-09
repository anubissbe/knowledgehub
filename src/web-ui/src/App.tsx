import { Routes, Route, Navigate } from 'react-router-dom'
import { Box } from '@mui/material'
import Layout from '@/components/Layout'
import Dashboard from '@/pages/Dashboard'
import Sources from '@/pages/Sources'
import Search from '@/pages/Search'
import Jobs from '@/pages/Jobs'
import Memory from '@/pages/Memory'
import Settings from '@/pages/Settings'
import Analytics from '@/pages/Analytics'

function App() {
  return (
    <Box sx={{ display: 'flex' }}>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="sources" element={<Sources />} />
          <Route path="search" element={<Search />} />
          <Route path="jobs" element={<Jobs />} />
          <Route path="memory" element={<Memory />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </Box>
  )
}

export default App