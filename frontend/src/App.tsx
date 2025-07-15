import { Routes, Route, Navigate } from 'react-router-dom'
import { Box } from '@mui/material'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import AiIntelligence from './pages/AiIntelligence'
import MemorySystem from './pages/MemorySystem'
import KnowledgeGraph from './pages/KnowledgeGraph'
import SearchKnowledge from './pages/SearchKnowledge'
import ApiDocs from './pages/ApiDocs'
import Settings from './pages/Settings'

function App() {
  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="ai" element={<AiIntelligence />} />
          <Route path="memory" element={<MemorySystem />} />
          <Route path="knowledge-graph" element={<KnowledgeGraph />} />
          <Route path="search" element={<SearchKnowledge />} />
          <Route path="api-docs" element={<ApiDocs />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </Box>
  )
}

export default App