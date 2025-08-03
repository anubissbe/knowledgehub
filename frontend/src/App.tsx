import { Routes, Route, Navigate } from 'react-router-dom'
import { Box } from '@mui/material'
import SimpleLayout from './components/SimpleLayout'
import Dashboard from './pages/Dashboard'
import UltraModernDashboard from './pages/UltraModernDashboard'
import AiIntelligence from './pages/AiIntelligence'
import AiIntelligenceFixed from './pages/AiIntelligenceFixed'
import MemorySystem from './pages/MemorySystem'
import HybridMemory from './pages/HybridMemory'
import KnowledgeGraph from './pages/KnowledgeGraph'
import SearchKnowledge from './pages/SearchKnowledge'
import Sources from './pages/Sources'
import ApiDocs from './pages/ApiDocs'
import Settings from './pages/Settings'
import TestResponsive from './pages/TestResponsive'

function App() {
  return (
    <Routes>
      <Route path="/" element={<SimpleLayout />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="ultra" element={<UltraModernDashboard />} />
        <Route path="ai" element={<AiIntelligenceFixed />} />
        <Route path="memory" element={<MemorySystem />} />
        <Route path="hybrid-memory" element={<HybridMemory />} />
        <Route path="knowledge-graph" element={<KnowledgeGraph />} />
        <Route path="search" element={<SearchKnowledge />} />
        <Route path="sources" element={<Sources />} />
        <Route path="api-docs" element={<ApiDocs />} />
        <Route path="settings" element={<Settings />} />
        <Route path="test" element={<TestResponsive />} />
      </Route>
    </Routes>
  )
}

export default App