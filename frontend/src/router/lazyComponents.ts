import { lazy } from 'react'

// Lazy load all page components for code splitting

// Core Pages
export const Dashboard = lazy(() => import('../pages/Dashboard'))
export const AiIntelligence = lazy(() => import('../pages/AiIntelligence'))
export const MemorySystem = lazy(() => import('../pages/MemorySystem'))
export const HybridMemory = lazy(() => import('../pages/HybridMemory'))
export const KnowledgeGraph = lazy(() => import('../pages/KnowledgeGraph'))
export const SearchKnowledge = lazy(() => import('../pages/SearchKnowledge'))
export const Sources = lazy(() => import('../pages/Sources'))
export const ApiDocs = lazy(() => import('../pages/ApiDocs'))
export const Settings = lazy(() => import('../pages/Settings'))

// Enhanced RAG Pages
export const HybridRAGDashboard = lazy(() => import('../pages/HybridRAGDashboard'))
export const AgentWorkflows = lazy(() => import('../pages/AgentWorkflows'))
export const RetrievalAnalytics = lazy(() => import('../pages/RetrievalAnalytics'))
export const WebIngestionMonitor = lazy(() => import('../pages/WebIngestionMonitor'))
export const MemoryClusterView = lazy(() => import('../pages/MemoryClusterView'))
export const SystemObservability = lazy(() => import('../pages/SystemObservability'))

// Layout components
export const SimpleLayout = lazy(() => import('../components/SimpleLayout'))
