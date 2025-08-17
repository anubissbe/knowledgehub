import { Suspense } from 'react'
import { createBrowserRouter, Navigate } from 'react-router-dom'
import { RouterErrorBoundary } from './ErrorBoundary'
import { RouteGuard } from './RouteGuard'
import { LoadingFallback } from './LoadingFallback'
import * as LazyComponents from './lazyComponents'

// Wrapper component for lazy loading with suspense
const LazyWrapper = ({ 
  Component, 
  requireAuth = false,
  fallbackText 
}: { 
  Component: React.LazyExoticComponent<any>
  requireAuth?: boolean
  fallbackText?: string
}) => (
  <RouteGuard requireAuth={requireAuth}>
    <Suspense fallback={<LoadingFallback text={fallbackText} />}>
      <Component />
    </Suspense>
  </RouteGuard>
)

// Create the modern React Router v6 configuration
export const router = createBrowserRouter([
  {
    path: '/',
    element: (
      <Suspense fallback={<LoadingFallback text="Loading application..." />}>
        <LazyComponents.SimpleLayout />
      </Suspense>
    ),
    errorElement: <RouterErrorBoundary />,
    children: [
      {
        index: true,
        element: <Navigate to="/dashboard" replace />,
      },
      {
        path: 'dashboard',
        element: <LazyWrapper 
          Component={LazyComponents.Dashboard} 
          fallbackText="Loading dashboard..." 
        />,
        errorElement: <RouterErrorBoundary />,
      },
      {
        path: 'ai',
        element: <LazyWrapper 
          Component={LazyComponents.AiIntelligence} 
          fallbackText="Loading AI Intelligence..." 
        />,
        errorElement: <RouterErrorBoundary />,
      },
      {
        path: 'memory',
        element: <LazyWrapper 
          Component={LazyComponents.MemorySystem} 
          fallbackText="Loading Memory System..." 
        />,
        errorElement: <RouterErrorBoundary />,
      },
      {
        path: 'hybrid-memory',
        element: <LazyWrapper 
          Component={LazyComponents.HybridMemory} 
          fallbackText="Loading Hybrid Memory..." 
        />,
        errorElement: <RouterErrorBoundary />,
      },
      {
        path: 'knowledge-graph',
        element: <LazyWrapper 
          Component={LazyComponents.KnowledgeGraph} 
          fallbackText="Loading Knowledge Graph..." 
        />,
        errorElement: <RouterErrorBoundary />,
      },
      {
        path: 'search',
        element: <LazyWrapper 
          Component={LazyComponents.SearchKnowledge} 
          fallbackText="Loading Search..." 
        />,
        errorElement: <RouterErrorBoundary />,
      },
      {
        path: 'sources',
        element: <LazyWrapper 
          Component={LazyComponents.Sources} 
          fallbackText="Loading Sources..." 
        />,
        errorElement: <RouterErrorBoundary />,
      },
      {
        path: 'api-docs',
        element: <LazyWrapper 
          Component={LazyComponents.ApiDocs} 
          fallbackText="Loading API Documentation..." 
        />,
        errorElement: <RouterErrorBoundary />,
      },
      {
        path: 'settings',
        element: <LazyWrapper 
          Component={LazyComponents.Settings} 
          fallbackText="Loading Settings..." 
          requireAuth={false}
        />,
        errorElement: <RouterErrorBoundary />,
      },
      // Enhanced RAG Routes
      {
        path: 'hybrid-rag',
        element: <LazyWrapper 
          Component={LazyComponents.HybridRAGDashboard} 
          fallbackText="Loading Hybrid RAG Dashboard..." 
        />,
        errorElement: <RouterErrorBoundary />,
      },
      {
        path: 'agent-workflows',
        element: <LazyWrapper 
          Component={LazyComponents.AgentWorkflows} 
          fallbackText="Loading Agent Workflows..." 
        />,
        errorElement: <RouterErrorBoundary />,
      },
      {
        path: 'retrieval-analytics',
        element: <LazyWrapper 
          Component={LazyComponents.RetrievalAnalytics} 
          fallbackText="Loading Retrieval Analytics..." 
        />,
        errorElement: <RouterErrorBoundary />,
      },
      {
        path: 'web-ingestion',
        element: <LazyWrapper 
          Component={LazyComponents.WebIngestionMonitor} 
          fallbackText="Loading Web Ingestion Monitor..." 
        />,
        errorElement: <RouterErrorBoundary />,
      },
      {
        path: 'memory-clusters',
        element: <LazyWrapper 
          Component={LazyComponents.MemoryClusterView} 
          fallbackText="Loading Memory Clusters..." 
        />,
        errorElement: <RouterErrorBoundary />,
      },
      {
        path: 'observability',
        element: <LazyWrapper 
          Component={LazyComponents.SystemObservability} 
          fallbackText="Loading System Observability..." 
        />,
        errorElement: <RouterErrorBoundary />,
      },
    ],
  },
  {
    path: '*',
    element: <RouterErrorBoundary />,
  },
], {
  future: {
    v7_startTransition: true,
    v7_relativeSplatPath: true,
  },
})

export default router
