import React from 'react'
import { Box, BoxProps } from '@mui/material'

// Enhanced interface with error boundary support
interface PageWrapperProps extends Omit<BoxProps, 'children'> {
  children: React.ReactNode
  className?: string
  'data-testid'?: string
  fallback?: React.ComponentType<{ error: Error; errorInfo: React.ErrorInfo }>
}

// Error Boundary Component
class ErrorBoundary extends React.Component<{
  children: React.ReactNode
  fallback?: React.ComponentType<{ error: Error; errorInfo: React.ErrorInfo }>
}, {
  hasError: boolean
  error?: Error
  errorInfo?: React.ErrorInfo
}> {
  constructor(props: { children: React.ReactNode; fallback?: React.ComponentType<{ error: Error; errorInfo: React.ErrorInfo }> }) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    this.setState({ error, errorInfo })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        const FallbackComponent = this.props.fallback
        return <FallbackComponent error={this.state.error!} errorInfo={this.state.errorInfo!} />
      }

      return (
        <Box
          sx={{
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexDirection: 'column',
            color: 'error.main',
            p: 3,
          }}
        >
          <h2>Something went wrong</h2>
          <p>An error occurred while rendering this page.</p>
          {import.meta.env.DEV && this.state.error && (
            <details style={{ marginTop: 16, fontSize: '0.875rem' }}>
              <summary>Error details</summary>
              <pre style={{ whiteSpace: 'pre-wrap', marginTop: 8 }}>
                {this.state.error.toString()}
              </pre>
            </details>
          )}
        </Box>
      )
    }

    return this.props.children
  }
}

// Enhanced PageWrapper with error boundary and performance optimizations
const PageWrapper = React.memo<PageWrapperProps>(({
  children,
  className,
  'data-testid': dataTestId = 'page-wrapper',
  fallback,
  sx,
  ...props
}) => {
  // Memoized default styles
  const defaultStyles = React.useMemo(() => ({
    width: '100%',
    height: '100%',
    display: 'flex',
    flexDirection: 'column' as const,
    overflow: 'hidden' as const,
    ...sx,
  }), [sx])

  return (
    <ErrorBoundary fallback={fallback}>
      <Box
        className={className}
        data-testid={dataTestId}
        sx={defaultStyles}
        {...props}
      >
        {children}
      </Box>
    </ErrorBoundary>
  )
})

PageWrapper.displayName = 'PageWrapper'

export default PageWrapper
