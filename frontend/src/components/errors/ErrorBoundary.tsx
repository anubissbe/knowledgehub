import React, { Component, ReactNode } from "react";
import { Box, Typography, Button, Paper, Alert } from "@mui/material";
import { ErrorHandler } from "../../services/errors";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: any) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: any;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    
    this.setState({
      error,
      errorInfo,
    });

    // Call custom error handler
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Log to error handling service
    const handledError = ErrorHandler.handle(error, "React ErrorBoundary");
  }

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          minHeight="200px"
          p={3}
        >
          <Paper elevation={1} sx={{ p: 3, maxWidth: 600 }}>
            <Alert severity="error" sx={{ mb: 2 }}>
              <Typography variant="h6" gutterBottom>
                Something went wrong
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {this.state.error?.message || "An unexpected error occurred"}
              </Typography>
            </Alert>

            <Box display="flex" gap={2} justifyContent="center">
              <Button variant="contained" onClick={this.handleRetry}>
                Try Again
              </Button>
              <Button
                variant="outlined"
                onClick={() => window.location.reload()}
              >
                Reload Page
              </Button>
            </Box>

            {import.meta.env.DEV && this.state.errorInfo && (
              <Box mt={2}>
                <Typography variant="subtitle2" gutterBottom>
                  Error Details (Development Only):
                </Typography>
                <Paper
                  variant="outlined"
                  sx={{
                    p: 2,
                    backgroundColor: "#f5f5f5",
                    overflow: "auto",
                    maxHeight: 300,
                  }}
                >
                  <pre style={{ margin: 0, fontSize: "0.75rem" }}>
                    {this.state.error?.stack}
                  </pre>
                  <pre style={{ margin: 0, fontSize: "0.75rem", marginTop: 10 }}>
                    {JSON.stringify(this.state.errorInfo, null, 2)}
                  </pre>
                </Paper>
              </Box>
            )}
          </Paper>
        </Box>
      );
    }

    return this.props.children;
  }
}

// HOC for wrapping components with error boundary
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  fallback?: ReactNode,
  onError?: (error: Error, errorInfo: any) => void
) {
  return function WrappedComponent(props: P) {
    return (
      <ErrorBoundary fallback={fallback} onError={onError}>
        <Component {...props} />
      </ErrorBoundary>
    );
  };
}
