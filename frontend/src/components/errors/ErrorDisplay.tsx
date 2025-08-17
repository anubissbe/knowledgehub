import React from "react";
import { Alert, AlertTitle, Box, Button, Typography, Collapse } from "@mui/material";
import { ExpandMore, ExpandLess, Refresh, Close } from "@mui/icons-material";
import { ErrorHandler, ApiClientError } from "../../services/errors";

interface ErrorDisplayProps {
  error: Error | ApiClientError | string | null;
  onRetry?: () => void;
  onDismiss?: () => void;
  showDetails?: boolean;
  severity?: "error" | "warning" | "info";
  title?: string;
}

export const ErrorDisplay: React.FC<ErrorDisplayProps> = ({
  error,
  onRetry,
  onDismiss,
  showDetails = false,
  severity = "error",
  title,
}) => {
  const [expanded, setExpanded] = React.useState(false);

  if (!error) return null;

  const errorMessage = typeof error === "string" 
    ? error 
    : error instanceof ApiClientError 
    ? ErrorHandler.getUserFriendlyMessage(error)
    : error.message;

  const errorCode = error instanceof ApiClientError ? error.code : undefined;
  const errorDetails = error instanceof ApiClientError ? error.details : undefined;
  const timestamp = error instanceof ApiClientError ? error.timestamp : new Date().toISOString();

  return (
    <Alert 
      severity={severity} 
      action={
        <Box display="flex" gap={1}>
          {showDetails && (
            <Button
              size="small"
              onClick={() => setExpanded(!expanded)}
              startIcon={expanded ? <ExpandLess /> : <ExpandMore />}
            >
              Details
            </Button>
          )}
          {onRetry && (
            <Button
              size="small"
              onClick={onRetry}
              startIcon={<Refresh />}
              variant="outlined"
            >
              Retry
            </Button>
          )}
          {onDismiss && (
            <Button
              size="small"
              onClick={onDismiss}
              startIcon={<Close />}
            >
              Dismiss
            </Button>
          )}
        </Box>
      }
    >
      {title && <AlertTitle>{title}</AlertTitle>}
      <Typography variant="body2">
        {errorMessage}
      </Typography>
      
      {showDetails && (
        <Collapse in={expanded}>
          <Box mt={2} p={2} bgcolor="rgba(0,0,0,0.05)" borderRadius={1}>
            <Typography variant="subtitle2" gutterBottom>
              Error Details
            </Typography>
            
            {errorCode && (
              <Typography variant="body2" component="div" gutterBottom>
                <strong>Code:</strong> {errorCode}
              </Typography>
            )}
            
            <Typography variant="body2" component="div" gutterBottom>
              <strong>Time:</strong> {new Date(timestamp).toLocaleString()}
            </Typography>
            
            {error instanceof Error && error.stack && (
              <Box mt={1}>
                <Typography variant="body2" component="div" gutterBottom>
                  <strong>Stack Trace:</strong>
                </Typography>
                <pre style={{ 
                  fontSize: "0.75rem", 
                  overflow: "auto", 
                  maxHeight: 200,
                  backgroundColor: "rgba(0,0,0,0.1)",
                  padding: "8px",
                  borderRadius: "4px",
                  margin: 0
                }}>
                  {error.stack}
                </pre>
              </Box>
            )}
            
            {errorDetails && (
              <Box mt={1}>
                <Typography variant="body2" component="div" gutterBottom>
                  <strong>Additional Details:</strong>
                </Typography>
                <pre style={{ 
                  fontSize: "0.75rem", 
                  overflow: "auto", 
                  maxHeight: 150,
                  backgroundColor: "rgba(0,0,0,0.1)",
                  padding: "8px",
                  borderRadius: "4px",
                  margin: 0
                }}>
                  {JSON.stringify(errorDetails, null, 2)}
                </pre>
              </Box>
            )}
          </Box>
        </Collapse>
      )}
    </Alert>
  );
};

// Hook for error state management
export function useErrorHandler() {
  const [error, setError] = React.useState<Error | ApiClientError | string | null>(null);

  const handleError = React.useCallback((error: unknown) => {
    const handledError = ErrorHandler.handle(error);
    setError(handledError);
  }, []);

  const clearError = React.useCallback(() => {
    setError(null);
  }, []);

  const retry = React.useCallback((operation: () => void | Promise<void>) => {
    setError(null);
    try {
      const result = operation();
      if (result instanceof Promise) {
        result.catch(handleError);
      }
    } catch (err) {
      handleError(err);
    }
  }, [handleError]);

  return {
    error,
    handleError,
    clearError,
    retry,
    hasError: error !== null,
  };
}
