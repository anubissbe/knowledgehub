import React from 'react';
import {
  Snackbar,
  Alert,
  AlertTitle,
  Button,
  Stack,
} from '@mui/material';
import { useNotifications } from '../store/hooks';

/**
 * Global notification system component
 * Displays notifications from the store as Material-UI alerts
 */
export const NotificationSystem: React.FC = () => {
  const { notifications, remove } = useNotifications();

  return (
    <Stack
      spacing={1}
      sx={{
        position: 'fixed',
        top: 16,
        right: 16,
        zIndex: 9999,
        maxWidth: 400,
      }}
    >
      {notifications.map((notification) => (
        <Snackbar
          key={notification.id}
          open={true}
          autoHideDuration={notification.duration || null}
          onClose={() => remove(notification.id)}
          anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
        >
          <Alert
            severity={notification.type}
            onClose={() => remove(notification.id)}
            variant="filled"
            sx={{
              minWidth: 300,
              boxShadow: 3,
            }}
            action={
              notification.action && (
                <Button
                  color="inherit"
                  size="small"
                  onClick={() => {
                    notification.action!.handler();
                    remove(notification.id);
                  }}
                >
                  {notification.action.label}
                </Button>
              )
            }
          >
            <AlertTitle>{notification.title}</AlertTitle>
            {notification.message}
          </Alert>
        </Snackbar>
      ))}
    </Stack>
  );
};
