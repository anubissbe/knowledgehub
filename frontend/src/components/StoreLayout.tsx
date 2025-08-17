import React from 'react';
import { Box } from '@mui/material';
import { NotificationSystem } from './NotificationSystem';
import { usePage } from '../store/hooks';

interface LayoutProps {
  children: React.ReactNode;
  title?: string;
  breadcrumbs?: Array<{ label: string; path: string }>;
}

export const Layout: React.FC<LayoutProps> = ({ 
  children, 
  title,
  breadcrumbs 
}) => {
  const { setPage } = usePage();

  React.useEffect(() => {
    if (title) {
      setPage(title, breadcrumbs);
    }
  }, [title, breadcrumbs, setPage]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {children}
      <NotificationSystem />
    </Box>
  );
};
