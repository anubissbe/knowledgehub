import { createTheme, alpha } from '@mui/material/styles'

// Modern color palette with neon accents
const modernColors = {
  primary: '#2196F3',    // Electric Blue
  secondary: '#FF00FF',  // Neon Magenta
  success: '#00FF88',    // Neon Green
  error: '#FF3366',      // Neon Red
  warning: '#FFD700',    // Gold
  info: '#00FFFF',       // Cyan
  violet: '#8B5CF6',     // Violet
  indigo: '#6366F1',     // Indigo
  pink: '#EC4899',       // Pink
}

export const createModernTheme = (mode: 'light' | 'dark') => {
  const isDark = mode === 'dark'
  
  return createTheme({
    palette: {
      mode,
      primary: {
        main: modernColors.primary,
        light: alpha(modernColors.primary, 0.8),
        dark: alpha(modernColors.primary, 1),
      },
      secondary: {
        main: modernColors.secondary,
        light: alpha(modernColors.secondary, 0.8),
        dark: alpha(modernColors.secondary, 1),
      },
      success: {
        main: modernColors.success,
      },
      error: {
        main: modernColors.error,
      },
      warning: {
        main: modernColors.warning,
      },
      info: {
        main: modernColors.info,
      },
      background: {
        default: isDark ? '#0A0A0F' : '#F5F5FF',
        paper: isDark ? '#121218' : '#FFFFFF',
      },
      text: {
        primary: isDark ? '#FFFFFF' : '#1A1A2E',
        secondary: isDark ? alpha('#FFFFFF', 0.7) : alpha('#1A1A2E', 0.7),
      },
    },
    typography: {
      fontFamily: "'Inter', 'Roboto', 'Helvetica Neue', Arial, sans-serif",
      h1: {
        fontWeight: 800,
        letterSpacing: '-0.02em',
      },
      h2: {
        fontWeight: 700,
        letterSpacing: '-0.01em',
      },
      h3: {
        fontWeight: 700,
        letterSpacing: '-0.01em',
      },
      h4: {
        fontWeight: 600,
      },
      h5: {
        fontWeight: 600,
      },
      h6: {
        fontWeight: 600,
      },
      body1: {
        letterSpacing: '0.01em',
      },
      button: {
        textTransform: 'none',
        fontWeight: 600,
      },
    },
    shape: {
      borderRadius: 16,
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            padding: '10px 24px',
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: `0 8px 24px ${alpha(modernColors.primary, 0.3)}`,
            },
          },
          contained: {
            background: `linear-gradient(135deg, ${modernColors.primary} 0%, ${modernColors.violet} 100%)`,
            '&:hover': {
              background: `linear-gradient(135deg, ${modernColors.violet} 0%, ${modernColors.primary} 100%)`,
            },
          },
          outlined: {
            borderWidth: 2,
            '&:hover': {
              borderWidth: 2,
              backgroundColor: alpha(modernColors.primary, 0.1),
            },
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            backdropFilter: 'blur(10px)',
            backgroundColor: isDark 
              ? alpha('#121218', 0.8) 
              : alpha('#FFFFFF', 0.8),
            border: `1px solid ${alpha(isDark ? '#FFFFFF' : '#000000', 0.1)}`,
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: 'none',
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            fontWeight: 600,
          },
          filled: {
            background: isDark
              ? alpha(modernColors.primary, 0.2)
              : alpha(modernColors.primary, 0.1),
            color: modernColors.primary,
            '&:hover': {
              background: isDark
                ? alpha(modernColors.primary, 0.3)
                : alpha(modernColors.primary, 0.2),
            },
          },
        },
      },
      MuiTooltip: {
        styleOverrides: {
          tooltip: {
            backgroundColor: isDark 
              ? alpha('#000000', 0.9) 
              : alpha('#FFFFFF', 0.9),
            backdropFilter: 'blur(10px)',
            border: `1px solid ${alpha(modernColors.primary, 0.2)}`,
            borderRadius: 8,
            fontSize: '0.875rem',
            padding: '8px 12px',
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: isDark
              ? alpha('#0A0A0F', 0.8)
              : alpha('#F5F5FF', 0.8),
            backdropFilter: 'blur(20px)',
            borderBottom: `1px solid ${alpha(isDark ? '#FFFFFF' : '#000000', 0.1)}`,
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            backgroundColor: isDark
              ? alpha('#0A0A0F', 0.95)
              : alpha('#F5F5FF', 0.95),
            backdropFilter: 'blur(20px)',
            borderRight: `1px solid ${alpha(isDark ? '#FFFFFF' : '#000000', 0.1)}`,
          },
        },
      },
    },
  })
}

// Custom CSS animations
export const animations = {
  glow: `
    @keyframes glow {
      0% { box-shadow: 0 0 5px rgba(33, 150, 243, 0.5); }
      50% { box-shadow: 0 0 20px rgba(33, 150, 243, 0.8), 0 0 30px rgba(33, 150, 243, 0.6); }
      100% { box-shadow: 0 0 5px rgba(33, 150, 243, 0.5); }
    }
  `,
  pulse: `
    @keyframes pulse {
      0% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.05); opacity: 0.8; }
      100% { transform: scale(1); opacity: 1; }
    }
  `,
  float: `
    @keyframes float {
      0% { transform: translateY(0px); }
      50% { transform: translateY(-10px); }
      100% { transform: translateY(0px); }
    }
  `,
  shimmer: `
    @keyframes shimmer {
      0% { background-position: -200% 0; }
      100% { background-position: 200% 0; }
    }
  `,
}

// Glassmorphism utilities
export const glass = {
  light: {
    background: alpha('#FFFFFF', 0.7),
    backdropFilter: 'blur(10px)',
    border: `1px solid ${alpha('#FFFFFF', 0.2)}`,
  },
  dark: {
    background: alpha('#121218', 0.7),
    backdropFilter: 'blur(10px)',
    border: `1px solid ${alpha('#FFFFFF', 0.1)}`,
  },
  colored: (color: string, opacity = 0.1) => ({
    background: alpha(color, opacity),
    backdropFilter: 'blur(10px)',
    border: `1px solid ${alpha(color, 0.2)}`,
  }),
}