// Enhanced Mobile-First Theme Configuration
import { createTheme, ThemeOptions } from '@mui/material/styles'
import { alpha } from '@mui/material'

declare module '@mui/material/styles' {
  interface BreakpointOverrides {
    xs: false // removes the `xs` breakpoint
    sm: false
    md: false
    lg: false
    xl: false
    mobile: true // adds the `mobile` breakpoint
    tablet: true
    laptop: true
    desktop: true
    wide: true
  }

  interface Theme {
    touchTargets: {
      minimum: string
      comfortable: string
      large: string
    }
    mobileSpacing: {
      xs: string
      sm: string
      md: string
      lg: string
      xl: string
    }
  }

  interface ThemeOptions {
    touchTargets?: {
      minimum?: string
      comfortable?: string
      large?: string
    }
    mobileSpacing?: {
      xs?: string
      sm?: string
      md?: string
      lg?: string
      xl?: string
    }
  }
}

export const mobileBreakpoints = {
  mobile: 0, // 0px and up (mobile-first)
  tablet: 768, // 768px and up (tablet)
  laptop: 1024, // 1024px and up (small laptop)
  desktop: 1280, // 1280px and up (desktop)
  wide: 1920, // 1920px and up (large screens)
}

export const createMobileTheme = (darkMode: boolean): ThemeOptions => ({
  breakpoints: {
    values: mobileBreakpoints,
  },

  // Touch target sizes for mobile accessibility
  touchTargets: {
    minimum: '44px', // WCAG minimum
    comfortable: '48px', // Comfortable size
    large: '56px', // Large buttons
  },

  // Mobile-optimized spacing
  mobileSpacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '24px',
    xl: '32px',
  },

  // Typography optimized for mobile readability
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    
    // Headings with mobile scaling
    h1: {
      fontSize: 'clamp(1.75rem, 5vw, 2.5rem)', // Responsive font size
      fontWeight: 700,
      lineHeight: 1.2,
      letterSpacing: '-0.02em',
    },
    h2: {
      fontSize: 'clamp(1.5rem, 4vw, 2rem)',
      fontWeight: 600,
      lineHeight: 1.3,
      letterSpacing: '-0.01em',
    },
    h3: {
      fontSize: 'clamp(1.25rem, 3.5vw, 1.75rem)',
      fontWeight: 600,
      lineHeight: 1.3,
      letterSpacing: '-0.01em',
    },
    h4: {
      fontSize: 'clamp(1.125rem, 3vw, 1.5rem)',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h5: {
      fontSize: 'clamp(1rem, 2.5vw, 1.25rem)',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h6: {
      fontSize: 'clamp(0.875rem, 2vw, 1rem)',
      fontWeight: 600,
      lineHeight: 1.4,
    },

    // Body text with mobile optimization
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
      // Improve readability on mobile
      '@media (max-width: 768px)': {
        fontSize: '0.9375rem',
        lineHeight: 1.5,
      },
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6,
      '@media (max-width: 768px)': {
        fontSize: '0.8125rem',
        lineHeight: 1.5,
      },
    },

    // Button text
    button: {
      fontSize: '0.9375rem',
      fontWeight: 600,
      textTransform: 'none',
      '@media (max-width: 768px)': {
        fontSize: '0.875rem',
      },
    },

    // Caption text
    caption: {
      fontSize: '0.75rem',
      lineHeight: 1.4,
      '@media (max-width: 768px)': {
        fontSize: '0.6875rem',
      },
    },
  },

  // Component overrides for mobile optimization
  components: {
    // Button optimizations
    MuiButton: {
      styleOverrides: {
        root: {
          minHeight: '44px', // WCAG touch target minimum
          padding: '12px 24px',
          borderRadius: '8px',
          textTransform: 'none',
          fontWeight: 600,
          boxShadow: 'none',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          
          // Mobile-specific styling
          '@media (max-width: 768px)': {
            minHeight: '48px', // Larger touch targets on mobile
            padding: '14px 20px',
            fontSize: '0.875rem',
          },

          '&:hover': {
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            transform: 'translateY(-1px)',
          },

          '&:active': {
            transform: 'translateY(0)',
          },
        },
        
        // Size variants
        small: {
          minHeight: '36px',
          padding: '8px 16px',
          '@media (max-width: 768px)': {
            minHeight: '44px', // Still meet touch target on mobile
            padding: '10px 16px',
          },
        },
        
        large: {
          minHeight: '52px',
          padding: '16px 32px',
          '@media (max-width: 768px)': {
            minHeight: '56px',
            padding: '16px 24px',
          },
        },
      },
    },

    // Icon Button optimizations
    MuiIconButton: {
      styleOverrides: {
        root: {
          minWidth: '44px',
          minHeight: '44px',
          padding: '10px',
          
          '@media (max-width: 768px)': {
            minWidth: '48px',
            minHeight: '48px',
            padding: '12px',
          },
        },
      },
    },

    // Card optimizations
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: '16px',
          boxShadow: darkMode
            ? '0 4px 24px rgba(0,0,0,0.4)'
            : '0 2px 12px rgba(0,0,0,0.08)',
          border: darkMode 
            ? '1px solid rgba(255,255,255,0.05)' 
            : '1px solid rgba(0,0,0,0.05)',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          
          // Mobile optimizations
          '@media (max-width: 768px)': {
            borderRadius: '12px',
            margin: '0 4px', // Small margin to prevent edge touching
          },

          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: darkMode
              ? '0 8px 32px rgba(0,0,0,0.5)'
              : '0 4px 20px rgba(0,0,0,0.12)',
          },
        },
      },
    },

    // Input field optimizations
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiInputBase-root': {
            minHeight: '44px',
            
            '@media (max-width: 768px)': {
              minHeight: '48px', // Larger touch targets
              fontSize: '16px', // Prevent zoom on iOS
            },
          },
        },
      },
    },

    // List item optimizations
    MuiListItemButton: {
      styleOverrides: {
        root: {
          minHeight: '48px',
          padding: '8px 16px',
          borderRadius: '8px',
          
          '@media (max-width: 768px)': {
            minHeight: '52px',
            padding: '10px 16px',
          },
        },
      },
    },

    // Chip optimizations
    MuiChip: {
      styleOverrides: {
        root: {
          height: '32px',
          borderRadius: '6px',
          fontSize: '0.875rem',
          
          '@media (max-width: 768px)': {
            height: '36px',
            fontSize: '0.8125rem',
          },
        },
        
        clickable: {
          '&:hover': {
            transform: 'scale(1.05)',
          },
        },
      },
    },

    // Dialog optimizations for mobile
    MuiDialog: {
      styleOverrides: {
        root: {
          '@media (max-width: 768px)': {
            '& .MuiDialog-paper': {
              margin: '16px',
              width: 'calc(100% - 32px)',
              maxWidth: 'calc(100% - 32px)',
              borderRadius: '16px',
            },
          },
        },
      },
    },

    // Drawer optimizations
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: darkMode ? '#0f1419' : '#ffffff',
          borderRight: darkMode
            ? '1px solid rgba(255,255,255,0.05)'
            : '1px solid rgba(0,0,0,0.05)',
          
          // Mobile drawer optimizations
          '@media (max-width: 768px)': {
            width: 'min(280px, 85vw)', // Responsive width
          },
        },
      },
    },

    // App Bar optimizations
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: darkMode ? '#141b2d' : '#ffffff',
          color: darkMode ? '#ffffff' : '#1a202c',
          boxShadow: 'none',
          borderBottom: darkMode
            ? '1px solid rgba(255,255,255,0.05)'
            : '1px solid rgba(0,0,0,0.05)',
          backdropFilter: 'blur(20px)',
          
          // Mobile optimizations
          '@media (max-width: 768px)': {
            height: '56px', // Standard mobile app bar height
            '& .MuiToolbar-root': {
              minHeight: '56px',
            },
          },
        },
      },
    },

    // Toolbar optimizations
    MuiToolbar: {
      styleOverrides: {
        root: {
          padding: '0 16px',
          
          '@media (max-width: 768px)': {
            padding: '0 8px',
          },
        },
      },
    },

    // Tab optimizations
    MuiTab: {
      styleOverrides: {
        root: {
          minHeight: '48px',
          textTransform: 'none',
          
          '@media (max-width: 768px)': {
            minHeight: '52px',
            fontSize: '0.875rem',
          },
        },
      },
    },

    // Snackbar optimizations
    MuiSnackbar: {
      styleOverrides: {
        root: {
          '@media (max-width: 768px)': {
            left: '8px',
            right: '8px',
            bottom: '8px',
            
            '& .MuiSnackbarContent-root': {
              borderRadius: '12px',
            },
          },
        },
      },
    },
  },
})

export default createMobileTheme
EOF < /dev/null
