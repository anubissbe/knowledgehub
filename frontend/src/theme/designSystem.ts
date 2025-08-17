// Modern Design System for KnowledgeHub Phase 5
// Production-ready design tokens and theme configuration

import { createTheme, ThemeOptions } from '@mui/material/styles'
import { alpha, darken, lighten } from '@mui/material/styles'

// Design Tokens
export const designTokens = {
  // Color Palette - Modern, high-contrast
  colors: {
    primary: {
      50: '#E3F2FD',
      100: '#BBDEFB', 
      200: '#90CAF9',
      300: '#64B5F6',
      400: '#42A5F5',
      500: '#2196F3', // Main brand color
      600: '#1E88E5',
      700: '#1976D2',
      800: '#1565C0',
      900: '#0D47A1',
    },
    secondary: {
      50: '#FCE4EC',
      100: '#F8BBD9',
      200: '#F48FB1',
      300: '#F06292',
      400: '#EC407A',
      500: '#E91E63',
      600: '#D81B60',
      700: '#C2185B',
      800: '#AD1457',
      900: '#880E4F',
    },
    accent: {
      cyan: '#00FFFF',
      green: '#00FF88',
      purple: '#8B5CF6',
      orange: '#FF6B35',
      yellow: '#FFD700',
    },
    semantic: {
      success: '#00FF88',
      warning: '#FFB020',
      error: '#FF3366',
      info: '#00BFFF',
    },
    neutral: {
      0: '#FFFFFF',
      50: '#FAFAFA',
      100: '#F5F5F5',
      200: '#EEEEEE',
      300: '#E0E0E0',
      400: '#BDBDBD',
      500: '#9E9E9E',
      600: '#757575',
      700: '#616161',
      800: '#424242',
      900: '#212121',
      1000: '#000000',
    },
  },

  // Typography Scale
  typography: {
    fontFamily: {
      primary: '"Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      mono: '"JetBrains Mono", "Fira Code", "SF Mono", Consolas, monospace',
    },
    fontSize: {
      xs: '0.75rem',    // 12px
      sm: '0.875rem',   // 14px
      base: '1rem',     // 16px
      lg: '1.125rem',   // 18px
      xl: '1.25rem',    // 20px
      '2xl': '1.5rem',  // 24px
      '3xl': '1.875rem',// 30px
      '4xl': '2.25rem', // 36px
      '5xl': '3rem',    // 48px
      '6xl': '4rem',    // 64px
    },
    fontWeight: {
      light: 300,
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
      extrabold: 800,
    },
    lineHeight: {
      tight: 1.2,
      normal: 1.5,
      relaxed: 1.75,
    },
  },

  // Spacing Scale (8px base unit)
  spacing: {
    0: '0',
    1: '0.25rem', // 4px
    2: '0.5rem',  // 8px
    3: '0.75rem', // 12px
    4: '1rem',    // 16px
    5: '1.25rem', // 20px
    6: '1.5rem',  // 24px
    8: '2rem',    // 32px
    10: '2.5rem', // 40px
    12: '3rem',   // 48px
    16: '4rem',   // 64px
    20: '5rem',   // 80px
    24: '6rem',   // 96px
    32: '8rem',   // 128px
    40: '10rem',  // 160px
    48: '12rem',  // 192px
    56: '14rem',  // 224px
    64: '16rem',  // 256px
  },

  // Border Radius
  borderRadius: {
    none: '0',
    sm: '0.25rem',  // 4px
    base: '0.5rem', // 8px
    md: '0.75rem',  // 12px
    lg: '1rem',     // 16px
    xl: '1.5rem',   // 24px
    full: '9999px',
  },

  // Shadows
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    base: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
    inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
    glass: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
    glow: '0 0 20px rgba(33, 150, 243, 0.3)',
  },

  // Transitions
  transitions: {
    fast: '150ms cubic-bezier(0.4, 0, 0.2, 1)',
    normal: '300ms cubic-bezier(0.4, 0, 0.2, 1)',
    slow: '500ms cubic-bezier(0.4, 0, 0.2, 1)',
    bounce: '600ms cubic-bezier(0.68, -0.55, 0.265, 1.55)',
  },

  // Z-Index Scale
  zIndex: {
    hide: -1,
    auto: 'auto',
    base: 0,
    docked: 10,
    dropdown: 1000,
    sticky: 1100,
    banner: 1200,
    overlay: 1300,
    modal: 1400,
    popover: 1500,
    skipLink: 1600,
    toast: 1700,
    tooltip: 1800,
  },

  // Glassmorphism Effects
  glass: {
    light: {
      background: 'rgba(255, 255, 255, 0.25)',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(255, 255, 255, 0.18)',
    },
    dark: {
      background: 'rgba(255, 255, 255, 0.05)',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(255, 255, 255, 0.1)',
    },
    colored: {
      background: 'rgba(33, 150, 243, 0.1)',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(33, 150, 243, 0.2)',
    },
  },
}

// Component Variants
export const componentVariants = {
  // Button variants
  button: {
    primary: {
      background: designTokens.colors.primary[500],
      color: designTokens.colors.neutral[0],
      shadow: designTokens.shadows.md,
      hover: {
        background: designTokens.colors.primary[600],
        transform: 'translateY(-2px)',
      },
    },
    glass: {
      background: designTokens.glass.light.background,
      backdropFilter: designTokens.glass.light.backdropFilter,
      border: designTokens.glass.light.border,
      color: designTokens.colors.neutral[900],
      hover: {
        background: 'rgba(255, 255, 255, 0.35)',
      },
    },
    neon: {
      background: 'transparent',
      border: `2px solid ${designTokens.colors.accent.cyan}`,
      color: designTokens.colors.accent.cyan,
      boxShadow: `0 0 10px ${designTokens.colors.accent.cyan}`,
      hover: {
        background: designTokens.colors.accent.cyan,
        color: designTokens.colors.neutral[900],
        boxShadow: `0 0 20px ${designTokens.colors.accent.cyan}`,
      },
    },
  },

  // Card variants
  card: {
    default: {
      background: designTokens.colors.neutral[0],
      border: `1px solid ${designTokens.colors.neutral[200]}`,
      borderRadius: designTokens.borderRadius.lg,
      shadow: designTokens.shadows.base,
    },
    glass: {
      background: designTokens.glass.light.background,
      backdropFilter: designTokens.glass.light.backdropFilter,
      border: designTokens.glass.light.border,
      borderRadius: designTokens.borderRadius.xl,
      shadow: designTokens.shadows.glass,
    },
    elevated: {
      background: designTokens.colors.neutral[0],
      borderRadius: designTokens.borderRadius.xl,
      shadow: designTokens.shadows.xl,
      border: 'none',
    },
  },
}

// Create Material-UI theme with design system
export const createDesignSystemTheme = (mode: 'light' | 'dark' = 'dark'): ThemeOptions => {
  const isDark = mode === 'dark'

  return createTheme({
    palette: {
      mode,
      primary: {
        main: designTokens.colors.primary[500],
        light: designTokens.colors.primary[300],
        dark: designTokens.colors.primary[700],
        contrastText: designTokens.colors.neutral[0],
      },
      secondary: {
        main: designTokens.colors.secondary[500],
        light: designTokens.colors.secondary[300],
        dark: designTokens.colors.secondary[700],
        contrastText: designTokens.colors.neutral[0],
      },
      success: {
        main: designTokens.colors.semantic.success,
        light: lighten(designTokens.colors.semantic.success, 0.2),
        dark: darken(designTokens.colors.semantic.success, 0.2),
      },
      warning: {
        main: designTokens.colors.semantic.warning,
        light: lighten(designTokens.colors.semantic.warning, 0.2),
        dark: darken(designTokens.colors.semantic.warning, 0.2),
      },
      error: {
        main: designTokens.colors.semantic.error,
        light: lighten(designTokens.colors.semantic.error, 0.2),
        dark: darken(designTokens.colors.semantic.error, 0.2),
      },
      info: {
        main: designTokens.colors.semantic.info,
        light: lighten(designTokens.colors.semantic.info, 0.2),
        dark: darken(designTokens.colors.semantic.info, 0.2),
      },
      background: {
        default: isDark ? designTokens.colors.neutral[900] : designTokens.colors.neutral[50],
        paper: isDark ? designTokens.colors.neutral[800] : designTokens.colors.neutral[0],
      },
      text: {
        primary: isDark ? designTokens.colors.neutral[100] : designTokens.colors.neutral[900],
        secondary: isDark ? designTokens.colors.neutral[400] : designTokens.colors.neutral[600],
        disabled: isDark ? designTokens.colors.neutral[600] : designTokens.colors.neutral[400],
      },
    },

    typography: {
      fontFamily: designTokens.typography.fontFamily.primary,
      h1: {
        fontSize: designTokens.typography.fontSize['5xl'],
        fontWeight: designTokens.typography.fontWeight.bold,
        lineHeight: designTokens.typography.lineHeight.tight,
        letterSpacing: '-0.025em',
      },
      h2: {
        fontSize: designTokens.typography.fontSize['4xl'],
        fontWeight: designTokens.typography.fontWeight.bold,
        lineHeight: designTokens.typography.lineHeight.tight,
        letterSpacing: '-0.025em',
      },
      h3: {
        fontSize: designTokens.typography.fontSize['3xl'],
        fontWeight: designTokens.typography.fontWeight.semibold,
        lineHeight: designTokens.typography.lineHeight.tight,
      },
      h4: {
        fontSize: designTokens.typography.fontSize['2xl'],
        fontWeight: designTokens.typography.fontWeight.semibold,
        lineHeight: designTokens.typography.lineHeight.normal,
      },
      h5: {
        fontSize: designTokens.typography.fontSize.xl,
        fontWeight: designTokens.typography.fontWeight.medium,
        lineHeight: designTokens.typography.lineHeight.normal,
      },
      h6: {
        fontSize: designTokens.typography.fontSize.lg,
        fontWeight: designTokens.typography.fontWeight.medium,
        lineHeight: designTokens.typography.lineHeight.normal,
      },
      body1: {
        fontSize: designTokens.typography.fontSize.base,
        lineHeight: designTokens.typography.lineHeight.normal,
      },
      body2: {
        fontSize: designTokens.typography.fontSize.sm,
        lineHeight: designTokens.typography.lineHeight.normal,
      },
      caption: {
        fontSize: designTokens.typography.fontSize.xs,
        lineHeight: designTokens.typography.lineHeight.normal,
      },
    },

    shape: {
      borderRadius: 12, // Base border radius in px
    },

    spacing: 8, // Base spacing unit

    shadows: [
      'none',
      designTokens.shadows.sm,
      designTokens.shadows.base,
      designTokens.shadows.md,
      designTokens.shadows.lg,
      designTokens.shadows.xl,
      designTokens.shadows['2xl'],
      designTokens.shadows.glass,
      designTokens.shadows.glow,
      // Add more shadows as needed for MUI's 25 shadow levels
      ...Array(16).fill(designTokens.shadows.xl),
    ],

    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: designTokens.borderRadius.lg,
            textTransform: 'none',
            fontWeight: designTokens.typography.fontWeight.semibold,
            padding: `${designTokens.spacing[3]} ${designTokens.spacing[6]}`,
            transition: designTokens.transitions.normal,
            '&:hover': {
              transform: 'translateY(-1px)',
            },
          },
          containedPrimary: {
            background: `linear-gradient(45deg, ${designTokens.colors.primary[500]}, ${designTokens.colors.primary[400]})`,
            boxShadow: designTokens.shadows.md,
            '&:hover': {
              background: `linear-gradient(45deg, ${designTokens.colors.primary[600]}, ${designTokens.colors.primary[500]})`,
              boxShadow: designTokens.shadows.lg,
            },
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: designTokens.borderRadius.xl,
            backdropFilter: isDark ? 'blur(10px)' : 'none',
            background: isDark 
              ? 'rgba(255, 255, 255, 0.05)'
              : designTokens.colors.neutral[0],
            border: isDark 
              ? '1px solid rgba(255, 255, 255, 0.1)'
              : `1px solid ${designTokens.colors.neutral[200]}`,
            boxShadow: designTokens.shadows.lg,
            transition: designTokens.transitions.normal,
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: designTokens.shadows.xl,
            },
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: designTokens.borderRadius.full,
            fontWeight: designTokens.typography.fontWeight.medium,
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            background: isDark
              ? 'rgba(33, 33, 33, 0.95)'
              : 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(10px)',
            borderBottom: isDark
              ? '1px solid rgba(255, 255, 255, 0.1)'
              : `1px solid ${designTokens.colors.neutral[200]}`,
            boxShadow: designTokens.shadows.sm,
          },
        },
      },
    },

    breakpoints: {
      values: {
        xs: 0,
        sm: 640,
        md: 768,
        lg: 1024,
        xl: 1280,
      },
    },
  })
}

export const lightTheme = createDesignSystemTheme('light')
export const darkTheme = createDesignSystemTheme('dark')
export default darkTheme