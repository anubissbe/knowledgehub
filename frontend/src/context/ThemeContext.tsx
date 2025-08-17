import React, { createContext, useContext, useState, useEffect, ReactNode } from "react"
import { ThemeProvider, createTheme, CssBaseline, alpha, responsiveFontSizes } from "@mui/material"

interface ThemeContextType {
  darkMode: boolean
  toggleDarkMode: () => void
  themeMode: "light" | "dark" | "system"
  setThemeMode: (mode: "light" | "dark" | "system") => void
  isMobile: boolean
  isTablet: boolean
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export const useTheme = () => {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider")
  }
  return context
}

interface ThemeProviderProps {
  children: ReactNode
}

export const CustomThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [themeMode, setThemeMode] = useState<"light" | "dark" | "system">(() => {
    if (typeof window === "undefined") return "system"
    const saved = localStorage.getItem("knowledgehub_darkMode")
    if (saved === null) return "system"
    return JSON.parse(saved) ? "dark" : "light"
  })

  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window === "undefined") return false
    if (themeMode === "system") {
      return window.matchMedia("(prefers-color-scheme: dark)").matches
    }
    return themeMode === "dark"
  })

  const [isMobile, setIsMobile] = useState(false)
  const [isTablet, setIsTablet] = useState(false)

  // Responsive detection
  useEffect(() => {
    const updateResponsive = () => {
      const width = window.innerWidth
      setIsMobile(width < 768)
      setIsTablet(width >= 768 && width < 1024)
    }

    updateResponsive()
    window.addEventListener("resize", updateResponsive)
    return () => window.removeEventListener("resize", updateResponsive)
  }, [])

  // Listen for system theme changes
  useEffect(() => {
    if (typeof window === "undefined") return
    
    if (themeMode === "system") {
      const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)")
      const handleChange = (e: MediaQueryListEvent) => {
        setDarkMode(e.matches)
      }
      
      mediaQuery.addEventListener("change", handleChange)
      setDarkMode(mediaQuery.matches)
      
      return () => mediaQuery.removeEventListener("change", handleChange)
    } else {
      setDarkMode(themeMode === "dark")
    }
  }, [themeMode])

  // Update localStorage when theme changes
  useEffect(() => {
    if (typeof window === "undefined") return
    
    if (themeMode === "system") {
      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches
      localStorage.setItem("knowledgehub_darkMode", JSON.stringify(prefersDark))
    } else {
      localStorage.setItem("knowledgehub_darkMode", JSON.stringify(themeMode === "dark"))
    }
  }, [themeMode, darkMode])

  const toggleDarkMode = () => {
    const newMode = darkMode ? "light" : "dark"
    setThemeMode(newMode)
  }

  const handleSetThemeMode = (mode: "light" | "dark" | "system") => {
    setThemeMode(mode)
  }

  // Mobile-first theme with responsive design
  let theme = createTheme({
    // Mobile-first breakpoints
    breakpoints: {
      values: {
        xs: 0,
        sm: 600,
        md: 768,
        lg: 1024,
        xl: 1200,
      },
    },
    
    palette: {
      mode: darkMode ? "dark" : "light",
      primary: {
        main: "#0066FF",
        light: "#64b5f6", 
        dark: "#1976d2",
      },
      secondary: {
        main: "#FF6B9D",
        light: "#ff5983",
        dark: "#c51162",
      },
      success: {
        main: "#00C853",
        light: "#81c784",
        dark: "#388e3c", 
      },
      error: {
        main: "#f44336",
        light: "#e57373",
        dark: "#d32f2f",
      },
      warning: {
        main: "#ff9800",
        light: "#ffb74d",
        dark: "#f57c00",
      },
      info: {
        main: "#00bcd4",
        light: "#4dd0e1",
        dark: "#0097a7",
      },
      background: darkMode
        ? {
            default: "#0a0e1a",
            paper: "#141b2d",
          }
        : {
            default: "#f5f7fa", 
            paper: "#ffffff",
          },
      text: darkMode
        ? {
            primary: "#ffffff",
            secondary: alpha("#ffffff", 0.7),
          }
        : {
            primary: "#1a202c",
            secondary: alpha("#1a202c", 0.6),
          },
    },
    
    typography: {
      fontFamily: "\"Inter\", \"Roboto\", \"Helvetica\", \"Arial\", sans-serif",
      // Responsive typography
      h1: {
        fontSize: "clamp(1.75rem, 5vw, 2.5rem)",
        fontWeight: 700,
        letterSpacing: "-0.02em",
        lineHeight: 1.2,
      },
      h2: {
        fontSize: "clamp(1.5rem, 4vw, 2rem)",
        fontWeight: 600,
        letterSpacing: "-0.01em", 
        lineHeight: 1.3,
      },
      h3: {
        fontSize: "clamp(1.25rem, 3.5vw, 1.75rem)",
        fontWeight: 600,
        letterSpacing: "-0.01em",
        lineHeight: 1.3,
      },
      h4: {
        fontSize: "clamp(1.125rem, 3vw, 1.5rem)",
        fontWeight: 600,
        lineHeight: 1.4,
      },
      h5: {
        fontSize: "clamp(1rem, 2.5vw, 1.25rem)",
        fontWeight: 600,
        lineHeight: 1.4,
      },
      h6: {
        fontSize: "clamp(0.875rem, 2vw, 1rem)",
        fontWeight: 600,
        lineHeight: 1.4,
      },
      body1: {
        fontSize: "1rem",
        lineHeight: 1.6,
      },
      body2: {
        fontSize: "0.875rem", 
        lineHeight: 1.6,
      },
    },

    shape: {
      borderRadius: isMobile ? 8 : 12,
    },

    components: {
      // Mobile-optimized Button
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: "none",
            fontWeight: 600,
            borderRadius: isMobile ? 6 : 8,
            minHeight: isMobile ? 48 : 44, // WCAG touch targets
            padding: isMobile ? "14px 20px" : "8px 20px",
            fontSize: isMobile ? "0.875rem" : "1rem",
            boxShadow: "none",
            transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
            "&:hover": {
              boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
              transform: "translateY(-1px)",
            },
          },
          contained: {
            background: "linear-gradient(135deg, #0066FF 0%, #1976d2 100%)",
            "&:hover": {
              background: "linear-gradient(135deg, #1976d2 0%, #1565c0 100%)",
            },
          },
        },
      },

      // Mobile-optimized IconButton
      MuiIconButton: {
        styleOverrides: {
          root: {
            minWidth: isMobile ? 48 : 44,
            minHeight: isMobile ? 48 : 44,
            padding: isMobile ? 12 : 10,
          },
        },
      },

      // Mobile-optimized Card
      MuiCard: {
        styleOverrides: {
          root: {
            boxShadow: darkMode
              ? "0 4px 24px rgba(0,0,0,0.4)"
              : "0 2px 12px rgba(0,0,0,0.08)",
            borderRadius: isMobile ? 12 : 16,
            border: darkMode 
              ? "1px solid rgba(255,255,255,0.05)" 
              : "1px solid rgba(0,0,0,0.05)",
            transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
            margin: isMobile ? "0 4px" : 0,
            "&:hover": {
              transform: "translateY(-2px)",
              boxShadow: darkMode
                ? "0 8px 32px rgba(0,0,0,0.5)" 
                : "0 4px 20px rgba(0,0,0,0.12)",
            },
          },
        },
      },

      // Mobile-optimized TextField
      MuiTextField: {
        styleOverrides: {
          root: {
            "& .MuiInputBase-root": {
              minHeight: isMobile ? 48 : 44,
              fontSize: isMobile ? "16px" : "1rem", // Prevent zoom on iOS
            },
          },
        },
      },

      // Mobile-optimized AppBar
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: darkMode ? "#141b2d" : "#ffffff",
            color: darkMode ? "#ffffff" : "#1a202c",
            boxShadow: "none",
            borderBottom: darkMode
              ? "1px solid rgba(255,255,255,0.05)"
              : "1px solid rgba(0,0,0,0.05)",
            backdropFilter: "blur(20px)",
            height: isMobile ? 56 : 64,
            "& .MuiToolbar-root": {
              minHeight: isMobile ? 56 : 64,
              padding: isMobile ? "0 8px" : "0 16px",
            },
          },
        },
      },

      // Mobile-optimized Drawer
      MuiDrawer: {
        styleOverrides: {
          paper: {
            backgroundColor: darkMode ? "#0f1419" : "#ffffff",
            borderRight: darkMode
              ? "1px solid rgba(255,255,255,0.05)"
              : "1px solid rgba(0,0,0,0.05)",
            width: isMobile ? "min(280px, 85vw)" : 280,
          },
        },
      },

      // Mobile-optimized ListItemButton
      MuiListItemButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            margin: "2px 8px",
            minHeight: isMobile ? 52 : 48,
            padding: isMobile ? "10px 16px" : "8px 16px",
            "&.Mui-selected": {
              backgroundColor: darkMode
                ? alpha("#0066FF", 0.15)
                : alpha("#0066FF", 0.08),
              "&:hover": {
                backgroundColor: darkMode
                  ? alpha("#0066FF", 0.25)
                  : alpha("#0066FF", 0.12),
              },
            },
          },
        },
      },
    },
  })

  // Apply responsive font sizes
  theme = responsiveFontSizes(theme)

  return (
    <ThemeContext.Provider value={{ 
      darkMode, 
      toggleDarkMode, 
      themeMode, 
      setThemeMode: handleSetThemeMode,
      isMobile,
      isTablet
    }}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </ThemeProvider>
    </ThemeContext.Provider>
  )
}
