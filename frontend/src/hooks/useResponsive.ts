import { useTheme, useMediaQuery } from '@mui/material'
import { useState, useEffect } from 'react'

export interface ResponsiveValues {
  isMobile: boolean
  isTablet: boolean
  isDesktop: boolean
  isLargeDesktop: boolean
  screenSize: 'xs' | 'sm' | 'md' | 'lg' | 'xl'
  orientation: 'portrait' | 'landscape'
  width: number
  height: number
}

export function useResponsive(): ResponsiveValues {
  const theme = useTheme()
  
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))
  const isTablet = useMediaQuery(theme.breakpoints.between('md', 'lg'))
  const isDesktop = useMediaQuery(theme.breakpoints.between('lg', 'xl'))
  const isLargeDesktop = useMediaQuery(theme.breakpoints.up('xl'))
  
  const isXs = useMediaQuery(theme.breakpoints.only('xs'))
  const isSm = useMediaQuery(theme.breakpoints.only('sm'))
  const isMd = useMediaQuery(theme.breakpoints.only('md'))
  const isLg = useMediaQuery(theme.breakpoints.only('lg'))
  const isXl = useMediaQuery(theme.breakpoints.up('xl'))

  const [windowSize, setWindowSize] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 0,
    height: typeof window !== 'undefined' ? window.innerHeight : 0,
  })

  useEffect(() => {
    if (typeof window === 'undefined') return

    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      })
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  const getScreenSize = (): 'xs' | 'sm' | 'md' | 'lg' | 'xl' => {
    if (isXs) return 'xs'
    if (isSm) return 'sm'
    if (isMd) return 'md'
    if (isLg) return 'lg'
    return 'xl'
  }

  const getOrientation = (): 'portrait' | 'landscape' => {
    return windowSize.width > windowSize.height ? 'landscape' : 'portrait'
  }

  return {
    isMobile,
    isTablet,
    isDesktop,
    isLargeDesktop,
    screenSize: getScreenSize(),
    orientation: getOrientation(),
    width: windowSize.width,
    height: windowSize.height,
  }
}

export default useResponsive