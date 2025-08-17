import { useEffect, useState } from "react"
import { getCLS, getFID, getFCP, getLCP, getTTFB, Metric } from "web-vitals"

export interface PerformanceMetrics {
  cls: number | null
  fid: number | null  
  fcp: number | null
  lcp: number | null
  ttfb: number | null
  domContentLoaded: number | null
  windowLoaded: number | null
}

export const usePerformanceMonitor = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    cls: null,
    fid: null,
    fcp: null,
    lcp: null,
    ttfb: null,
    domContentLoaded: null,
    windowLoaded: null,
  })

  useEffect(() => {
    // Web Vitals
    getCLS((metric: Metric) => {
      setMetrics(prev => ({ ...prev, cls: metric.value }))
    })

    getFID((metric: Metric) => {
      setMetrics(prev => ({ ...prev, fid: metric.value }))
    })

    getFCP((metric: Metric) => {
      setMetrics(prev => ({ ...prev, fcp: metric.value }))
    })

    getLCP((metric: Metric) => {
      setMetrics(prev => ({ ...prev, lcp: metric.value }))
    })

    getTTFB((metric: Metric) => {
      setMetrics(prev => ({ ...prev, ttfb: metric.value }))
    })

    // Navigation timing
    const measureNavigationTiming = () => {
      if ("performance" in window) {
        const perfData = window.performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming
        
        if (perfData) {
          setMetrics(prev => ({
            ...prev,
            domContentLoaded: perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
            windowLoaded: perfData.loadEventEnd - perfData.loadEventStart,
          }))
        }
      }
    }

    if (document.readyState === "complete") {
      measureNavigationTiming()
    } else {
      window.addEventListener("load", measureNavigationTiming)
      return () => window.removeEventListener("load", measureNavigationTiming)
    }
  }, [])

  return metrics
}
