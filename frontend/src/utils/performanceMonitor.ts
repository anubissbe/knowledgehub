import { getCLS, getFID, getFCP, getLCP, getTTFB } from "web-vitals"

interface PerformanceMetric {
  name: string
  value: number
  rating: "good" | "needs-improvement" | "poor"
  timestamp: number
}

interface PerformanceReport {
  cls: PerformanceMetric | null
  fid: PerformanceMetric | null
  fcp: PerformanceMetric | null
  lcp: PerformanceMetric | null
  ttfb: PerformanceMetric | null
  customMetrics: Record<string, PerformanceMetric>
  deviceInfo: {
    connection: string
    deviceMemory: number
    hardwareConcurrency: number
    userAgent: string
  }
}

class PerformanceMonitor {
  private metrics: PerformanceReport = {
    cls: null,
    fid: null,
    fcp: null,
    lcp: null,
    ttfb: null,
    customMetrics: {},
    deviceInfo: this.getDeviceInfo(),
  }

  private observers: Map<string, PerformanceObserver> = new Map()

  constructor() {
    this.initWebVitals()
    this.initCustomMetrics()
  }

  private getDeviceInfo() {
    if (typeof window === "undefined") {
      return {
        connection: "unknown",
        deviceMemory: 0,
        hardwareConcurrency: 0,
        userAgent: "server",
      }
    }

    const connection = (navigator as any).connection || (navigator as any).mozConnection || (navigator as any).webkitConnection
    
    return {
      connection: connection?.effectiveType || "unknown",
      deviceMemory: (navigator as any).deviceMemory || 0,
      hardwareConcurrency: navigator.hardwareConcurrency || 0,
      userAgent: navigator.userAgent,
    }
  }

  private initWebVitals() {
    // Cumulative Layout Shift
    getCLS((metric) => {
      this.metrics.cls = {
        name: "CLS",
        value: metric.value,
        rating: metric.value <= 0.1 ? "good" : metric.value <= 0.25 ? "needs-improvement" : "poor",
        timestamp: Date.now(),
      }
      this.reportMetric(this.metrics.cls)
    })

    // First Input Delay
    getFID((metric) => {
      this.metrics.fid = {
        name: "FID", 
        value: metric.value,
        rating: metric.value <= 100 ? "good" : metric.value <= 300 ? "needs-improvement" : "poor",
        timestamp: Date.now(),
      }
      this.reportMetric(this.metrics.fid)
    })

    // First Contentful Paint
    getFCP((metric) => {
      this.metrics.fcp = {
        name: "FCP",
        value: metric.value,
        rating: metric.value <= 1800 ? "good" : metric.value <= 3000 ? "needs-improvement" : "poor",
        timestamp: Date.now(),
      }
      this.reportMetric(this.metrics.fcp)
    })

    // Largest Contentful Paint
    getLCP((metric) => {
      this.metrics.lcp = {
        name: "LCP",
        value: metric.value,
        rating: metric.value <= 2500 ? "good" : metric.value <= 4000 ? "needs-improvement" : "poor", 
        timestamp: Date.now(),
      }
      this.reportMetric(this.metrics.lcp)
    })

    // Time to First Byte
    getTTFB((metric) => {
      this.metrics.ttfb = {
        name: "TTFB",
        value: metric.value,
        rating: metric.value <= 800 ? "good" : metric.value <= 1800 ? "needs-improvement" : "poor",
        timestamp: Date.now(),
      }
      this.reportMetric(this.metrics.ttfb)
    })
  }

  private initCustomMetrics() {
    if (typeof window === "undefined") return

    // Monitor long tasks
    if ("PerformanceObserver" in window) {
      const longTaskObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          this.addCustomMetric("long-task", entry.duration, "poor")
        }
      })

      try {
        longTaskObserver.observe({ entryTypes: ["longtask"] })
        this.observers.set("longtask", longTaskObserver)
      } catch (e) {
      }
    }

    // Monitor navigation timing
    window.addEventListener("load", () => {
      const navigation = performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming
      
      if (navigation) {
        const domContentLoaded = navigation.domContentLoadedEventEnd - navigation.navigationStart
        const loadComplete = navigation.loadEventEnd - navigation.navigationStart
        
        this.addCustomMetric("dom-content-loaded", domContentLoaded, 
          domContentLoaded <= 1500 ? "good" : domContentLoaded <= 3000 ? "needs-improvement" : "poor"
        )
        
        this.addCustomMetric("load-complete", loadComplete,
          loadComplete <= 3000 ? "good" : loadComplete <= 5000 ? "needs-improvement" : "poor"
        )
      }
    })

    // Monitor memory usage (if available)
    if ("memory" in performance) {
      setInterval(() => {
        const memory = (performance as any).memory
        this.addCustomMetric("heap-used", memory.usedJSHeapSize,
          memory.usedJSHeapSize <= 50000000 ? "good" : memory.usedJSHeapSize <= 100000000 ? "needs-improvement" : "poor"
        )
      }, 10000) // Check every 10 seconds
    }
  }

  addCustomMetric(name: string, value: number, rating: "good" | "needs-improvement" | "poor") {
    const metric: PerformanceMetric = {
      name,
      value,
      rating,
      timestamp: Date.now(),
    }

    this.metrics.customMetrics[name] = metric
    this.reportMetric(metric)
  }

  private async reportMetric(metric: PerformanceMetric) {
    // Log to console in development
    if (import.meta.env.DEV) {
    }

    // Send to analytics service
    try {
      await fetch("/api/analytics/performance", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          metric,
          deviceInfo: this.metrics.deviceInfo,
          timestamp: metric.timestamp,
          url: window.location.href,
        }),
      })
    } catch (error) {
    }
  }

  getReport(): PerformanceReport {
    return { ...this.metrics }
  }

  getLighthouseScore(): number {
    const { cls, fid, fcp, lcp, ttfb } = this.metrics
    
    if (!cls || !fid || !fcp || !lcp || !ttfb) {
      return 0
    }

    // Simplified Lighthouse scoring
    const scores = {
      fcp: this.getMetricScore(fcp.rating),
      lcp: this.getMetricScore(lcp.rating),
      cls: this.getMetricScore(cls.rating),
      fid: this.getMetricScore(fid.rating),
      ttfb: this.getMetricScore(ttfb.rating),
    }

    return Math.round(
      (scores.fcp * 0.15 + 
       scores.lcp * 0.25 + 
       scores.cls * 0.15 + 
       scores.fid * 0.25 + 
       scores.ttfb * 0.2) * 100
    )
  }

  private getMetricScore(rating: "good" | "needs-improvement" | "poor"): number {
    switch (rating) {
      case "good": return 1
      case "needs-improvement": return 0.5
      case "poor": return 0
    }
  }

  startResourceMonitoring() {
    if (!("PerformanceObserver" in window)) return

    const resourceObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const resource = entry as PerformanceResourceTiming
        
        // Monitor slow resources
        if (resource.duration > 1000) {
          this.addCustomMetric(`slow-resource-${resource.name.split("/").pop()}`, resource.duration, "poor")
        }
        
        // Monitor large resources
        if (resource.transferSize && resource.transferSize > 1000000) { // 1MB
          this.addCustomMetric(`large-resource-${resource.name.split("/").pop()}`, resource.transferSize, "poor")
        }
      }
    })

    try {
      resourceObserver.observe({ entryTypes: ["resource"] })
      this.observers.set("resource", resourceObserver)
    } catch (e) {
    }
  }

  disconnect() {
    this.observers.forEach((observer) => observer.disconnect())
    this.observers.clear()
  }
}

// Global performance monitor instance
export const performanceMonitor = new PerformanceMonitor()

// Auto-start resource monitoring
if (typeof window !== "undefined") {
  performanceMonitor.startResourceMonitoring()
}

// Export for manual usage
export { PerformanceMonitor }
export type { PerformanceMetric, PerformanceReport }
