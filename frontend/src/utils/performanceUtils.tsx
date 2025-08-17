import React from "react";
// Performance optimization utilities for KnowledgeHub Phase 5
// Handles lazy loading, code splitting, caching, and performance monitoring

interface PerformanceMetrics {
  loadTime: number
  renderTime: number
  interactionTime: number
  memoryUsage: number
  cacheHitRate: number
  networkLatency: number
}

interface LazyLoadOptions {
  threshold?: number
  rootMargin?: string
  triggerOnce?: boolean
  fallback?: () => void
}

class PerformanceManager {
  private metrics: Map<string, number> = new Map()
  private observers: Map<string, IntersectionObserver> = new Map()
  private performanceEntries: PerformanceEntry[] = []

  constructor() {
    this.initializePerformanceMonitoring()
  }

  // Performance Monitoring
  private initializePerformanceMonitoring(): void {
    if (typeof window === 'undefined') return

    // Monitor navigation timing
    window.addEventListener('load', () => {
      setTimeout(() => {
        this.collectNavigationMetrics()
      }, 0)
    })

    // Monitor performance entries
    if ('PerformanceObserver' in window) {
      const observer = new PerformanceObserver((list) => {
        this.performanceEntries.push(...list.getEntries())
      })

      observer.observe({ entryTypes: ['navigation', 'resource', 'measure', 'mark'] })
    }

    // Monitor memory usage
    this.startMemoryMonitoring()
  }

  private collectNavigationMetrics(): void {
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
    
    if (navigation) {
      this.metrics.set('loadTime', navigation.loadEventEnd - navigation.loadEventStart)
      this.metrics.set('domContentLoaded', navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart)
      this.metrics.set('firstPaint', this.getFirstPaint())
      this.metrics.set('firstContentfulPaint', this.getFirstContentfulPaint())
      this.metrics.set('timeToInteractive', this.getTTI())
    }
  }

  private getFirstPaint(): number {
    const paintEntries = performance.getEntriesByType('paint')
    const firstPaint = paintEntries.find(entry => entry.name === 'first-paint')
    return firstPaint?.startTime || 0
  }

  private getFirstContentfulPaint(): number {
    const paintEntries = performance.getEntriesByType('paint')
    const fcp = paintEntries.find(entry => entry.name === 'first-contentful-paint')
    return fcp?.startTime || 0
  }

  private getTTI(): number {
    // Simplified TTI calculation
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
    return navigation ? navigation.domInteractive - navigation.navigationStart : 0
  }

  private startMemoryMonitoring(): void {
    if ('memory' in performance) {
      setInterval(() => {
        const memory = (performance as any).memory
        this.metrics.set('memoryUsed', memory.usedJSHeapSize)
        this.metrics.set('memoryTotal', memory.totalJSHeapSize)
        this.metrics.set('memoryLimit', memory.jsHeapSizeLimit)
      }, 30000) // Check every 30 seconds
    }
  }

  // Lazy Loading Utilities
  createLazyLoader<T>(
    importFn: () => Promise<{ default: T }>,
    options: LazyLoadOptions = {}
  ): {
    component: T | null
    loading: boolean
    error: Error | null
    load: () => Promise<void>
  } {
    let component: T | null = null
    let loading = false
    let error: Error | null = null

    const load = async (): Promise<void> => {
      if (component || loading) return

      try {
        loading = true
        const startTime = performance.now()
        
        const module = await importFn()
        component = module.default
        
        const loadTime = performance.now() - startTime
        this.metrics.set(`lazy-load-${module.default.name || 'component'}`, loadTime)
        
        loading = false
      } catch (err) {
        error = err as Error
        loading = false
        options.fallback?.()
      }
    }

    return { component, loading, error, load }
  }

  createIntersectionObserver(
    callback: (entries: IntersectionObserverEntry[]) => void,
    options: LazyLoadOptions = {}
  ): IntersectionObserver | null {
    if (typeof window === 'undefined' || !('IntersectionObserver' in window)) {
      return null
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const visibleEntries = entries.filter(entry => entry.isIntersecting)
        if (visibleEntries.length > 0) {
          callback(visibleEntries)
          
          if (options.triggerOnce) {
            visibleEntries.forEach(entry => observer.unobserve(entry.target))
          }
        }
      },
      {
        threshold: options.threshold || 0.1,
        rootMargin: options.rootMargin || '50px',
      }
    )

    return observer
  }

  // Image Lazy Loading
  lazyLoadImages(selector = 'img[data-src]'): void {
    const images = document.querySelectorAll(selector)
    
    const observer = this.createIntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          const img = entry.target as HTMLImageElement
          if (img.dataset.src) {
            img.src = img.dataset.src
            img.removeAttribute('data-src')
            img.classList.add('loaded')
          }
        })
      },
      { triggerOnce: true }
    )

    if (observer) {
      images.forEach(img => observer.observe(img))
    }
  }

  // Code Splitting Utilities
  async preloadRoute(routePath: string): Promise<void> {
    try {
      const startTime = performance.now()
      
      // This would be configured based on your route splitting strategy
      const routeMap: Record<string, () => Promise<any>> = {
        '/dashboard': () => import('../pages/ModernDashboardV2'),
        '/ai': () => import('../pages/AiIntelligenceModern'),
        '/memory': () => import('../pages/MemorySystemModern'),
        '/search': () => import('../pages/SearchKnowledgeModern'),
        '/sources': () => import('../pages/SourcesModern'),
        '/settings': () => import('../pages/SettingsModern'),
      }

      const loadRoute = routeMap[routePath]
      if (loadRoute) {
        await loadRoute()
        const loadTime = performance.now() - startTime
        this.metrics.set(`preload-${routePath}`, loadTime)
      }
    } catch (error) {
    }
  }

  preloadCriticalRoutes(): void {
    // Preload likely next routes based on current page
    const currentPath = window.location.pathname
    const criticalRoutes: Record<string, string[]> = {
      '/dashboard': ['/ai', '/memory'],
      '/ai': ['/dashboard', '/analytics'],
      '/memory': ['/search', '/dashboard'],
      '/search': ['/memory', '/sources'],
    }

    const routesToPreload = criticalRoutes[currentPath] || []
    routesToPreload.forEach(route => {
      // Use requestIdleCallback if available, otherwise setTimeout
      if ('requestIdleCallback' in window) {
        (window as any).requestIdleCallback(() => this.preloadRoute(route))
      } else {
        setTimeout(() => this.preloadRoute(route), 100)
      }
    })
  }

  // Bundle Analysis
  analyzeBundlePerformance(): {
    totalSize: number
    chunkSizes: Record<string, number>
    unusedCode: string[]
  } {
    const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[]
    const jsResources = resources.filter(r => r.name.endsWith('.js'))
    
    const totalSize = jsResources.reduce((sum, resource) => sum + (resource.transferSize || 0), 0)
    const chunkSizes: Record<string, number> = {}
    
    jsResources.forEach(resource => {
      const name = resource.name.split('/').pop() || 'unknown'
      chunkSizes[name] = resource.transferSize || 0
    })

    return {
      totalSize,
      chunkSizes,
      unusedCode: [], // Would require coverage API in real implementation
    }
  }

  // Performance Budgets
  checkPerformanceBudgets(): {
    loadTime: { budget: number; actual: number; passed: boolean }
    firstContentfulPaint: { budget: number; actual: number; passed: boolean }
    timeToInteractive: { budget: number; actual: number; passed: boolean }
    bundleSize: { budget: number; actual: number; passed: boolean }
  } {
    const budgets = {
      loadTime: 3000, // 3 seconds
      firstContentfulPaint: 1500, // 1.5 seconds
      timeToInteractive: 3500, // 3.5 seconds
      bundleSize: 500000, // 500KB
    }

    const results = {
      loadTime: {
        budget: budgets.loadTime,
        actual: this.metrics.get('loadTime') || 0,
        passed: (this.metrics.get('loadTime') || 0) < budgets.loadTime,
      },
      firstContentfulPaint: {
        budget: budgets.firstContentfulPaint,
        actual: this.metrics.get('firstContentfulPaint') || 0,
        passed: (this.metrics.get('firstContentfulPaint') || 0) < budgets.firstContentfulPaint,
      },
      timeToInteractive: {
        budget: budgets.timeToInteractive,
        actual: this.metrics.get('timeToInteractive') || 0,
        passed: (this.metrics.get('timeToInteractive') || 0) < budgets.timeToInteractive,
      },
      bundleSize: {
        budget: budgets.bundleSize,
        actual: this.analyzeBundlePerformance().totalSize,
        passed: this.analyzeBundlePerformance().totalSize < budgets.bundleSize,
      },
    }

    return results
  }

  // Cache Performance
  getCachePerformance(): {
    hitRate: number
    missCount: number
    totalRequests: number
  } {
    const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[]
    const cachedResources = resources.filter(r => r.transferSize === 0 && r.decodedBodySize > 0)
    
    return {
      hitRate: resources.length > 0 ? (cachedResources.length / resources.length) * 100 : 0,
      missCount: resources.length - cachedResources.length,
      totalRequests: resources.length,
    }
  }

  // Resource Hints
  addResourceHints(resources: Array<{
    url: string
    type: 'preload' | 'prefetch' | 'dns-prefetch' | 'preconnect'
    as?: string
    crossorigin?: boolean
  }>): void {
    if (typeof document === 'undefined') return

    resources.forEach(({ url, type, as, crossorigin }) => {
      const link = document.createElement('link')
      link.rel = type
      link.href = url
      
      if (as) link.setAttribute('as', as)
      if (crossorigin) link.crossOrigin = 'anonymous'
      
      document.head.appendChild(link)
    })
  }

  // Performance Reporting
  getPerformanceReport(): {
    metrics: Record<string, number>
    budgets: ReturnType<typeof this.checkPerformanceBudgets>
    cache: ReturnType<typeof this.getCachePerformance>
    bundle: ReturnType<typeof this.analyzeBundlePerformance>
    timestamp: string
  } {
    return {
      metrics: Object.fromEntries(this.metrics.entries()),
      budgets: this.checkPerformanceBudgets(),
      cache: this.getCachePerformance(),
      bundle: this.analyzeBundlePerformance(),
      timestamp: new Date().toISOString(),
    }
  }

  // Send performance data to analytics
  reportPerformance(): void {
    const report = this.getPerformanceReport()
    
    // Send to analytics service (in real implementation)
    
    // Send to API for tracking
    fetch('/api/analytics/performance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(report),
    }).catch(error => {
    })
  }

  // Cleanup
  cleanup(): void {
    this.observers.forEach(observer => observer.disconnect())
    this.observers.clear()
    this.metrics.clear()
  }
}

// Create singleton instance
export const performanceManager = new PerformanceManager()

// Utility functions
export const measurePerformance = (name: string, fn: () => any) => {
  const start = performance.now()
  const result = fn()
  const end = performance.now()
  
  return result
}

export const measureAsyncPerformance = async (name: string, fn: () => Promise<any>) => {
  const start = performance.now()
  const result = await fn()
  const end = performance.now()
  
  return result
}

export const createLazyComponent = <T extends React.ComponentType<any>>(
  importFn: () => Promise<{ default: T }>
): React.ComponentType<React.ComponentProps<T>> => {
  const LazyComponent = React.lazy(importFn)
  
  return React.forwardRef<any, React.ComponentProps<T>>((props, ref) => (
    <React.Suspense fallback={<div>Loading...</div>}>
      <LazyComponent {...props} ref={ref} />
    </React.Suspense>
  ))
}

export const preloadImage = (src: string): Promise<void> => {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve()
    img.onerror = reject
    img.src = src
  })
}

export const debounce = <T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void => {
  let timeoutId: NodeJS.Timeout
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => fn(...args), delay)
  }
}

export const throttle = <T extends (...args: any[]) => any>(
  fn: T,
  limit: number
): (...args: Parameters<T>) => void => {
  let inThrottle: boolean
  
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      fn(...args)
      inThrottle = true
      setTimeout(() => inThrottle = false, limit)
    }
  }
}

export default performanceManager