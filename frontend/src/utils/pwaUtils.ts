// Progressive Web App utilities for KnowledgeHub
// Handles installation prompts, service worker management, and offline capabilities

interface BeforeInstallPromptEvent extends Event {
  readonly platforms: string[]
  readonly userChoice: Promise<{
    outcome: 'accepted' | 'dismissed'
    platform: string
  }>
  prompt(): Promise<void>
}

interface PWAInstallationState {
  canInstall: boolean
  isInstalled: boolean
  isStandalone: boolean
  deferredPrompt: BeforeInstallPromptEvent | null
}

class PWAManager {
  private deferredPrompt: BeforeInstallPromptEvent | null = null
  private installCallbacks: Set<(canInstall: boolean) => void> = new Set()
  private updateCallbacks: Set<(updateAvailable: boolean) => void> = new Set()
  
  constructor() {
    this.setupInstallPrompt()
    this.setupServiceWorker()
  }

  // Installation Management
  private setupInstallPrompt() {
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault()
      this.deferredPrompt = e as BeforeInstallPromptEvent
      this.notifyInstallCallbacks(true)
    })

    window.addEventListener('appinstalled', () => {
      this.deferredPrompt = null
      this.notifyInstallCallbacks(false)
    })
  }

  async showInstallPrompt(): Promise<boolean> {
    if (!this.deferredPrompt) {
      return false
    }

    try {
      await this.deferredPrompt.prompt()
      const { outcome } = await this.deferredPrompt.userChoice
      
      
      if (outcome === 'accepted') {
        this.deferredPrompt = null
        this.notifyInstallCallbacks(false)
        return true
      }
      
      return false
    } catch (error) {
      return false
    }
  }

  getInstallationState(): PWAInstallationState {
    return {
      canInstall: !!this.deferredPrompt,
      isInstalled: this.isAppInstalled(),
      isStandalone: this.isStandaloneMode(),
      deferredPrompt: this.deferredPrompt,
    }
  }

  private isAppInstalled(): boolean {
    // Check for installed PWA indicators
    return (
      window.navigator.standalone === true || // iOS
      window.matchMedia('(display-mode: standalone)').matches || // Android
      window.matchMedia('(display-mode: minimal-ui)').matches ||
      document.referrer.includes('android-app://') ||
      window.location.search.includes('utm_source=pwa')
    )
  }

  private isStandaloneMode(): boolean {
    return (
      window.navigator.standalone === true ||
      window.matchMedia('(display-mode: standalone)').matches
    )
  }

  onInstallStateChange(callback: (canInstall: boolean) => void): () => void {
    this.installCallbacks.add(callback)
    
    // Call immediately with current state
    callback(!!this.deferredPrompt)
    
    return () => {
      this.installCallbacks.delete(callback)
    }
  }

  private notifyInstallCallbacks(canInstall: boolean) {
    this.installCallbacks.forEach(callback => callback(canInstall))
  }

  // Service Worker Management
  private async setupServiceWorker() {
    if ('serviceWorker' in navigator) {
      try {
        const registration = await navigator.serviceWorker.register('/sw.js')

        // Handle updates
        registration.addEventListener('updatefound', () => {
          const newWorker = registration.installing
          if (newWorker) {
            newWorker.addEventListener('statechange', () => {
              if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                this.notifyUpdateCallbacks(true)
              }
            })
          }
        })

        // Handle messages from service worker
        navigator.serviceWorker.addEventListener('message', (event) => {
          if (event.data && event.data.type === 'CACHE_UPDATED') {
          }
        })

      } catch (error) {
      }
    }
  }

  async updateServiceWorker(): Promise<void> {
    if ('serviceWorker' in navigator) {
      const registration = await navigator.serviceWorker.getRegistration()
      if (registration) {
        await registration.update()
        window.location.reload()
      }
    }
  }

  onUpdateAvailable(callback: (updateAvailable: boolean) => void): () => void {
    this.updateCallbacks.add(callback)
    return () => {
      this.updateCallbacks.delete(callback)
    }
  }

  private notifyUpdateCallbacks(updateAvailable: boolean) {
    this.updateCallbacks.forEach(callback => callback(updateAvailable))
  }

  // Offline Support
  getOnlineStatus(): boolean {
    return navigator.onLine
  }

  onOnlineStatusChange(callback: (online: boolean) => void): () => void {
    const handleOnline = () => callback(true)
    const handleOffline = () => callback(false)

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    // Call immediately with current status
    callback(navigator.onLine)

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }

  // Device Capabilities
  getDeviceCapabilities() {
    return {
      // Network
      connection: (navigator as any).connection || (navigator as any).mozConnection || (navigator as any).webkitConnection,
      onLine: navigator.onLine,

      // Storage
      storage: 'storage' in navigator,
      storageEstimate: 'estimate' in navigator.storage,

      // Notifications
      notifications: 'Notification' in window,
      pushMessaging: 'PushManager' in window,

      // Hardware
      deviceMemory: (navigator as any).deviceMemory,
      hardwareConcurrency: navigator.hardwareConcurrency,

      // Sensors
      geolocation: 'geolocation' in navigator,
      vibration: 'vibrate' in navigator,

      // Web APIs
      share: 'share' in navigator,
      clipboard: 'clipboard' in navigator,
      wakeLock: 'wakeLock' in navigator,
      
      // Display
      fullscreen: 'requestFullscreen' in document.documentElement,
      screenOrientation: 'screen' in window && 'orientation' in window.screen,
    }
  }

  // Storage Management
  async getStorageUsage(): Promise<{
    used: number
    quota: number
    percentage: number
  }> {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      try {
        const estimate = await navigator.storage.estimate()
        const used = estimate.usage || 0
        const quota = estimate.quota || 0
        const percentage = quota > 0 ? (used / quota) * 100 : 0

        return { used, quota, percentage }
      } catch (error) {
      }
    }

    return { used: 0, quota: 0, percentage: 0 }
  }

  async requestPersistentStorage(): Promise<boolean> {
    if ('storage' in navigator && 'persist' in navigator.storage) {
      try {
        const persistent = await navigator.storage.persist()
        return persistent
      } catch (error) {
      }
    }
    return false
  }

  // Notifications
  async requestNotificationPermission(): Promise<NotificationPermission> {
    if ('Notification' in window) {
      try {
        const permission = await Notification.requestPermission()
        return permission
      } catch (error) {
        return 'denied'
      }
    }
    return 'denied'
  }

  async showNotification(title: string, options?: NotificationOptions): Promise<void> {
    if ('Notification' in window && Notification.permission === 'granted') {
      try {
        if ('serviceWorker' in navigator) {
          const registration = await navigator.serviceWorker.getRegistration()
          if (registration) {
            await registration.showNotification(title, {
              badge: '/icons/icon-96x96.png',
              icon: '/icons/icon-192x192.png',
              ...options,
            })
            return
          }
        }
        
        // Fallback to regular notification
        new Notification(title, options)
      } catch (error) {
      }
    }
  }

  // Share API
  async shareContent(data: ShareData): Promise<boolean> {
    if ('share' in navigator) {
      try {
        await navigator.share(data)
        return true
      } catch (error) {
        if (error instanceof Error && error.name !== 'AbortError') {
        }
        return false
      }
    }
    return false
  }

  // Wake Lock
  async requestWakeLock(): Promise<WakeLockSentinel | null> {
    if ('wakeLock' in navigator) {
      try {
        const wakeLock = await navigator.wakeLock.request('screen')
        return wakeLock
      } catch (error) {
      }
    }
    return null
  }
}

// Create singleton instance
export const pwaManager = new PWAManager()

// Utility functions
export const isPWACapable = (): boolean => {
  return (
    'serviceWorker' in navigator &&
    'Cache' in window &&
    'caches' in window &&
    'indexedDB' in window
  )
}

export const getBrowserName = (): string => {
  const userAgent = navigator.userAgent.toLowerCase()
  
  if (userAgent.includes('chrome')) return 'chrome'
  if (userAgent.includes('firefox')) return 'firefox'
  if (userAgent.includes('safari')) return 'safari'
  if (userAgent.includes('edge')) return 'edge'
  if (userAgent.includes('opera')) return 'opera'
  
  return 'unknown'
}

export const getInstallInstructions = (): { browser: string; instructions: string[] } => {
  const browser = getBrowserName()
  
  const instructions: Record<string, string[]> = {
    chrome: [
      'Click the three dots menu (⋮) in the top right corner',
      'Select "Install KnowledgeHub" or "Add to Home screen"',
      'Click "Install" in the dialog that appears',
    ],
    firefox: [
      'Click the address bar',
      'Look for the "Install" button or house icon',
      'Click "Install" to add to your home screen',
    ],
    safari: [
      'Tap the share button (⎗) at the bottom of the screen',
      'Scroll down and tap "Add to Home Screen"',
      'Tap "Add" in the top right corner',
    ],
    edge: [
      'Click the three dots menu (⋯) in the top right corner',
      'Select "Apps" > "Install this site as an app"',
      'Click "Install" in the dialog that appears',
    ],
  }

  return {
    browser,
    instructions: instructions[browser] || instructions.chrome,
  }
}

export default pwaManager