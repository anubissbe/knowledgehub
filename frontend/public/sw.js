// Service Worker for KnowledgeHub PWA
// Provides offline functionality and performance optimizations

const CACHE_NAME = "knowledgehub-v1.0.0"
const STATIC_ASSETS = [
  "/",
  "/index.html", 
  "/manifest.json",
  "/icons/icon-192x192.png",
  "/icons/icon-512x512.png"
]

// API endpoints to cache
const API_CACHE = "knowledgehub-api-v1"
const API_ENDPOINTS = [
  "/api/ai-features/status",
  "/api/memory/stats",
  "/api/system/status"
]

// Install event - cache static assets
self.addEventListener("install", (event) => {
  console.log("[SW] Installing service worker...")
  
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log("[SW] Caching static assets")
      return cache.addAll(STATIC_ASSETS)
    })
  )
  
  // Force activation of new service worker
  self.skipWaiting()
})

// Activate event - cleanup old caches
self.addEventListener("activate", (event) => {
  console.log("[SW] Activating service worker...")
  
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName \!== CACHE_NAME && cacheName \!== API_CACHE) {
            console.log("[SW] Deleting old cache:", cacheName)
            return caches.delete(cacheName)
          }
        })
      )
    })
  )
  
  // Take control of all clients
  self.clients.claim()
})

// Fetch event - implement caching strategy
self.addEventListener("fetch", (event) => {
  const { request } = event
  const url = new URL(request.url)
  
  // Skip non-GET requests
  if (request.method \!== "GET") {
    return
  }
  
  // Handle API requests with network-first strategy
  if (url.pathname.startsWith("/api")) {
    event.respondWith(
      networkFirstStrategy(request, API_CACHE)
    )
    return
  }
  
  // Handle static assets with cache-first strategy  
  if (STATIC_ASSETS.some(asset => url.pathname === asset)) {
    event.respondWith(
      cacheFirstStrategy(request, CACHE_NAME)
    )
    return
  }
  
  // Handle other requests with stale-while-revalidate
  event.respondWith(
    staleWhileRevalidateStrategy(request, CACHE_NAME)
  )
})

// Network-first strategy (for API calls)
async function networkFirstStrategy(request, cacheName) {
  try {
    const response = await fetch(request)
    
    if (response.ok) {
      const cache = await caches.open(cacheName)
      cache.put(request, response.clone())
    }
    
    return response
  } catch (error) {
    console.log("[SW] Network failed, trying cache:", error)
    const cachedResponse = await caches.match(request)
    
    if (cachedResponse) {
      return cachedResponse
    }
    
    // Return offline fallback
    return new Response(
      JSON.stringify({ 
        error: "Offline", 
        message: "Network unavailable" 
      }),
      {
        status: 503,
        headers: { "Content-Type": "application/json" }
      }
    )
  }
}

// Cache-first strategy (for static assets)
async function cacheFirstStrategy(request, cacheName) {
  const cachedResponse = await caches.match(request)
  
  if (cachedResponse) {
    return cachedResponse
  }
  
  try {
    const response = await fetch(request)
    
    if (response.ok) {
      const cache = await caches.open(cacheName)
      cache.put(request, response.clone())
    }
    
    return response
  } catch (error) {
    console.log("[SW] Cache and network failed:", error)
    return new Response("Offline", { status: 503 })
  }
}

// Stale-while-revalidate strategy
async function staleWhileRevalidateStrategy(request, cacheName) {
  const cache = await caches.open(cacheName)
  const cachedResponse = await cache.match(request)
  
  // Fetch in background to update cache
  const fetchPromise = fetch(request).then((response) => {
    if (response.ok) {
      cache.put(request, response.clone())
    }
    return response
  })
  
  // Return cached version immediately if available
  if (cachedResponse) {
    return cachedResponse
  }
  
  // Otherwise wait for network
  return fetchPromise
}

// Background sync for offline actions
self.addEventListener("sync", (event) => {
  if (event.tag === "background-sync") {
    event.waitUntil(
      // Handle any queued offline actions
      handleBackgroundSync()
    )
  }
})

async function handleBackgroundSync() {
  // Implement offline action replay
  console.log("[SW] Handling background sync")
}

// Push notification support
self.addEventListener("push", (event) => {
  if (\!event.data) return
  
  const data = event.data.json()
  const options = {
    body: data.body,
    icon: "/icons/icon-192x192.png",
    badge: "/icons/badge-72x72.png",
    vibrate: [200, 100, 200],
    data: data.data,
    actions: [
      {
        action: "open",
        title: "Open App",
        icon: "/icons/action-open.png"
      },
      {
        action: "close", 
        title: "Dismiss",
        icon: "/icons/action-close.png"
      }
    ]
  }
  
  event.waitUntil(
    self.registration.showNotification(data.title, options)
  )
})

// Handle notification clicks
self.addEventListener("notificationclick", (event) => {
  event.notification.close()
  
  if (event.action === "open") {
    event.waitUntil(
      clients.openWindow("/")
    )
  }
})

// Handle messages from main thread
self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "SKIP_WAITING") {
    self.skipWaiting()
  }
})

console.log("[SW] Service worker script loaded")
