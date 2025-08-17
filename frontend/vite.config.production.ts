import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'
import { visualizer } from 'rollup-plugin-visualizer'
import { resolve } from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react({
      // Enable React Fast Refresh in development
      fastRefresh: true,
      // Remove React DevTools in production
      babel: {
        plugins: process.env.NODE_ENV === 'production' ? [
          ['babel-plugin-transform-react-remove-prop-types', { removeImport: true }]
        ] : [],
      },
    }),
    
    // Progressive Web App
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'masked-icon.svg'],
      manifest: {
        name: 'KnowledgeHub - AI Intelligence Platform',
        short_name: 'KnowledgeHub',
        description: 'Advanced AI-powered knowledge management and intelligence platform',
        theme_color: '#2196F3',
        background_color: '#121212',
        display: 'standalone',
        icons: [
          {
            src: 'icons/icon-192x192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'maskable any'
          },
          {
            src: 'icons/icon-512x512.png', 
            sizes: '512x512',
            type: 'image/png',
            purpose: 'maskable any'
          }
        ]
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2}'],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/fonts\.googleapis\.com\/.*/i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'google-fonts-cache',
              expiration: {
                maxEntries: 10,
                maxAgeSeconds: 60 * 60 * 24 * 365 // <== 365 days
              },
              cacheKeyWillBeUsed: async ({ request }) => {
                return `${request.url}?version=1.0.0`
              }
            }
          },
          {
            urlPattern: /^https:\/\/fonts\.gstatic\.com\/.*/i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'gstatic-fonts-cache',
              expiration: {
                maxEntries: 10,
                maxAgeSeconds: 60 * 60 * 24 * 365 // <== 365 days
              }
            }
          },
          {
            urlPattern: /\/api\/.*$/,
            handler: 'NetworkFirst',
            options: {
              cacheName: 'api-cache',
              expiration: {
                maxEntries: 100,
                maxAgeSeconds: 60 * 5 // <== 5 minutes
              },
              networkTimeoutSeconds: 10,
            }
          }
        ]
      }
    }),

    // Bundle analyzer (only in analyze mode)
    process.env.ANALYZE && visualizer({
      filename: 'dist/stats.html',
      open: true,
      gzipSize: true,
      brotliSize: true,
    }),
  ],

  // Development server configuration  
  server: {
    port: 3100,
    host: true, // Listen on all network interfaces
    proxy: {
      // Memory system API endpoints
      '/api/v1/memories': {
        target: 'http://192.168.1.25:8003',
        changeOrigin: true,
      },
      '/api/v1/graph': {
        target: 'http://192.168.1.25:8003',
        changeOrigin: true,
      },
      '/api/v1/analytics': {
        target: 'http://192.168.1.25:8003',
        changeOrigin: true,
      },
      // Main KnowledgeHub API endpoints
      '/api': {
        target: 'http://192.168.1.25:3000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://192.168.1.25:3000',
        ws: true,
      },
    },
  },

  // Build configuration
  build: {
    target: 'es2015',
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: process.env.NODE_ENV === 'development',
    minify: 'esbuild',
    
    // Optimize chunks
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
      },
      output: {
        manualChunks: {
          // Vendor chunk for stable dependencies
          vendor: [
            'react',
            'react-dom',
            'react-router-dom',
          ],
          
          // UI library chunk
          mui: [
            '@mui/material',
            '@mui/icons-material',
            '@mui/system',
            '@emotion/react',
            '@emotion/styled',
          ],
          
          // Charts and visualization
          charts: [
            'recharts',
            'framer-motion',
            '@react-three/fiber',
            '@react-three/drei',
          ],
          
          // Utilities
          utils: [
            'axios',
            'date-fns',
            'socket.io-client',
          ],
        },
        
        // Optimize chunk naming
        chunkFileNames: (chunkInfo) => {
          const facadeModuleId = chunkInfo.facadeModuleId
          if (facadeModuleId) {
            const fileName = facadeModuleId.split('/').pop()?.replace('.tsx', '').replace('.ts', '')
            return `assets/${fileName}-[hash].js`
          }
          return 'assets/[name]-[hash].js'
        },
        
        assetFileNames: 'assets/[name]-[hash].[ext]',
        entryFileNames: 'assets/[name]-[hash].js',
      },
    },

    // Performance optimizations
    chunkSizeWarningLimit: 1000,
    cssCodeSplit: true,
    
    // Terser options for better minification
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
        passes: 2,
      },
      mangle: {
        safari10: true,
      },
      format: {
        safari10: true,
      },
    },
  },

  // Dependency optimization
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      '@mui/material',
      '@mui/icons-material',
      'framer-motion',
      'axios',
    ],
    exclude: ['@vitejs/plugin-react'],
  },

  // Path resolution
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@components': resolve(__dirname, 'src/components'),
      '@pages': resolve(__dirname, 'src/pages'),
      '@services': resolve(__dirname, 'src/services'),
      '@utils': resolve(__dirname, 'src/utils'),
      '@theme': resolve(__dirname, 'src/theme'),
      '@hooks': resolve(__dirname, 'src/hooks'),
    },
  },

  // CSS configuration
  css: {
    modules: {
      localsConvention: 'camelCase',
    },
    preprocessorOptions: {
      scss: {
        additionalData: `@import "@/theme/variables.scss";`,
      },
    },
  },

  // Environment variables
  define: {
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
    __VERSION__: JSON.stringify(process.env.npm_package_version || '1.0.0'),
  },

  // Experimental features
  experimental: {
    renderBuiltUrl(filename, { hostType }) {
      if (hostType === 'js') {
        return { js: `/${filename}` }
      } else {
        return { relative: true }
      }
    },
  },

  // Security headers for preview
  preview: {
    port: 4173,
    host: true,
    headers: {
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY',
      'X-XSS-Protection': '1; mode=block',
      'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
      'Referrer-Policy': 'strict-origin-when-cross-origin',
      'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
    },
  },
})