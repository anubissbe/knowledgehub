import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3100,
    proxy: {
      // Memory system API endpoints
      '/api/v1/memories': {
        target: 'http://localhost:8003',
        changeOrigin: true,
      },
      '/api/v1/graph': {
        target: 'http://localhost:8003',
        changeOrigin: true,
      },
      '/api/v1/analytics': {
        target: 'http://localhost:8003',
        changeOrigin: true,
      },
      // Main KnowledgeHub API endpoints
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:3000',
        ws: true,
      },
    },
  },
})