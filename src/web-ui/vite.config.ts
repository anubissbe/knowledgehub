import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3001,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://192.168.1.25:3000',
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: 'ws://192.168.1.25:3000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
})