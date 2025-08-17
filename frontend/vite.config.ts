import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3100,
    host: "0.0.0.0",
    proxy: {
      "/api/v1/memories": {
        target: "http://192.168.1.25:8003",
        changeOrigin: true,
      },
      "/api/v1/graph": {
        target: "http://192.168.1.25:8003", 
        changeOrigin: true,
      },
      "/api/v1/analytics": {
        target: "http://192.168.1.25:8003",
        changeOrigin: true,
      },
      "/api": {
        target: "http://192.168.1.25:3000",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://192.168.1.25:3000",
        ws: true,
      },
    },
  },
  build: {
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["react", "react-dom", "react-router-dom"],
          ui: ["@mui/material", "@mui/icons-material", "@emotion/react", "@emotion/styled"],
          charts: ["recharts", "d3-array", "d3-scale"],
          vis: ["vis-network", "vis-data"],
          animation: ["framer-motion"],
          utils: ["axios", "date-fns", "immer", "zustand"]
        },
      },
    },
    sourcemap: false,
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
      },
    },
  },
  optimizeDeps: {
    include: [
      "react",
      "react-dom",
      "@mui/material",
      "recharts"
    ],
  },
})
