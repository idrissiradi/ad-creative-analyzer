import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(),],
  server: {
    port: 5173,
    proxy: {
      // Proxy API calls to FastAPI — no CORS issues in dev
      '/analyze': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
