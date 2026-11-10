import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Project page (Pianoid showcase) — single mode, no Supabase backend.
export default defineConfig({
  plugins: [react()],
  publicDir: 'public',
  base: './',
  build: { outDir: 'dist' },
  // Allow Tailscale MagicDNS names when serving to other machines on the tailnet.
  server: { allowedHosts: ['.ts.net', 'herbie-ssd'] },
  preview: { allowedHosts: ['.ts.net', 'herbie-ssd'] },
})
