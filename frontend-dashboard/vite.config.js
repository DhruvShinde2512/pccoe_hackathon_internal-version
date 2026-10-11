import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// `host: true` lets teammates open the dashboard over the Tailscale network.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, host: true },
  preview: { port: 4173, host: true },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom'],
          cytoscape: ['cytoscape'],
        },
      },
    },
  },
});
