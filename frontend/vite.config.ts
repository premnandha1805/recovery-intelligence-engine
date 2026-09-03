import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const targetUrl = env.VITE_API_BASE_URL || env.VITE_API_TARGET_URL || 'https://recovery-intelligence-api.onrender.com';

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/decisions': {
          target: targetUrl,
          changeOrigin: true,
          secure: true,
        },
        '/health': {
          target: targetUrl,
          changeOrigin: true,
          secure: true,
        },
      },
    },
    preview: {
      port: 5173,
      proxy: {
        '/decisions': {
          target: targetUrl,
          changeOrigin: true,
          secure: true,
        },
        '/health': {
          target: targetUrl,
          changeOrigin: true,
          secure: true,
        },
      },
    },
  };
});
