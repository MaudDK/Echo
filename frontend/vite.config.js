import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Proxy API calls to the generation service so the browser hits the dev server
// origin (no CORS needed in dev). Adjust target if the API runs elsewhere.
const API_TARGET = process.env.ECHO_API_URL || "http://localhost:8100";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/auth": API_TARGET,
      "/sessions": API_TARGET,
      "/health": API_TARGET,
    },
  },
});
