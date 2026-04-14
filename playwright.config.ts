import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 10000,
  retries: 0,
  use: {
    baseURL: "http://127.0.0.1:5173",
    headless: true,
  },
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 5173",
    cwd: "frontend",
    url: "http://127.0.0.1:5173",
    timeout: 120000,
    reuseExistingServer: true,
  },
});
