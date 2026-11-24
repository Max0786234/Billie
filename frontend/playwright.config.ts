import { defineConfig } from "@playwright/test";

// End-to-end tests drive the real stack (React UI + FastAPI + local Ollama models).
// They use the Microsoft Edge that ships with Windows, so no browser download is needed.
// Start the app first (scripts/dev.ps1), then: npm run e2e
export default defineConfig({
  testDir: "./e2e",
  timeout: 240_000, // a CPU-only laptop needs up to a minute per answer
  expect: { timeout: 180_000 },
  fullyParallel: false,
  workers: 1, // one local model; parallel questions would just queue
  reporter: [["list"]],
  use: {
    baseURL: process.env.BILLIE_UI_URL ?? "http://127.0.0.1:5173",
    channel: "msedge",
    headless: true,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
});
