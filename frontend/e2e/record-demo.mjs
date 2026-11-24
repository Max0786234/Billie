// Records README demo frames: a real question streaming into a cited answer.
// Usage (app running):  node e2e/record-demo.mjs <frames-dir>
// Then:                 python ../scripts/make_demo_gif.py <frames-dir> ../docs/images/demo.gif
import { mkdirSync } from "node:fs";
import { chromium } from "@playwright/test";

const out = process.argv[2] ?? "demo-frames";
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ channel: "msedge" });
const page = await browser.newPage({ viewport: { width: 1280, height: 760 }, colorScheme: "light" });
await page.goto(process.env.BILLIE_UI_URL ?? "http://127.0.0.1:5173/");

let frame = 0;
const snap = async () => page.screenshot({ path: `${out}/${String(frame++).padStart(4, "0")}.png` });

await page.getByRole("button", { name: "Medical", exact: true }).click();
await snap();
await page.getByLabel("Message").pressSequentially("What are the warning signs of a stroke?", { delay: 45 });
await snap();
await page.getByLabel("Message").press("Enter");

const deadline = Date.now() + 150_000;
while (Date.now() < deadline) {
  await snap();
  if (await page.getByText(/Grounded in \d+ source/).isVisible()) break;
  await page.waitForTimeout(400);
}
await page.locator("button[title^='Show source']").first().click(); // open the cited source
await page.waitForTimeout(600);
for (let i = 0; i < 4; i++) await snap(); // hold the final frame
await browser.close();
console.log(`${frame} frames written to ${out}`);
