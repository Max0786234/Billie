// Captures the README media from a running Billie (e.g. the live demo).
// Usage: node e2e/readme-media.mjs <base url> <frames dir> <images dir> <document to upload>
// Then:  python ../scripts/make_demo_gif.py <frames dir> ../docs/images/demo.gif
import { mkdirSync } from "node:fs";
import { chromium } from "@playwright/test";

const [BASE, FRAMES, IMAGES, DOC] = process.argv.slice(2);
mkdirSync(FRAMES, { recursive: true });
mkdirSync(IMAGES, { recursive: true });
const browser = await chromium.launch({ channel: "msedge" });
const page = await browser.newPage({ viewport: { width: 1280, height: 800 }, colorScheme: "light" });
await page.goto(BASE, { timeout: 180_000 });

let frame = 0;
const snap = async () => page.screenshot({ path: `${FRAMES}/${String(frame++).padStart(4, "0")}.png` });
const box = page.getByLabel("Message");

// 1. GIF: what Medical mode contains -> typing suggests real questions -> a cited answer.
await page.getByRole("button", { name: "Medical", exact: true }).click();
await page.getByText(/expert answers on/).waitFor({ timeout: 120_000 });
for (let i = 0; i < 6; i++) await snap(); // hold on the coverage card
for (const ch of "glauco") {
  await box.press(ch === " " ? "Space" : ch);
  await page.waitForTimeout(160);
  await snap();
}
await page.getByRole("listbox", { name: "Suggested questions" }).waitFor({ timeout: 20_000 });
for (let i = 0; i < 6; i++) await snap();
await page.getByRole("option", { name: "What is Glaucoma?" }).click();
const deadline = Date.now() + 90_000;
while (Date.now() < deadline) {
  await snap();
  if (await page.getByText(/Grounded in \d+ source/).isVisible()) break;
  await page.waitForTimeout(150);
}
await page.locator("button[title^='Show source']").first().click();
for (let i = 0; i < 4; i++) await snap();

// 2. Research mode: what it contains.
await page.getByRole("button", { name: /^New chat/ }).click();
await page.getByRole("button", { name: "Research", exact: true }).click();
await page.getByText(/papers from arXiv/).first().waitFor({ timeout: 60_000 });
await page.screenshot({ path: `${IMAGES}/mode-guide.png` });

// 3. Research answer.
await box.fill("Recent approaches to reduce hallucination in large language models");
await box.press("Escape");
await box.press("Enter");
await page.getByText(/Grounded in \d+ source/).waitFor({ timeout: 120_000 });
await page.screenshot({ path: `${IMAGES}/research.png` });

// 4. A visitor's own document, answered with citations ("Your document").
await page.getByRole("button", { name: /^New chat/ }).click();
await page.getByRole("button", { name: "General", exact: true }).click();
await page.getByText("Have your own document?").click();
await page.locator('input[accept=".pdf,.md,.txt"]').setInputFiles(DOC);
await page.getByText(/passages? searchable/).waitFor({ timeout: 240_000 });
await page.getByRole("button", { name: "Done" }).click();
await box.fill("Which department received the largest budget, and how much was it?");
await box.press("Escape");
await box.press("Enter");
await page.getByText(/Grounded in \d+ source/).waitFor({ timeout: 120_000 });
await page.screenshot({ path: `${IMAGES}/private-docs.png` });

// Clean up the uploaded file.
await page.getByRole("button", { name: "Documents", exact: true }).click();
await page.getByRole("button", { name: /^Remove / }).first().click();
await page.getByText("No documents yet.").waitFor();
await browser.close();
console.log(`frames: ${frame}`);
