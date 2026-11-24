// One-off visual check of the mode guide, suggestions and dead-end guidance (screenshots to OUT).
import { chromium } from "@playwright/test";

const OUT = process.argv[2] ?? ".";
const browser = await chromium.launch({ channel: "msedge" });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
await page.goto("http://127.0.0.1:5173/");

await page.getByRole("button", { name: /^Medical/ }).first().click();
await page.getByText(/expert answers on/).waitFor({ timeout: 30000 });
await page.screenshot({ path: `${OUT}/1-medical-guide.png` });

await page.getByRole("button", { name: /^Research/ }).first().click();
await page.getByText(/papers from arXiv \(/).waitFor({ timeout: 60000 });
await page.screenshot({ path: `${OUT}/2-research-guide.png` });

await page.getByRole("button", { name: /^Medical/ }).first().click();
const box = page.getByLabel("Message");
await box.fill("glauc");
await page.getByRole("listbox", { name: "Suggested questions" }).waitFor({ timeout: 10000 });
await page.screenshot({ path: `${OUT}/3-suggestions.png` });

// A computer-science question asked in Medical mode is a dead end: expect guidance.
await box.fill("");
await page.getByRole("button", { name: /^New chat/ }).click();
await page.getByRole("button", { name: /^Medical/ }).first().click();
await box.fill("How does BM25 ranking work in search engines?");
await box.press("Escape");
await box.press("Enter");
await page.getByText(/Ask in (General|Research) mode|Closest topics I do have/).first().waitFor({ timeout: 120000 });
await page.screenshot({ path: `${OUT}/4-dead-end.png` });
await page.getByRole("button", { name: /Ask in (General|Research) mode/ }).click();
await page.getByText(/Grounded in/).first().waitFor({ timeout: 120000 });
await page.screenshot({ path: `${OUT}/5-switched.png` });
await browser.close();
