// One-off browser check of visitors' private documents on a demo-mode server (screenshots to OUT).
// Usage: node e2e/private-docs-check.mjs <out dir> <base url> <document to upload>
import { chromium } from "@playwright/test";

const [OUT = ".", BASE = "http://127.0.0.1:5174", DOC] = process.argv.slice(2);
const browser = await chromium.launch({ channel: "msedge" });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
await page.goto(BASE);
await page.getByText("Have your own document?").waitFor({ timeout: 30000 });
await page.screenshot({ path: `${OUT}/p1-general-hint.png` });

await page.getByText("Have your own document?").click();
await page.getByText("Your private documents").waitFor();
await page.locator('input[accept=".pdf,.md,.txt"]').setInputFiles(DOC);
await page.getByText(/passages? searchable/).waitFor({ timeout: 120000 });
await page.screenshot({ path: `${OUT}/p2-dialog.png` });
await page.getByRole("button", { name: "Done" }).click();

const box = page.getByLabel("Message");
await box.fill("Who approves remote work from outside India?");
await box.press("Escape");
await box.press("Enter");
await page.getByText(/Grounded in|No sources cited/).first().waitFor({ timeout: 180000 });
await page.screenshot({ path: `${OUT}/p3-answer.png` });
await browser.close();
