import { expect, test } from "@playwright/test";

test.describe("Billie end to end", () => {
  test("home page shows the three modes and example questions", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "What would you like to know?" })).toBeVisible();
    for (const mode of ["General", "Medical", "Research"]) {
      await expect(page.getByRole("button", { name: mode, exact: true })).toBeVisible();
    }
    await page.getByRole("button", { name: "Medical", exact: true }).click();
    await expect(page.getByText("What are the early symptoms of glaucoma?")).toBeVisible();
  });

  test("a question streams a grounded, cited answer and citations open their source", async ({ page }) => {
    await page.goto("/");
    await page.getByLabel("Message").fill("How does Billie decide which documents to use?");
    await page.getByLabel("Message").press("Enter");

    await expect(page.getByText("Sources", { exact: true })).toBeVisible();
    await expect(page.getByText(/Grounded in \d+ source/)).toBeVisible();
    // URL becomes a shareable deep link to the conversation.
    await expect(page).toHaveURL(/\?c=[0-9a-f]{32}/);

    const chip = page.locator("button[title^='Show source']").first();
    await chip.click();
    await expect(page.locator(".ring-2").first()).toBeVisible(); // highlighted source card
  });

  test("an emergency message shows the safety banner, and Stop halts the answer", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Medical", exact: true }).click();
    await page.getByLabel("Message").fill("I have crushing chest pain and my left arm is numb");
    await page.getByLabel("Message").press("Enter");

    await expect(page.getByRole("alert")).toContainText("call emergency services");
    await page.getByTitle("Stop").click();
    await expect(page.getByTitle("Send (Enter)")).toBeVisible({ timeout: 15_000 });
  });

  test("settings and documents dialogs open", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Model settings", exact: true }).click();
    const settings = page.getByRole("dialog", { name: "Model settings" });
    await expect(settings.getByText("Ollama (local, free)")).toBeVisible();
    await expect(settings.getByText("Anthropic Claude")).toBeVisible();
    await settings.getByRole("button", { name: "Cancel" }).click();

    await page.getByRole("button", { name: "Documents", exact: true }).click();
    await expect(page.getByRole("dialog", { name: "Your documents" })).toBeVisible();
  });
});
