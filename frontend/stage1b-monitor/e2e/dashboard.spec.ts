import { expect, test } from "@playwright/test";

import snapshot from "./snapshot.json" with { type: "json" };

test("renders a live telemetry snapshot supplied by the monitoring API contract", async ({ page }) => {
  await page.route("**/api/v1/**", async (route) => {
    const endpoint = route.request().url().split("/").at(-1) as keyof typeof snapshot;
    await route.fulfill({ json: snapshot[endpoint] });
  });
  await page.goto("/");
  await expect(page).toHaveTitle("TARCA 运行监督");
  await expect(page.getByRole("heading", { name: "Stage 2 v1" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "GPU 0" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "GPU 1" })).toBeVisible();
  await expect(page.getByText("预计剩余时间")).toBeVisible();
  await expect(page.getByText("数据正常").first()).toBeVisible();
  await expect(page.getByText("最后采样")).toBeVisible();
});
