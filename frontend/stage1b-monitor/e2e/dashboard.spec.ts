import { expect, test } from "@playwright/test";

import snapshot from "./snapshot.json" with { type: "json" };

test("renders a live telemetry snapshot supplied by the monitoring API contract", async ({ page }) => {
  await page.route("**/api/v1/**", async (route) => {
    const endpoint = route.request().url().split("/").at(-1) as keyof typeof snapshot;
    await route.fulfill({ json: snapshot[endpoint] });
  });
  await page.goto("/");
  await expect(page.getByText("Stage1B v2")).toBeVisible();
  await expect(page.getByText("GPU 0")).toBeVisible();
  await expect(page.getByText("GPU 1")).toBeVisible();
  await expect(page.getByText("预计剩余时间")).toBeVisible();
  await expect(page.getByText("数据正常").first()).toBeVisible();
  await expect(page.getByText("最后采样")).toBeVisible();
});
