import { expect, test } from "@playwright/test";

const ONE_PIXEL_PNG_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAoMBgUY8ailAAAAASUVORK5CYII=";

test("pipeline surfaces backend errors and stream can be stopped", async ({ page }) => {
  await page.route("**/health", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "ok" }),
    });
  });

  await page.route("**/rpc", async (route) => {
    const body = route.request().postDataJSON() as { id: number; method: string };

    if (body.method === "face_detector") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: body.id,
          result: { boxes: [[0, 0, 1, 1]], landmarks: [], error: null },
        }),
      });
      return;
    }

    if (body.method === "perturbation_generator") {
      await route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: body.id,
          error: { code: -32000, message: "Injected perturbation failure" },
        }),
      });
      return;
    }

    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ jsonrpc: "2.0", id: body.id, result: {} }),
    });
  });

  await page.goto("/");

  await page.locator("#frame-upload").setInputFiles({
    name: "frame.png",
    mimeType: "image/png",
    buffer: Buffer.from(ONE_PIXEL_PNG_BASE64, "base64"),
  });

  await page.getByRole("button", { name: "Enable Stream Mode" }).click();
  await page.getByRole("button", { name: "Start Stream" }).click();

  await expect(page.getByRole("button", { name: "Stop Stream" })).toBeVisible();
  await expect(page.getByText("Injected perturbation failure")).toBeVisible();

  await page.getByRole("button", { name: "Stop Stream" }).click();
  await expect(page.getByRole("button", { name: "Start Stream" })).toBeVisible();
});
