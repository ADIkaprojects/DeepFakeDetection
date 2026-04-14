import { expect, test } from "@playwright/test";

const ONE_PIXEL_PNG_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAoMBgUY8ailAAAAASUVORK5CYII=";

test("pipeline happy path renders output without error", async ({ page }) => {
  let rpcCallCount = 0;

  await page.route("**/rpc", async (route) => {
    const body = route.request().postDataJSON() as {
      id: number;
      method: string;
      params: Record<string, unknown>;
    };

    rpcCallCount += 1;

    if (body.method === "face_detector") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: body.id,
          result: {
            boxes: [[0, 0, 1, 1]],
            landmarks: [],
            error: null,
          },
        }),
      });
      return;
    }

    if (body.method === "perturbation_generator") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: body.id,
          result: {
            perturbation_b64: ONE_PIXEL_PNG_BASE64,
            latency_ms: 1,
          },
        }),
      });
      return;
    }

    if (body.method === "frame_blender") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: body.id,
          result: {
            shielded_frame_b64: ONE_PIXEL_PNG_BASE64,
          },
        }),
      });
      return;
    }

    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: body.id,
        result: {
          confidence: 0.1,
          label: "real",
        },
      }),
    });
  });

  await page.route("**/health", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "ok" }),
    });
  });

  await page.goto("/");

  const fileInput = page.locator("#frame-upload");
  await fileInput.setInputFiles({
    name: "frame.png",
    mimeType: "image/png",
    buffer: Buffer.from(ONE_PIXEL_PNG_BASE64, "base64"),
  });

  const runButton = page.getByRole("button", { name: "Run Pipeline" });
  await expect(runButton).toBeEnabled();
  await runButton.click();

  await expect(page.getByRole("button", { name: "Running..." })).toBeVisible();
  await expect(page.getByText("Shielded Output")).toBeVisible();
  await expect(page.locator("img[alt='Shielded frame']")).toBeVisible();
  await expect(page.getByText("Pipeline execution failed")).toHaveCount(0);
  expect(rpcCallCount).toBeGreaterThan(0);
});
