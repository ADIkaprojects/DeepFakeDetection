import { appEnv } from "../utils/env";

interface RequestOptions {
  timeoutMs?: number;
  retries?: number;
}

function sleep(delayMs: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, delayMs);
  });
}

export async function postJson<TResponse>(
  url: string,
  payload: unknown,
  options: RequestOptions = {},
): Promise<TResponse> {
  const timeoutMs = options.timeoutMs ?? appEnv.requestTimeoutMs;
  const retries = options.retries ?? appEnv.maxRetries;

  let attempt = 0;
  let lastError: unknown;

  while (attempt <= retries) {
    attempt += 1;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(appEnv.apiKey ? { "x-api-key": appEnv.apiKey } : {}),
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!response.ok) {
        const bodyText = await response.text();
        throw new Error(`HTTP ${response.status}: ${bodyText || response.statusText}`);
      }

      const data = (await response.json()) as TResponse;
      clearTimeout(timeout);
      return data;
    } catch (error) {
      clearTimeout(timeout);
      lastError = error;
      if (attempt > retries) {
        break;
      }
      await sleep(200 * 2 ** (attempt - 1));
    }
  }

  throw lastError instanceof Error ? lastError : new Error("Request failed");
}
