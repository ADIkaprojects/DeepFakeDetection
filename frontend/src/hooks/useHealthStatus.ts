import { useEffect, useMemo, useState } from "react";
import { appEnv } from "../utils/env";

type ConnectionState = "idle" | "online" | "degraded" | "offline";

export interface HealthSnapshot {
  state: ConnectionState;
  lastCheckedAt: number | null;
  latencyMs: number | null;
  error: string | null;
}

async function checkHealth(endpoint: string): Promise<number> {
  const started = performance.now();
  // Keep health polling a "simple request" to avoid CORS preflight failures in local dev.
  // Some backends allow GET /health but do not implement OPTIONS for custom headers.
  let response = await fetch(endpoint, {
    method: "GET",
  });

  if ((response.status === 401 || response.status === 403) && appEnv.apiKey) {
    response = await fetch(endpoint, {
      method: "GET",
      headers: { "x-api-key": appEnv.apiKey },
    });
  }

  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }

  const payload = (await response.json()) as { status?: string };
  if (payload.status !== "ok") {
    throw new Error("Unexpected health response payload");
  }

  return performance.now() - started;
}

export function useHealthStatus(pollIntervalMs = 5000): HealthSnapshot {
  const [snapshot, setSnapshot] = useState<HealthSnapshot>({
    state: "idle",
    lastCheckedAt: null,
    latencyMs: null,
    error: null,
  });

  useEffect(() => {
    let mounted = true;

    const poll = async () => {
      try {
        const latencyMs = await checkHealth(appEnv.healthEndpoint);
        if (!mounted) {
          return;
        }
        setSnapshot({
          state: latencyMs < 1200 ? "online" : "degraded",
          lastCheckedAt: Date.now(),
          latencyMs,
          error: null,
        });
      } catch (error) {
        if (!mounted) {
          return;
        }
        setSnapshot({
          state: "offline",
          lastCheckedAt: Date.now(),
          latencyMs: null,
          error: error instanceof Error ? error.message : "Unknown health check error",
        });
      }
    };

    void poll();
    const timer = window.setInterval(() => {
      void poll();
    }, pollIntervalMs);

    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, [pollIntervalMs]);

  return useMemo(() => snapshot, [snapshot]);
}
