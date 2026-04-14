const DEFAULT_RPC_ENDPOINT = "http://127.0.0.1:8080/rpc";
const DEFAULT_HEALTH_ENDPOINT = "http://127.0.0.1:8080/health";

function trimTrailingSlash(value: string): string {
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

export const appEnv = {
  rpcEndpoint: import.meta.env.VITE_AFS_RPC_ENDPOINT || DEFAULT_RPC_ENDPOINT,
  healthEndpoint:
    import.meta.env.VITE_AFS_HEALTH_ENDPOINT || DEFAULT_HEALTH_ENDPOINT,
  apiKey: import.meta.env.VITE_AFS_API_KEY || "",
  requestTimeoutMs: Number(import.meta.env.VITE_AFS_REQUEST_TIMEOUT_MS || 20000),
  maxRetries: Number(import.meta.env.VITE_AFS_MAX_RETRIES || 2),
};

export function getBaseServerUrl(): string {
  const configured = import.meta.env.VITE_AFS_BASE_URL;
  if (configured) {
    return trimTrailingSlash(configured);
  }

  if (appEnv.rpcEndpoint.includes("/rpc")) {
    return trimTrailingSlash(appEnv.rpcEndpoint.replace(/\/rpc$/, ""));
  }

  return trimTrailingSlash(appEnv.rpcEndpoint);
}
