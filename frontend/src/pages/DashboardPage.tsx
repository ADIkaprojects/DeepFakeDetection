import { useMemo, useState } from "react";
import { Button } from "../components/common/Button";
import { Panel } from "../components/common/Panel";
import { Slider } from "../components/common/Slider";
import { StatusPill } from "../components/common/StatusPill";
import { Toggle } from "../components/common/Toggle";
import { FramePanel } from "../components/dashboard/FramePanel";
import { LogViewer } from "../components/dashboard/LogViewer";
import { MetricRow } from "../components/dashboard/MetricRow";
import { useHealthStatus } from "../hooks/useHealthStatus";
import { usePipelineRunner } from "../hooks/usePipelineRunner";
import { appEnv, getBaseServerUrl } from "../utils/env";
import { fileToDataUrl } from "../utils/image";

function formatMs(value?: number): string {
  if (typeof value !== "number") {
    return "-";
  }
  return `${value.toFixed(1)} ms`;
}

function formatTimestamp(value: number | null): string {
  if (!value) {
    return "Never";
  }
  return new Date(value).toLocaleTimeString();
}

export function DashboardPage() {
  const health = useHealthStatus();
  const pipeline = usePipelineRunner();

  const [frameDataUrl, setFrameDataUrl] = useState<string>("");
  const [alpha, setAlpha] = useState<number>(0.12);
  const [feedbackEnabled, setFeedbackEnabled] = useState<boolean>(true);
  const [streamMode, setStreamMode] = useState<boolean>(false);
  const [streamIntervalMs, setStreamIntervalMs] = useState<number>(1000);

  const canRun = useMemo(
    () => !!frameDataUrl && !pipeline.isRunning && !pipeline.isStreaming,
    [frameDataUrl, pipeline.isRunning, pipeline.isStreaming],
  );

  const handleFilePicked = async (file: File | null) => {
    if (!file) {
      return;
    }
    const dataUrl = await fileToDataUrl(file);
    setFrameDataUrl(dataUrl);
    pipeline.reset();
  };

  const handleRun = async () => {
    if (!frameDataUrl) {
      return;
    }
    await pipeline.run(frameDataUrl, { alpha, feedbackEnabled });
  };

  const handleStreamStart = () => {
    if (!frameDataUrl) {
      return;
    }

    pipeline.startStream(frameDataUrl, { alpha, feedbackEnabled }, streamIntervalMs);
  };

  const backendSummary = useMemo(
    () => ({
      baseUrl: getBaseServerUrl(),
      rpcEndpoint: appEnv.rpcEndpoint,
      healthEndpoint: appEnv.healthEndpoint,
    }),
    [],
  );

  return (
    <main className="mx-auto flex w-full max-w-7xl flex-col gap-5 px-4 py-6 sm:px-6 lg:px-8">
      <header className="glass-panel p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.2em] text-cyan-700">Adversarial Face Shield</p>
            <h1 className="font-display text-3xl font-semibold text-slate-950 sm:text-4xl">Realtime Pipeline Console</h1>
            <p className="mt-1 max-w-3xl text-sm text-slate-600">
              Upload a frame, tune alpha, and execute the MCP toolchain against live backend endpoints.
            </p>
          </div>
          <StatusPill
            tone={health.state}
            label={health.state === "online" ? "Server Online" : health.state === "degraded" ? "Server Slow" : health.state === "offline" ? "Server Offline" : "Checking"}
          />
        </div>
      </header>

      <section className="grid gap-5 lg:grid-cols-[360px,1fr]">
        <div className="space-y-5">
          <Panel title="Pipeline Controls" subtitle="Tune runtime behavior and execution settings.">
            <div className="space-y-4">
              <Slider label="Alpha Intensity" min={0} max={1} step={0.01} value={alpha} onChange={setAlpha} />
              <Toggle
                label="Enable Feedback Loop"
                helpText="Calls deepfake_feedback after frame blending."
                enabled={feedbackEnabled}
                onChange={setFeedbackEnabled}
              />
              <Toggle
                label="Enable Stream Mode"
                helpText="Repeatedly processes the current frame to emulate live updates."
                enabled={streamMode}
                onChange={setStreamMode}
              />
              {streamMode ? (
                <Slider
                  label="Stream Interval (ms)"
                  min={250}
                  max={5000}
                  step={50}
                  value={streamIntervalMs}
                  onChange={setStreamIntervalMs}
                />
              ) : null}
              <div className="rounded-xl border border-dashed border-slate-300 p-3">
                <label className="text-sm font-medium text-slate-900" htmlFor="frame-upload">
                  Input Frame
                </label>
                <input
                  id="frame-upload"
                  className="mt-2 block w-full cursor-pointer rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700"
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  onChange={(event) => {
                    void handleFilePicked(event.target.files?.[0] || null);
                  }}
                />
                <p className="mt-2 text-xs text-slate-500">TODO: Switch to live webcam capture once browser-safe stream endpoint is available.</p>
              </div>
              <div className="flex flex-wrap gap-2">
                {streamMode ? (
                  <>
                    <Button onClick={handleStreamStart} disabled={!frameDataUrl || pipeline.isStreaming}>
                      {pipeline.isStreaming ? "Streaming..." : "Start Stream"}
                    </Button>
                    <Button variant="danger" onClick={pipeline.stopStream} disabled={!pipeline.isStreaming}>
                      Stop Stream
                    </Button>
                  </>
                ) : (
                  <Button onClick={handleRun} disabled={!canRun}>
                    {pipeline.isRunning ? "Running..." : "Run Pipeline"}
                  </Button>
                )}
                <Button variant="neutral" onClick={() => pipeline.reset()}>
                  Reset Result
                </Button>
              </div>
            </div>
          </Panel>

          <Panel title="Service Health" subtitle="Connectivity and response profile.">
            <div className="space-y-1 text-sm text-slate-700">
              <p>
                Last checked: <span className="font-mono">{formatTimestamp(health.lastCheckedAt)}</span>
              </p>
              <p>
                Latency: <span className="font-mono">{formatMs(health.latencyMs ?? undefined)}</span>
              </p>
              <p>
                Status: <span className="font-mono">{health.state}</span>
              </p>
              <p>
                Base URL: <span className="font-mono break-all">{backendSummary.baseUrl}</span>
              </p>
              {health.error ? <p className="text-rose-700">{health.error}</p> : null}
            </div>
          </Panel>

          <Panel
            title="Execution Metrics"
            subtitle="Per-stage timings and throughput for the latest pipeline run."
          >
            {pipeline.result ? (
              <div>
                <MetricRow label="Detection" value={formatMs(pipeline.result.metrics.detectionMs)} />
                <MetricRow label="Perturbation" value={formatMs(pipeline.result.metrics.perturbationMs)} />
                <MetricRow label="Blending" value={formatMs(pipeline.result.metrics.blendingMs)} />
                <MetricRow label="Feedback" value={formatMs(pipeline.result.metrics.feedbackMs)} />
                <MetricRow label="Total" value={formatMs(pipeline.result.metrics.totalMs)} />
                <MetricRow label="Runs" value={String(pipeline.runCount)} />
              </div>
            ) : (
              <p className="text-sm text-slate-500">No execution yet.</p>
            )}
          </Panel>
        </div>

        <div className="space-y-5">
          <Panel title="Frame Workspace" subtitle="Input and shielded output comparison.">
            <div className="grid gap-4 md:grid-cols-2">
              <FramePanel
                title="Input"
                imageSrc={frameDataUrl}
                alt="Input frame"
                emptyMessage="Upload a frame to begin."
              />
              <FramePanel
                title="Shielded Output"
                imageSrc={pipeline.result?.outputFrameDataUrl}
                alt="Shielded frame"
                emptyMessage="Run pipeline to render output."
              />
            </div>
          </Panel>

          <Panel title="Detection + Feedback" subtitle="Model outputs and runtime observations.">
            {pipeline.result ? (
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                  <h3 className="text-sm font-semibold text-slate-900">Detected Bounding Boxes</h3>
                  <pre className="mt-2 max-h-48 overflow-auto rounded-lg bg-slate-900 p-3 font-mono text-xs text-cyan-100">
                    {JSON.stringify(pipeline.result.boxes, null, 2)}
                  </pre>
                </div>
                <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                  <h3 className="text-sm font-semibold text-slate-900">Feedback Result</h3>
                  {pipeline.result.feedback ? (
                    <pre className="mt-2 rounded-lg bg-slate-900 p-3 font-mono text-xs text-cyan-100">
                      {JSON.stringify(pipeline.result.feedback, null, 2)}
                    </pre>
                  ) : (
                    <p className="mt-2 text-sm text-slate-500">Feedback disabled or unavailable.</p>
                  )}
                </div>
              </div>
            ) : (
              <p className="text-sm text-slate-500">No model outputs yet.</p>
            )}
            {pipeline.error ? (
              <div className="mt-3 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
                {pipeline.error}
              </div>
            ) : null}
          </Panel>

          <Panel
            title="Runtime Events"
            subtitle="Recent actions and backend responses."
            actions={
              <Button variant="neutral" onClick={pipeline.clearLogs}>
                Clear Logs
              </Button>
            }
          >
            <LogViewer entries={pipeline.logs} />
          </Panel>

          <Panel title="Integration Notes" subtitle="Endpoints and extension points.">
            <div className="space-y-2 text-sm text-slate-700">
              <p>
                RPC endpoint: <span className="font-mono break-all">{backendSummary.rpcEndpoint}</span>
              </p>
              <p>
                Health endpoint: <span className="font-mono break-all">{backendSummary.healthEndpoint}</span>
              </p>
              <p>
                TODO: Add a backend endpoint that returns Prometheus metrics snapshots for richer dashboard charts.
              </p>
              <p>
                TODO: Replace frame upload with live browser capture once a secure stream/WebRTC contract is finalized.
              </p>
            </div>
          </Panel>
        </div>
      </section>
    </main>
  );
}
