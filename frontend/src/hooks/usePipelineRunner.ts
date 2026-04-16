import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PipelineLogEntry, PipelineResult } from "../types/api";
import { runPipelineOnFrame } from "../services/pipelineService";

export interface PipelineControls {
  alpha: number;
  feedbackEnabled: boolean;
  nsfwFeedbackEnabled?: boolean;
  protectionProfile?: "shield_only" | "nsfw_trigger_only" | "shield_and_nsfw";
}

interface PipelineState {
  isRunning: boolean;
  isStreaming: boolean;
  skippedFrames: number;
  error: string | null;
  result: PipelineResult | null;
  runCount: number;
  logs: PipelineLogEntry[];
}

const initialState: PipelineState = {
  isRunning: false,
  isStreaming: false,
  skippedFrames: 0,
  error: null,
  result: null,
  runCount: 0,
  logs: [],
};

function makeLog(level: PipelineLogEntry["level"], message: string): PipelineLogEntry {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
    createdAt: Date.now(),
    level,
    message,
  };
}

export function usePipelineRunner() {
  const [state, setState] = useState<PipelineState>(initialState);
  const [nsfwScore, setNsfwScore] = useState<number | null>(null);
  const [protectionProfile, setProtectionProfile] = useState<
    "shield_only" | "nsfw_trigger_only" | "shield_and_nsfw"
  >("shield_only");
  const intervalRef = useRef<number | null>(null);
  const runInFlightRef = useRef<boolean>(false);
  const latestFrameRef = useRef<string>("");
  const latestControlsRef = useRef<PipelineControls>({
    alpha: 0.12,
    feedbackEnabled: false,
    nsfwFeedbackEnabled: false,
    protectionProfile: "shield_only",
  });

  const stopStream = useCallback(() => {
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    runInFlightRef.current = false;
    setState((prev) => ({
      ...prev,
      isStreaming: false,
      logs: [makeLog("info", "Stream mode stopped"), ...prev.logs].slice(0, 50),
    }));
  }, [protectionProfile]);

  const run = useCallback(async (frameDataUrl: string, controls: PipelineControls) => {
    if (runInFlightRef.current) {
      return null;
    }

    runInFlightRef.current = true;
    latestFrameRef.current = frameDataUrl;
    latestControlsRef.current = controls;

    setState((prev) => ({ ...prev, isRunning: true, error: null }));

    try {
      const result = await runPipelineOnFrame(
        frameDataUrl,
        controls.alpha,
        controls.feedbackEnabled,
        controls.protectionProfile ?? protectionProfile,
        controls.nsfwFeedbackEnabled ?? false,
      );
      setNsfwScore(typeof result.nsfwScore === "number" ? result.nsfwScore : null);
      setState((prev) => ({
        ...prev,
        isRunning: false,
        error: null,
        result,
        runCount: prev.runCount + 1,
        logs: [
          makeLog("info", `Run completed in ${result.metrics.totalMs.toFixed(1)} ms`),
          ...prev.logs,
        ].slice(0, 50),
      }));
      return result;
    } catch (error) {
      setState((prev) => ({
        ...prev,
        isRunning: false,
        error: error instanceof Error ? error.message : "Pipeline execution failed",
        logs: [
          makeLog(
            "error",
            error instanceof Error ? error.message : "Pipeline execution failed",
          ),
          ...prev.logs,
        ].slice(0, 50),
      }));
      return null;
    } finally {
      runInFlightRef.current = false;
    }
  }, []);

  const runLatest = useCallback(async () => {
    if (!latestFrameRef.current) {
      return;
    }
    await run(latestFrameRef.current, latestControlsRef.current);
  }, [run]);

  const startStream = useCallback(
    (frameDataUrl: string, controls: PipelineControls, intervalMs: number) => {
      latestFrameRef.current = frameDataUrl;
      latestControlsRef.current = controls;

      const safeInterval = Math.max(250, intervalMs);

      if (intervalRef.current !== null) {
        window.clearInterval(intervalRef.current);
      }

      setState((prev) => ({
        ...prev,
        isStreaming: true,
        skippedFrames: 0,
        error: null,
        logs: [
          makeLog("info", `Stream mode started (${safeInterval} ms interval)`),
          ...prev.logs,
        ].slice(0, 50),
      }));

      intervalRef.current = window.setInterval(() => {
        if (runInFlightRef.current) {
          setState((prev) => ({
            ...prev,
            skippedFrames: prev.skippedFrames + 1,
          }));
          return;
        }
        void runLatest();
      }, safeInterval);

      void runLatest();
    },
    [runLatest],
  );

  const clearLogs = useCallback(() => {
    setState((prev) => ({ ...prev, logs: [] }));
  }, []);

  const reset = useCallback(() => {
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    runInFlightRef.current = false;
    setState(initialState);
  }, []);

  useEffect(() => {
    return () => {
      if (intervalRef.current !== null) {
        window.clearInterval(intervalRef.current);
      }
      runInFlightRef.current = false;
    };
  }, []);

  return useMemo(
    () => ({
      ...state,
      nsfwScore,
      protectionProfile,
      setProtectionProfile,
      run,
      startStream,
      stopStream,
      clearLogs,
      reset,
    }),
    [
      state,
      nsfwScore,
      protectionProfile,
      run,
      startStream,
      stopStream,
      clearLogs,
      reset,
    ],
  );
}
