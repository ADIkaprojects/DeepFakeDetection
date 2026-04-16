export type RpcMethod =
  | "face_detector"
  | "perturbation_generator"
  | "frame_blender"
  | "deepfake_feedback"
  | "nsfw_feedback";

export interface JsonRpcRequest<TParams> {
  jsonrpc: "2.0";
  id: number;
  method: RpcMethod;
  params: TParams;
}

export interface JsonRpcResponse<TResult> {
  jsonrpc: "2.0";
  id: number;
  result?: TResult;
  error?: {
    code: number;
    message: string;
  };
}

export interface FaceDetectorRequest {
  frame_b64: string;
}

export interface FaceDetectorResponse {
  boxes: number[][];
  landmarks?: number[][];
  error?: string | null;
}

export interface PerturbationGeneratorRequest {
  face_b64: string;
  protection_profile?: "shield_only" | "nsfw_trigger_only" | "shield_and_nsfw";
}

export interface PerturbationGeneratorResponse {
  perturbation_b64: string;
  latency_ms?: number;
  protection_profile?: string;
}

export interface FrameBlenderRequest {
  frame_b64: string;
  perturbation_b64: string;
  boxes: number[][];
  alpha: number;
}

export interface FrameBlenderResponse {
  shielded_frame_b64: string;
}

export interface DeepfakeFeedbackRequest {
  frame_b64: string;
}

export interface DeepfakeFeedbackResponse {
  confidence: number;
  label?: "real" | "fake";
}

export interface NsfwFeedbackRequest {
  frame_b64: string;
}

export interface NsfwFeedbackResponse {
  nsfw_score: number;
  label: "nsfw_flagged" | "safe";
  proxies_used: string[];
  error?: string;
}

export interface PipelineMetrics {
  detectionMs?: number;
  perturbationMs?: number;
  blendingMs?: number;
  feedbackMs?: number;
  totalMs: number;
}

export interface PipelineResult {
  inputFrameDataUrl: string;
  outputFrameDataUrl: string;
  boxes: number[][];
  alpha: number;
  feedback?: DeepfakeFeedbackResponse;
  nsfwScore?: number;
  metrics: PipelineMetrics;
}

export type PipelineLogLevel = "info" | "warn" | "error";

export interface PipelineLogEntry {
  id: string;
  createdAt: number;
  level: PipelineLogLevel;
  message: string;
}
