import { postJson } from "./httpClient";
import { appEnv } from "../utils/env";
import type {
  DeepfakeFeedbackRequest,
  DeepfakeFeedbackResponse,
  FaceDetectorRequest,
  FaceDetectorResponse,
  FrameBlenderRequest,
  FrameBlenderResponse,
  JsonRpcRequest,
  JsonRpcResponse,
  PerturbationGeneratorRequest,
  PerturbationGeneratorResponse,
  RpcMethod,
} from "../types/api";

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function assertHasStringField(value: unknown, field: string, context: string): void {
  if (!isObject(value) || typeof value[field] !== "string") {
    throw new Error(`${context} is missing required string field: ${field}`);
  }
}

function assertHasArrayField(value: unknown, field: string, context: string): void {
  if (!isObject(value) || !Array.isArray(value[field])) {
    throw new Error(`${context} is missing required array field: ${field}`);
  }
}

export class RpcClient {
  private nextId = 1;

  private async call<TParams, TResult>(
    method: RpcMethod,
    params: TParams,
  ): Promise<TResult> {
    const request: JsonRpcRequest<TParams> = {
      jsonrpc: "2.0",
      id: this.nextId,
      method,
      params,
    };
    this.nextId += 1;

    const response = await postJson<JsonRpcResponse<TResult>>(appEnv.rpcEndpoint, request);

    if (!isObject(response) || response.jsonrpc !== "2.0") {
      throw new Error("Invalid JSON-RPC envelope received from MCP server");
    }

    if (response.error) {
      throw new Error(`RPC ${method} failed: ${response.error.message}`);
    }

    if (!response.result) {
      throw new Error(`Missing result payload for method: ${method}`);
    }

    return response.result;
  }

  async detectFace(payload: FaceDetectorRequest): Promise<FaceDetectorResponse> {
    const result = await this.call<FaceDetectorRequest, FaceDetectorResponse>(
      "face_detector",
      payload,
    );
    assertHasArrayField(result, "boxes", "face_detector result");
    return result;
  }

  async generatePerturbation(
    payload: PerturbationGeneratorRequest,
  ): Promise<PerturbationGeneratorResponse> {
    const result = await this.call<
      PerturbationGeneratorRequest,
      PerturbationGeneratorResponse
    >("perturbation_generator", payload);
    assertHasStringField(result, "perturbation_b64", "perturbation_generator result");
    return result;
  }

  async blendFrame(payload: FrameBlenderRequest): Promise<FrameBlenderResponse> {
    const result = await this.call<FrameBlenderRequest, FrameBlenderResponse>(
      "frame_blender",
      payload,
    );
    assertHasStringField(result, "shielded_frame_b64", "frame_blender result");
    return result;
  }

  async getDeepfakeFeedback(
    payload: DeepfakeFeedbackRequest,
  ): Promise<DeepfakeFeedbackResponse> {
    const result = await this.call<DeepfakeFeedbackRequest, DeepfakeFeedbackResponse>(
      "deepfake_feedback",
      payload,
    );

    if (!isObject(result) || typeof result.confidence !== "number") {
      throw new Error("deepfake_feedback returned an invalid confidence payload");
    }

    return result;
  }
}

export const rpcClient = new RpcClient();
