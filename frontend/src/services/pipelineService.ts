import { rpcClient } from "./rpcClient";
import type { FaceDetectorResponse, PipelineResult } from "../types/api";
import { base64ToDataUrl, clampAlpha, dataUrlToBase64 } from "../utils/image";

function now(): number {
  return performance.now();
}

const CROP_PADDING_RATIO = 0.08;
type FaceBox = FaceDetectorResponse["boxes"][number];
const NON_FATAL_DETECTION_ERRORS = new Set([
  "mediapipe_unavailable_full_frame_fallback",
]);

function normalizeBoxToCorners(box: FaceBox): [number, number, number, number] {
  const [x1, y1, x2OrW, y2OrH] = box;
  if (x2OrW > x1 && y2OrH > y1) {
    return [x1, y1, x2OrW, y2OrH];
  }
  return [x1, y1, x1 + x2OrW, y1 + y2OrH];
}

function toFrameDataUrl(frameBase64: string): string {
  return frameBase64.startsWith("data:image/")
    ? frameBase64
    : base64ToDataUrl(frameBase64);
}

function drawCroppedRegionToCanvas(
  image: HTMLImageElement,
  box: FaceBox,
  context: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D,
  width: number,
  height: number,
): void {
  const [rawX1, rawY1, rawX2, rawY2] = normalizeBoxToCorners(box);
  const padX = Math.max(1, Math.floor((rawX2 - rawX1) * CROP_PADDING_RATIO));
  const padY = Math.max(1, Math.floor((rawY2 - rawY1) * CROP_PADDING_RATIO));

  const x1 = Math.max(0, Math.floor(rawX1) - padX);
  const y1 = Math.max(0, Math.floor(rawY1) - padY);
  const x2 = Math.min(image.width, Math.ceil(rawX2) + padX);
  const y2 = Math.min(image.height, Math.ceil(rawY2) + padY);
  const cropWidth = Math.max(1, x2 - x1);
  const cropHeight = Math.max(1, y2 - y1);

  context.drawImage(image, x1, y1, cropWidth, cropHeight, 0, 0, width, height);
}

/**
 * Crop a padded face ROI and return JPEG payload as raw base64 bytes.
 */
export async function cropFaceRegion(frameBase64: string, box: FaceBox): Promise<string> {
  const frameDataUrl = toFrameDataUrl(frameBase64);

  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const [rawX1, rawY1, rawX2, rawY2] = normalizeBoxToCorners(box);
      const width = Math.max(1, Math.ceil(rawX2 - rawX1));
      const height = Math.max(1, Math.ceil(rawY2 - rawY1));

      if (typeof OffscreenCanvas !== "undefined") {
        const canvas = new OffscreenCanvas(width, height);
        const context = canvas.getContext("2d");
        if (!context) {
          reject(new Error("Unable to create offscreen canvas context for face crop."));
          return;
        }
        drawCroppedRegionToCanvas(image, box, context, width, height);
        void canvas.convertToBlob({ type: "image/jpeg", quality: 0.92 }).then((blob) => {
          const reader = new FileReader();
          reader.onload = () => {
            const result = reader.result;
            if (typeof result !== "string") {
              reject(new Error("Failed to serialize offscreen JPEG crop."));
              return;
            }
            resolve(dataUrlToBase64(result));
          };
          reader.onerror = () => reject(new Error("Failed to read offscreen JPEG crop."));
          reader.readAsDataURL(blob);
        }).catch((error: unknown) => {
          reject(error instanceof Error ? error : new Error("Offscreen crop conversion failed."));
        });
        return;
      }

      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext("2d");
      if (!context) {
        reject(new Error("Unable to create canvas context for face crop."));
        return;
      }

      drawCroppedRegionToCanvas(image, box, context, width, height);
      resolve(dataUrlToBase64(canvas.toDataURL("image/jpeg", 0.92)));
    };
    image.onerror = () => reject(new Error("Failed to load input frame for ROI crop."));
    image.src = frameDataUrl;
  });
}

export async function runPipelineOnFrame(
  frameDataUrl: string,
  alpha: number,
  feedbackEnabled: boolean,
  protectionProfile: "shield_only" | "nsfw_trigger_only" | "shield_and_nsfw" = "shield_only",
  nsfwFeedbackEnabled = false,
): Promise<PipelineResult> {
  if (!frameDataUrl.startsWith("data:image/")) {
    throw new Error("Input must be a valid image data URL.");
  }

  const inputFrameB64 = dataUrlToBase64(frameDataUrl);
  const normalizedAlpha = clampAlpha(alpha);
  const startedAt = now();

  const detectionStart = now();
  const detection = await rpcClient.detectFace({ frame_b64: inputFrameB64 });
  const detectionMs = now() - detectionStart;

  const boxes = (detection.boxes ?? []).filter(
    (box) =>
      Array.isArray(box) &&
      box.length === 4 &&
      box.every((item) => typeof item === "number" && Number.isFinite(item)),
  );

  if (detection.error && !NON_FATAL_DETECTION_ERRORS.has(detection.error)) {
    throw new Error(`Face detection error: ${detection.error}`);
  }

  let perturbationMs = 0;
  let perturbationB64 = inputFrameB64;

  if (!boxes.length) {
    console.debug("Skipping perturbation_generator: no face detection for current frame");
  } else {
    const primaryBox = boxes[0];
    const faceCropB64 = await cropFaceRegion(inputFrameB64, primaryBox);

    const perturbStart = now();
    const perturb = await rpcClient.generatePerturbation({
      face_b64: faceCropB64,
      protection_profile: protectionProfile,
    });
    perturbationMs = now() - perturbStart;
    perturbationB64 = perturb.perturbation_b64;
  }

  const blendStart = now();
  const blended = await rpcClient.blendFrame({
    frame_b64: inputFrameB64,
    perturbation_b64: perturbationB64,
    boxes,
    alpha: normalizedAlpha,
  });
  const blendingMs = now() - blendStart;

  let feedbackMs: number | undefined;
  let feedback;

  if (feedbackEnabled) {
    const feedbackStart = now();
    feedback = await rpcClient.getDeepfakeFeedback({
      frame_b64: blended.shielded_frame_b64,
    });
    feedbackMs = now() - feedbackStart;
  }

  let nsfwScore: number | undefined;
  if (nsfwFeedbackEnabled) {
    const nsfw = await runNsfwFeedback(blended.shielded_frame_b64);
    nsfwScore = nsfw.nsfw_score;
  }

  const outputFrameDataUrl = base64ToDataUrl(blended.shielded_frame_b64);

  return {
    inputFrameDataUrl: frameDataUrl,
    outputFrameDataUrl,
    boxes,
    alpha: normalizedAlpha,
    feedback,
    nsfwScore,
    metrics: {
      detectionMs,
      perturbationMs,
      blendingMs,
      feedbackMs,
      totalMs: now() - startedAt,
    },
  };
}

export async function runNsfwFeedback(
  frameB64: string,
): Promise<{ nsfw_score: number; label: string; proxies_used: string[] }> {
  const result = await rpcClient.getNsfwFeedback({ frame_b64: frameB64 });
  return {
    nsfw_score: result.nsfw_score,
    label: result.label,
    proxies_used: result.proxies_used,
  };
}
