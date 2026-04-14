export async function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result);
      } else {
        reject(new Error("Unable to read file as data URL."));
      }
    };
    reader.onerror = () => reject(new Error("File read failed."));
    reader.readAsDataURL(file);
  });
}

export function dataUrlToBase64(dataUrl: string): string {
  const parts = dataUrl.split(",");
  if (parts.length < 2 || !parts[1]) {
    throw new Error("Invalid data URL payload.");
  }
  return parts[1];
}

export function base64ToDataUrl(base64Payload: string, mime = "image/png"): string {
  return `data:${mime};base64,${base64Payload}`;
}

export function clampAlpha(value: number): number {
  if (Number.isNaN(value)) {
    return 0.12;
  }
  return Math.max(0, Math.min(1, value));
}
