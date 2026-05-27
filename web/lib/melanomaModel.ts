"use client";

export type PredictionResult = {
  label: "benign" | "malignant";
  scores: Record<"benign" | "malignant", number>;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_INFERENCE_API_URL ?? "http://127.0.0.1:8000";

export async function predictImage(file: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Prediction request failed.");
  }

  const data = (await response.json()) as PredictionResult;
  return data;
}

export function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

