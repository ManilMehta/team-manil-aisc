"use client";

import * as tf from "@tensorflow/tfjs";

const IMAGE_SIZE = 224;

let cachedModel: tf.LayersModel | null = null;

export type PredictionResult = {
  label: "benign" | "malignant";
  scores: Record<"benign" | "malignant", number>;
};

export async function loadMelanomaModel() {
  if (cachedModel) return cachedModel;
  cachedModel = await tf.loadLayersModel("/model/model.json");
  return cachedModel;
}

export async function predictImage(file: File): Promise<PredictionResult> {
  const model = await loadMelanomaModel();
  const imageBitmap = await createImageBitmap(file);
  const tensor = tf.tidy(() => {
    const image = tf.browser.fromPixels(imageBitmap).toFloat();
    const resized = tf.image.resizeBilinear(image, [IMAGE_SIZE, IMAGE_SIZE]);
    const normalized = resized.div(255);
    return normalized.expandDims(0);
  });

  try {
    const output = model.predict(tensor) as tf.Tensor;
    const probabilities = Array.from(await output.data());
    output.dispose();

    const scores = {
      benign: probabilities[0] ?? 0,
      malignant: probabilities[1] ?? 0,
    };

    const label = scores.malignant > scores.benign ? "malignant" : "benign";
    return { label, scores };
  } finally {
    tensor.dispose();
    imageBitmap.close();
  }
}

export function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

