"use client";

import { useMemo, useState } from "react";
import { formatPercent, predictImage } from "@/lib/melanomaModel";

type UiState = "idle" | "running" | "done" | "error";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<UiState>("idle");
  const [error, setError] = useState<string>("");
  const [result, setResult] = useState<{
    label: "benign" | "malignant";
    benign: number;
    malignant: number;
  } | null>(null);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  async function runPrediction() {
    if (!file) return;
    setStatus("running");
    setError("");
    try {
      const prediction = await predictImage(file);
      setResult({
        label: prediction.label,
        benign: prediction.scores.benign,
        malignant: prediction.scores.malignant,
      });
      setStatus("done");
    } catch {
      setStatus("error");
      setError(
        "Could not run prediction. Make sure /public/model/model.json exists (run export script first).",
      );
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 px-6 py-10 text-slate-900">
      <div className="mx-auto grid w-full max-w-5xl gap-6 lg:grid-cols-2">
        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <h1 className="text-2xl font-semibold">Skin Lesion Demo</h1>
          <p className="mt-2 text-sm text-slate-600">
            Upload a lesion image to run your benign vs malignant classifier directly in
            the browser.
          </p>

          <label className="mt-6 block rounded-xl border border-dashed border-slate-300 p-4 text-sm">
            <span className="mb-2 block text-slate-700">Choose an image (jpg/png)</span>
            <input
              className="block w-full text-sm"
              type="file"
              accept="image/png,image/jpeg,image/jpg"
              onChange={(e) => {
                const selected = e.target.files?.[0] ?? null;
                setFile(selected);
                setResult(null);
                setStatus("idle");
              }}
            />
          </label>

          <button
            type="button"
            onClick={runPrediction}
            disabled={!file || status === "running"}
            className="mt-4 w-full rounded-xl bg-slate-900 px-4 py-3 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {status === "running" ? "Analyzing..." : "Run prediction"}
          </button>

          <p className="mt-4 rounded-lg bg-amber-50 p-3 text-xs text-amber-800">
            Demo only. This tool is not a medical diagnosis system.
          </p>
        </section>

        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold">Result</h2>
          {!file && <p className="mt-4 text-sm text-slate-500">Upload an image to begin.</p>}

          {file && (
            <>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={previewUrl}
                alt="Uploaded lesion preview"
                className="mt-4 h-56 w-full rounded-xl border border-slate-200 object-cover"
              />
            </>
          )}

          {status === "error" && (
            <p className="mt-4 rounded-lg bg-rose-50 p-3 text-sm text-rose-700">{error}</p>
          )}

          {result && (
            <div className="mt-4 space-y-3 rounded-xl border border-slate-200 p-4">
              <p className="text-sm">
                Predicted class:{" "}
                <span className="font-semibold capitalize text-slate-900">{result.label}</span>
              </p>
              <div className="text-sm text-slate-700">
                <p>Benign: {formatPercent(result.benign)}</p>
                <p>Malignant: {formatPercent(result.malignant)}</p>
              </div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
