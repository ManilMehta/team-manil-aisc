import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    model: "melanoma_cnn",
    labels: ["benign", "malignant"],
    input: {
      width: 224,
      height: 224,
      channels: 3,
      normalize: "divide_by_255",
    },
  });
}

