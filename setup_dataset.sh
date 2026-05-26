#!/usr/bin/env bash
set -euo pipefail

DATASET_SLUG="hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images"
DATA_DIR="data"
ZIP_FILE="melanoma-skin-cancer-dataset-of-10000-images.zip"

echo "=== Step 1: Pre-flight checks ==="

if ! command -v kaggle &>/dev/null; then
    echo "ERROR: Kaggle CLI not found. Install with: pip install kaggle"
    exit 1
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "ERROR: Kaggle credentials not found at ~/.kaggle/kaggle.json"
    echo "  1. Go to https://www.kaggle.com/settings -> API -> Create New Token"
    echo "  2. Place the downloaded kaggle.json in ~/.kaggle/"
    echo "  3. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo "=== Step 2: Download dataset from Kaggle ==="

kaggle datasets download -d "$DATASET_SLUG" -p .
echo "Download complete."

echo "=== Step 3: Unzip into $DATA_DIR/ ==="

mkdir -p "$DATA_DIR"
unzip -o "$ZIP_FILE" -d "$DATA_DIR"
rm -f "$ZIP_FILE"
echo "Extracted and cleaned up zip file."

echo ""
echo "=== Done! ==="
echo "Dataset is in ./$DATA_DIR/ (ignored by Git via .gitignore)."
echo "Expected layout for training: data/melanoma_cancer_dataset/train/{benign,malignant} and test/{benign,malignant}"
