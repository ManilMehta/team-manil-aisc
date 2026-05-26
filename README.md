# team-manil-aisc

Skin lesion classification project focused on benign vs malignant melanoma detection.

Dataset: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images/data

## Train the model

```bash
pip install -r requirements.txt
python train_melanoma.py
```

This saves the best checkpoint to:

```text
checkpoints/melanoma_best.keras
```

## Local CLI inference

```bash
python predict.py /absolute/path/to/image.jpg
```

## Web demo app (Vercel-ready)

A polished Next.js demo lives in:

```text
web/
```

### 1) Install and run locally

```bash
cd /Users/manilmehta/team-manil-aisc/web
npm install
npm run dev
```

### 2) Export the Keras checkpoint to browser format

The web app expects:

```text
web/public/model/model.json
web/public/model/*.bin
```

Export command:

```bash
python /Users/manilmehta/team-manil-aisc/scripts/export_tfjs.py
```

If the model files are missing, the UI shows a clear error message.

### 3) Deploy to Vercel

1. Push this repo to GitHub.
2. Import the project in Vercel and set the root directory to `web`.
3. Build command: `npm run build`
4. Output: Next.js default
5. Ensure exported model files are in `web/public/model` before deploy.

## Notes

- This demo is for education/research and is not a medical diagnosis tool.
- Preprocessing in training and inference is aligned to resize to 224x224 and divide by 255.