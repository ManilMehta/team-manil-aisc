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

### 2) Run backend inference API

Start the Python backend from repo root:

```bash
python -m venv .venv-api
source .venv-api/bin/activate
pip install -r api/requirements.txt
uvicorn api.inference_api:app --host 0.0.0.0 --port 8000
```

The frontend calls `NEXT_PUBLIC_INFERENCE_API_URL` (defaults to
`http://127.0.0.1:8000` for local dev).

### 3) Deploy frontend to Vercel

1. Push this repo to GitHub.
2. Import the project in Vercel and set the root directory to `web`.
3. Build command: `npm run build`
4. Output: Next.js default
5. In Vercel project settings, add env var
   `NEXT_PUBLIC_INFERENCE_API_URL=https://<your-backend-domain>`.

### 4) Deploy backend API on Render

1. Push repo to GitHub.
2. In Render, click **New +** -> **Blueprint** and choose this repo.
3. Render auto-detects `render.yaml` and creates `melanoma-inference-api`.
4. Set `ALLOWED_ORIGINS` in Render to your frontend URL(s), comma-separated.
   Example:
   `https://your-app.vercel.app,https://www.yourdomain.com`
5. Deploy and verify:
   - `GET https://<render-service>.onrender.com/health`
   - `POST https://<render-service>.onrender.com/predict`
6. In Vercel env vars, set:
   `NEXT_PUBLIC_INFERENCE_API_URL=https://<render-service>.onrender.com`

### 5) Deploy backend API on Hugging Face Spaces (free option)

This repo includes a Docker Space backend in `hf_space/`.

1. Create a new Space on Hugging Face:
   - Space SDK: **Docker**
   - Visibility: your choice
2. Clone the Space repo locally and copy these folders/files into it:
   - `api/`
   - `hf_space/`
   - `checkpoints/melanoma_best.keras`
3. In the Space repo root, move `hf_space/README.md` to `README.md`.
4. Ensure the root `Dockerfile` is `hf_space/Dockerfile` contents.
5. Commit and push to the Space repo.
6. Wait for Space build to finish, then test:
   - `GET https://<space-subdomain>.hf.space/health`
   - `POST https://<space-subdomain>.hf.space/predict`
7. In Vercel env vars, set:
   `NEXT_PUBLIC_INFERENCE_API_URL=https://<space-subdomain>.hf.space`
8. Redeploy Vercel frontend.

Note: If your model file is large, use Git LFS in the Space repo for
`checkpoints/melanoma_best.keras`.

## Notes

- This demo is for education/research and is not a medical diagnosis tool.
- Preprocessing in training and inference is aligned to resize to 224x224 and divide by 255.