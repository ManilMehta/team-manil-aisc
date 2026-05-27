---
title: Melanoma Inference API
emoji: 🩺
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

FastAPI backend for benign vs malignant skin lesion inference.

Routes:

- `GET /health`
- `POST /predict` with multipart form field `file` (image upload)

