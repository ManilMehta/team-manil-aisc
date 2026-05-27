FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CHECKPOINT_PATH=/app/checkpoints/melanoma_best.keras

WORKDIR /app

COPY api/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

COPY api /app/api
COPY hf_space /app/hf_space
COPY checkpoints /app/checkpoints

EXPOSE 7860

CMD ["uvicorn", "hf_space.app:app", "--host", "0.0.0.0", "--port", "7860"]

