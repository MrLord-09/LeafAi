FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-deploy.txt

COPY backend/*.py ./backend/
COPY backend/class_names.json backend/plant_detector_class_names.json ./backend/
COPY backend/model_quantized.tflite backend/plant_detector.tflite ./backend/
COPY data/disease_catalog.json ./data/disease_catalog.json
COPY frontend/ ./frontend/

EXPOSE 10000

CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
