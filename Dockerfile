# WorldStreet Vivid voice bot: FastAPI signaling + Pipecat pipeline.
# Pinned to 3.12 — Pipecat/aiortc wheels lag on 3.13.
FROM python:3.12-slim

# Runtime libs:
#   libgomp1   -> onnxruntime (Silero VAD)
#   libopus0   -> aiortc audio codec
#   ffmpeg     -> PyAV / media handling
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libopus0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BOT_PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Daily transport (Linux-only wheels), used when TRANSPORT=daily. Kept out of
# requirements.txt so local Windows `pip install` still works (no daily wheels).
RUN pip install --no-cache-dir "pipecat-ai[daily]==1.2.1"

COPY . .

EXPOSE 7860

CMD ["python", "server.py"]
