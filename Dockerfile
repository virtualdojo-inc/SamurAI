FROM python:3.12-slim
# git: repo sync. ffmpeg: Loom video keyframe/audio extraction (kb/ingest_loom).
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Persistent storage directory — mount a Cloud Run volume at /data
RUN mkdir -p /data
ENV SAMURAI_DATA_DIR=/data
CMD ["python", "app.py"]
