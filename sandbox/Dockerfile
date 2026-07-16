# samurai-sandbox — minimal, zero-dependency-beyond-aiohttp executor image.
# Deliberately NO google-cloud / db / secret libraries: this container holds no
# credentials and needs no cloud SDKs. Runs as a non-root user.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
# Curated pure-Python deps the isolated child may import (copied into each run's
# workdir at exec time). Keep minimal — see _VENDOR_DIR in app.py.
COPY vendor/ ./vendor/

# Drop root.
RUN useradd --create-home --uid 10001 sandboxer
USER sandboxer

EXPOSE 8080
CMD ["python", "app.py"]
