# Coral's published Linux runtime wheel targets CPython 3.9.
FROM python:3.9-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg libglib2.0-0 libgl1 libusb-1.0-0 \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
       | gpg --dearmor -o /usr/share/keyrings/coral.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/coral.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main' \
       > /etc/apt/sources.list.d/coral.list \
    && apt-get update && apt-get install -y --no-install-recommends libedgetpu1-std \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY human_detect ./human_detect
RUN pip install . \
    && pip install --extra-index-url https://google-coral.github.io/py-repo/ \
       'tflite-runtime==2.5.0.post1'
COPY models/manifest.json ./models/manifest.json
COPY scripts/download_models.py ./scripts/download_models.py
RUN python scripts/download_models.py
RUN useradd --uid 10001 --create-home app \
    && mkdir -p /app/data && chown app:app /app/data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"
CMD ["human-detect", "serve", "--host", "0.0.0.0"]
