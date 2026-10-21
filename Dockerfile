# Coral's published Linux runtime wheel targets CPython 3.9.
FROM python:3.9-slim-bookworm

ARG WITH_VAAPI=false
ARG DEBIAN_MIRROR=https://deb.debian.org

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN sed -i "s|http://deb.debian.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libglib2.0-0 libgl1 libusb-1.0-0 ffmpeg \
    && if [ "$WITH_VAAPI" = "true" ]; then \
         apt-get install -y --no-install-recommends intel-media-va-driver; \
       fi \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/install_edgetpu.py /tmp/install_edgetpu.py
RUN python /tmp/install_edgetpu.py && ldconfig && rm /tmp/install_edgetpu.py

WORKDIR /app
COPY pyproject.toml ./
COPY human_detect ./human_detect
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --index-url "$PIP_INDEX_URL" . \
    && pip install --index-url "$PIP_INDEX_URL" \
       --extra-index-url https://google-coral.github.io/py-repo/ \
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
