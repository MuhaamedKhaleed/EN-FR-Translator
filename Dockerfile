# ---- build stage: install dependencies into a virtualenv ----
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
# CPU-only torch keeps the image small; remove the index flag for GPU images
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]"

# ---- runtime stage ----
FROM python:3.11-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    ENFR_MODEL_DIR=/models/enfr-finetuned
RUN useradd --create-home --uid 1000 app && mkdir -p /models && chown app /models
COPY --from=builder /opt/venv /opt/venv
USER app
WORKDIR /home/app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["enfr", "serve", "--host", "0.0.0.0", "--port", "8000"]
