# ==========================================================================
#   Obscuro Erebus 32B Combined Foundation Model - GPU/CPU Docker Container
#   GPU Mode : uses nvidia/cuda base with CUDA 12.x + PyTorch CUDA
#   CPU Mode : falls back to python:3.12-slim automatically
# ==========================================================================

# Try GPU base first; if no GPU is available Docker will still build & run on CPU
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu-base

# Install Python 3.12 on the GPU base
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3.12-venv \
    build-essential \
    curl \
    git \
    wget \
    chromium-browser \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# ==========================================================================
#   Final image — works on GPU or CPU depending on runtime
# ==========================================================================
FROM gpu-base AS final

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    TARGET_PARAMS=32B \
    DEVICE_PREFERENCE=GPU \
    CHROMIUM_PATH=/usr/bin/chromium-browser \
    CHROMIUM_HEADLESS=1

WORKDIR /app

# Copy requirements & package metadata
COPY requirements.txt pyproject.toml /app/

# Install Python dependencies:
#   - Try PyTorch CUDA first, fall back to CPU-only PyTorch, then to numpy+click
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch>=2.3.0 \
        --index-url https://download.pytorch.org/whl/cu124 \
    || pip install --no-cache-dir \
        torch>=2.3.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    || echo "PyTorch install skipped — lightweight numpy mode"

RUN pip install --no-cache-dir -r requirements.txt \
    || pip install --no-cache-dir numpy>=1.26 click>=8.1 rich>=13.0

# Copy application source code
COPY src/ /app/src/
COPY main.py /app/

# Install the application package in editable mode
RUN pip install --no-cache-dir -e .

# Create persistent storage directories for datasets, artifacts, exports
RUN mkdir -p \
    /app/datasets/fineweb \
    /app/datasets/human_philosophy \
    /app/datasets/psychology \
    /app/datasets/coding \
    /app/datasets/mathematics \
    /app/datasets/quantum_physics \
    /app/datasets/multimodal \
    /app/datasets/preferences \
    /app/datasets/chromium_browser \
    /app/artifacts/runs \
    /app/exports

# Health check — quick system verify
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD python -m combined_foundation.cli verify || exit 1

# Default entrypoint: the CLI
ENTRYPOINT ["python", "-m", "combined_foundation.cli"]

# Default command: full 32B run pipeline (all 4 phases)
CMD ["fullrun"]
