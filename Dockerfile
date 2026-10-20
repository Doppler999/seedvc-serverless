# SeedVC serverless worker — baked image (scale-to-zero, no runtime downloads).
#
# The Vast workergroup boots this image and nothing else: seed-vc, torch, every
# python dep and every model weight (V1 + V2) are inside. The previous boot.sh
# installed ~2.5 GB of wheels and downloaded ~6.5 GB of weights on every cold
# host, which takes 5-7 min — longer than the autoscaler's ~470 s load window,
# so workers were killed mid-boot and respawned forever. Here a cold worker only
# pays the image pull + a ~15 s checkpoint load.
#
# Base: CUDA 12.1 runtime on Ubuntu 22.04 (python3.10). The `-cudnn8` variant of
# this base is deliberately not used: torch 2.4.0+cu121 bundles cuDNN 8.9 and
# cuBLAS in torch/lib, so the base cuDNN layer would be ~2 GB of dead weight in
# the layer that has to be pulled on every cold start. verify_image.py asserts
# that no shared object the wheel needs is unresolved.
#
# Build (CI does this on push to main, see .github/workflows/image.yml):
#   docker build -t seedvc-serverless .
#   docker run --gpus all -p 8080:8080 seedvc-serverless

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ARG SEED_VC_COMMIT=51383efd921027683c89e5348211d93ff12ac2a8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    TORCH_HOME=/opt/torch-cache

# python3.10 + the system bits boot.sh used to install: ffmpeg for pydub/mp3,
# libsndfile for soundfile, openssl+curl for the PyWorker TLS cert.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git ffmpeg libsndfile1 libgomp1 \
        curl openssl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 --version && python3 -m pip --version

# ── seed-vc @ pinned commit (no .git in the final layer) ─────────────────────
RUN mkdir -p /workspace \
    && git init -q /workspace/seed-vc \
    && cd /workspace/seed-vc \
    && git remote add origin https://github.com/Plachtaa/seed-vc.git \
    && git fetch -q --depth 1 origin "${SEED_VC_COMMIT}" \
    && git checkout -q FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${SEED_VC_COMMIT}" \
    && rm -rf .git

WORKDIR /workspace/seed-vc

# ── torch 2.4.0 / torchaudio 2.4.0 (cu121) ───────────────────────────────────
# Boot.sh's working pair. torchaudio must come from the same index or it drags
# in a PyPI torch build without CUDA.
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install "torch==2.4.0" "torchaudio==2.4.0" \
         --index-url https://download.pytorch.org/whl/cu121 \
    && python3 -c "import torch, torchaudio; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchaudio', torchaudio.__version__)"

# ── the rest of the python deps (single source of truth: requirements.txt) ───
COPY requirements.txt requirements-pyworker.txt /opt/seedvc/
RUN python3 -m pip install -r /opt/seedvc/requirements.txt \
    && python3 -c "from dac.nn.quantize import VectorQuantize; import hydra, librosa, fastapi; print('deps ok')"

# ── Vast PyWorker SDK, pinned, without its transformers~=4.52 extra ──────────
# The SDK's pyproject pins transformers~=4.52 which conflicts with seed-vc's
# 4.46.3; the Worker code never imports transformers, borb or nltk, so install
# --no-deps and add the modules it really imports (requirements-pyworker.txt).
ARG VASTAI_SDK_COMMIT=1c39c42
RUN python3 -m pip install --no-deps \
        "vastai-sdk @ git+https://github.com/vast-ai/vast-sdk.git@${VASTAI_SDK_COMMIT}" \
    && python3 -m pip install -r /opt/seedvc/requirements-pyworker.txt \
    && python3 -c "from vastai import Worker, WorkerConfig, HandlerConfig, BenchmarkConfig, LogActionConfig; print('vastai sdk ok')"

# ── bake every weight: V1 + V2 + the HF caches the wrappers read ─────────────
# Downloads into the exact cache layout the runtime looks at (repo-local
# ./checkpoints for hf_utils, $HF_HOME/hub for transformers/bigvgan), pins each
# HF repo to an immutable revision and verifies the SHA-256 of every large file.
# The app sources come first so verify_image.py can import api_server/worker.
COPY api_server.py worker.py onstart.sh /workspace/seed-vc/
COPY prefetch_weights.py verify_image.py /opt/seedvc/
RUN python3 /opt/seedvc/prefetch_weights.py \
    && du -sh /workspace/seed-vc/checkpoints /opt/hf-cache

# Fail the build — not the cold worker — if anything is missing, if the offline
# cache layout does not resolve, or if both engines cannot load from it.
RUN python3 /opt/seedvc/verify_image.py

# From here on the container must never reach the network for weights.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY api_server.py worker.py onstart.sh /workspace/seed-vc/
RUN chmod +x /workspace/seed-vc/onstart.sh \
    && echo "seed-vc@${SEED_VC_COMMIT}" > /etc/seedvc-image

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)" || exit 1

# onstart.sh: api_server on 8080 -> wait /health -> PyWorker TLS cert -> worker.py
CMD ["bash", "/workspace/seed-vc/onstart.sh"]
