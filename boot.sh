#!/bin/bash
# SeedVC serverless worker boot script (Vast.ai base image, --onstart-cmd).
#
# The workergroup has no template bound and "vastai/base:latest" no longer
# exists on Docker Hub, so this script provisions everything at boot on top of
# vastai/base-image (Ubuntu 22.04, CUDA 12.1, Python 3.10):
#   * seed-vc pinned at 51383ef
#   * torch/torchaudio 2.1.0 cu121 + the small pinned dep set from our Dockerfile
#   * api_server.py and worker.py fetched from this repo's main branch
#   * model weights download on first boot (api_server blocks on load_models)
#
# Every step logs to stdout, which Vast exposes via the instance log.
set -x

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

SEED_VC_COMMIT=51383efd921027683c89e5348211d93ff12ac2a8
RAW=https://raw.githubusercontent.com/Doppler999/seedvc-serverless/main
WORK=/workspace
LOG_DIR=/var/log/seedvc

mkdir -p "$LOG_DIR" "$WORK"
# Keep everything on stdout (instance log) and in a file a later SSH can read.
exec > >(tee -a "$LOG_DIR/boot.log") 2>&1
cd "$WORK"

echo "=== [1/7] system packages ==="
apt-get update -qq
apt-get install -y -qq --no-install-recommends git ffmpeg libsndfile1 curl ca-certificates

echo "=== [2/7] seed-vc @ ${SEED_VC_COMMIT} ==="
if [ ! -f "$WORK/seed-vc/modules/v2/vc_wrapper.py" ]; then
  rm -rf "$WORK/seed-vc"
  git init -q "$WORK/seed-vc"
  cd "$WORK/seed-vc"
  git remote add origin https://github.com/Plachtaa/seed-vc.git
  git fetch -q --depth 1 origin "$SEED_VC_COMMIT"
  git checkout -q FETCH_HEAD || exit 11
  rm -rf .git
fi
cd "$WORK/seed-vc"

echo "=== [3/7] torch 2.1.0 + torchaudio 2.1.0 (cu121) ==="
python3 -m pip install -q --no-cache-dir "torch==2.1.0" "torchaudio==2.1.0" \
  --index-url https://download.pytorch.org/whl/cu121 || exit 12

echo "=== [4/7] python deps ==="
python3 -m pip install -q --no-cache-dir \
  "numpy==1.26.4" "scipy==1.13.1" "librosa==0.10.2" "pydub==0.25.1" \
  "soundfile==0.12.1" "hydra-core==1.3.2" "omegaconf==2.3.1" \
  "pyyaml==6.0.2" "einops==0.8.0" "munch==4.0.0" "transformers==4.46.3" \
  "huggingface-hub==0.28.1" "tqdm" "matplotlib" \
  "fastapi" "uvicorn" "python-multipart" || exit 13
python3 -c "import torch, torchaudio; print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available(), 'torchaudio', torchaudio.__version__)" || exit 14

echo "=== [5/7] api_server.py + worker.py from $RAW ==="
curl -fsSL "$RAW/api_server.py" -o "$WORK/seed-vc/api_server.py" || exit 15
curl -fsSL "$RAW/worker.py" -o "$WORK/worker.py" || exit 15
grep -q source_audio_b64 "$WORK/seed-vc/api_server.py" || { echo "FATAL: api_server.py from main has no JSON support"; exit 16; }
wc -l "$WORK/seed-vc/api_server.py" "$WORK/worker.py"

echo "=== [6/7] start model server (weights download on first boot) ==="
cd "$WORK/seed-vc"
if curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; then
  # Instance reboot: the model server from the previous boot is still up.
  echo "model server already healthy, reusing it"
else
  nohup python3 api_server.py > "$LOG_DIR/server.log" 2>&1 &
  API_PID=$!
  echo "api_server pid $API_PID"
  HEALTHY=0
  for i in $(seq 1 900); do
    if curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; then
      echo "model server healthy after ${i}s"
      HEALTHY=1
      break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
      echo "FATAL: api_server.py died"
      tail -120 "$LOG_DIR/server.log"
      exit 17
    fi
    sleep 1
  done
  [ "$HEALTHY" = "1" ] || { echo "FATAL: model server never became healthy"; tail -120 "$LOG_DIR/server.log"; exit 18; }
  curl -sf http://127.0.0.1:8080/health || exit 18
fi
echo

echo "=== [7/7] PyWorker ==="
# The PyWorker serves HTTPS on WORKER_PORT with a cert the Vast console signs for
# this instance; Vast's start_server.sh normally sets this up, we are not using
# it, so reproduce the three env vars + the cert it needs (VaST docs convention).
export WORKER_PORT="${WORKER_PORT:-3000}"
export USE_SSL="${USE_SSL:-true}"
# Empty UNSECURED breaks the SDK's strtobool; leave it unset instead.
if [ -z "${UNSECURED:-}" ]; then unset UNSECURED; fi
export REPORT_ADDR="${REPORT_ADDR:-https://run.vast.ai}"
echo "WORKER_PORT=$WORKER_PORT USE_SSL=$USE_SSL REPORT_ADDR=$REPORT_ADDR CONTAINER_ID=${CONTAINER_ID:-unset}"

if [ "$USE_SSL" = "true" ] && [ ! -s /etc/instance.crt ]; then
  cat > /etc/openssl-san.cnf <<'EOF'
    [req]
    default_bits       = 2048
    distinguished_name = req_distinguished_name
    req_extensions     = v3_req

    [req_distinguished_name]
    countryName         = US
    stateOrProvinceName = CA
    organizationName    = Vast.ai Inc.
    commonName          = vast.ai

    [v3_req]
    basicConstraints = CA:FALSE
    keyUsage         = nonRepudiation, digitalSignature, keyEncipherment
    subjectAltName   = @alt_names

    [alt_names]
    IP.1   = 0.0.0.0
EOF
  openssl req -newkey rsa:2048 -subj "/C=US/ST=CA/CN=pyworker.vast.ai/" \
    -nodes -sha256 -keyout /etc/instance.key -out /etc/instance.csr \
    -config /etc/openssl-san.cnf || exit 20
  curl -sS --header 'Content-Type: application/octet-stream' \
    --data-binary @/etc/instance.csr \
    -X POST "https://console.vast.ai/api/v0/sign_cert/?instance_id=${CONTAINER_ID}" > /etc/instance.crt || exit 21
  echo "cert bytes: $(wc -c < /etc/instance.crt)"
fi

mkdir -p "$WORK/pyworker"
cd "$WORK/pyworker"
python3 -m pip install -q --no-cache-dir "vastai-sdk @ git+https://github.com/vast-ai/vast-sdk.git" || exit 19
exec python3 "$WORK/worker.py"
