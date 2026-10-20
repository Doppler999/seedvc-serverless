#!/bin/bash
# SeedVC serverless entrypoint for the baked image (ghcr.io/doppler999/seedvc-serverless).
#
# This is boot.sh minus every install step. The image already contains the
# pinned seed-vc checkout, torch 2.4.0+cu121, all python deps and every V1/V2
# weight, so a cold worker only has to:
#
#   1. start api_server.py on 8080 and wait for /health (the V2 wrapper loads
#      from local disk: ~10-20 s, no network)
#   2. make sure the PyWorker TLS cert exists
#   3. exec worker.py (the Vast SDK PyWorker) on $WORKER_PORT
#
# Everything is logged to stdout (the Vast instance log) and to
# /var/log/seedvc/boot.log.
set -uo pipefail

LOG_DIR=/var/log/seedvc
SEED_VC_DIR=/workspace/seed-vc
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-300}

mkdir -p "$LOG_DIR"
# tee to stdout (instance log) and to a file a later SSH can read.
exec > >(tee -a "$LOG_DIR/boot.log") 2>&1
cd "$SEED_VC_DIR" || exit 1

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# The PyWorker serves HTTPS on WORKER_PORT with a cert the Vast console signs
# for this instance. Vast's own start_server.sh sets these; we do not use it, so
# reproduce them here (same values boot.sh used).
export WORKER_PORT="${WORKER_PORT:-3000}"
export USE_SSL="${USE_SSL:-true}"
# Empty UNSECURED breaks the SDK's strtobool; leave it unset instead.
if [ -z "${UNSECURED:-}" ]; then unset UNSECURED; fi
export REPORT_ADDR="${REPORT_ADDR:-https://run.vast.ai}"

echo "=== seedvc baked image boot: $(date -u +%FT%TZ) ==="
echo "WORKER_PORT=$WORKER_PORT USE_SSL=$USE_SSL REPORT_ADDR=$REPORT_ADDR CONTAINER_ID=${CONTAINER_ID:-unset}"
echo "image: $(cat /etc/seedvc-image 2>/dev/null || echo 'seedvc-serverless:latest')"

T0=$(date +%s)

# ── model server ─────────────────────────────────────────────────────────────
if curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; then
  # Instance reboot: the model server from the previous boot is still up.
  echo "model server already healthy, reusing it"
else
  nohup python3 api_server.py > "$LOG_DIR/server.log" 2>&1 &
  API_PID=$!
  echo "api_server pid $API_PID"
  HEALTHY=0
  for i in $(seq 1 "$HEALTH_TIMEOUT"); do
    if curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; then
      HEALTHY=1
      echo "model server healthy after ${i}s"
      break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
      echo "FATAL: api_server.py died"
      tail -120 "$LOG_DIR/server.log"
      exit 17
    fi
    sleep 1
  done
  if [ "$HEALTHY" != "1" ]; then
    echo "FATAL: model server never became healthy in ${HEALTH_TIMEOUT}s"
    tail -120 "$LOG_DIR/server.log"
    exit 18
  fi
  curl -sf http://127.0.0.1:8080/health || exit 18
  echo
fi
echo "MODEL_SERVER_READY_SECONDS=$(( $(date +%s) - T0 ))"

# ── PyWorker TLS cert ────────────────────────────────────────────────────────
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

# ── PyWorker (blocks for the life of the instance) ───────────────────────────
echo "BOOT_TO_WORKER_SECONDS=$(( $(date +%s) - T0 ))"
echo "=== starting PyWorker on port $WORKER_PORT ==="
exec python3 "$SEED_VC_DIR/worker.py"
