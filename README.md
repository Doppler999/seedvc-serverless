# SeedVC Serverless

Baked Docker image for the Seed-VC voice-conversion API (scale-to-zero on
Vast.ai serverless). Nothing is installed or downloaded at boot: a cold worker
starts `api_server.py`, waits for `/health` and runs the PyWorker.

## Why it is baked

`boot.sh` (still in this repo as the rollback path) installed torch + pip deps,
cloned seed-vc and pulled ~6.5 GB of weights on every fresh host. That takes
5-7 minutes on a Vast host — longer than the autoscaler's ~470 s load window, so
every worker was killed mid-boot and replaced, burning credit without ever
serving a request. With the image baked, a cold worker pays only the image pull
and a ~15 s checkpoint load.

## What the image contains

- seed-vc pinned at `51383efd921027683c89e5348211d93ff12ac2a8` (no `.git` layer)
- CUDA 12.1 runtime base (Ubuntu 22.04, python3.10) + torch/torchaudio 2.4.0
  cu121. The base is deliberately the plain `runtime` flavour: torch's cu121
  wheel ships cuDNN 8.9 and cuBLAS itself, so `-cudnn8` would only add ~2 GB to
  the layer every cold start has to pull.
- the dep set from `requirements.txt`, including `descript-audio-codec==1.0.0`
  (V1's DiT build imports `dac.nn.quantize`) and the Vast PyWorker SDK
- every weight, checksum-pinned at build time by `prefetch_weights.py`:
  - V2 (quality): `v2/cfm_small.pth`, `v2/ar_base.pth`, ASTRAL
    `bsq32_light.pth` + `bsq2048_light.pth`, campplus, BigVGAN 22 kHz,
    hubert-large-ll60k, whisper-small tokenizer
  - V1 (fast): the 22.05 kHz base DiT, the 44.1 kHz f0 DiT, both BigVGAN
    vocoders (22 + 44 kHz), RMVPE and the whisper-small weights its
    `WhisperModel.from_pretrained` needs
- runtime runs with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`: a missing
  artifact fails at `docker build` (see `verify_image.py`) instead of on a cold
  user

`verify_image.py` runs inside the build and fails it unless every artifact
resolves offline and both engines (V1 + V2) load from the baked cache on CPU.

## Measured (2026-10-20, digest `sha256:3186c52a…`)

- image: 9.13 GB compressed / 22 layers; weights 5.2 GB (2.2 GB repo-local
  checkpoints + 3.0 GB HF cache)
- container start → worker ready (image already on the host): **14-20 s**
  (`loadtime` reported by the PyWorker; model server healthy after 15-16 s)
- first pull on a virgin host: **114 s** (RTX PRO 5000) to **542 s** (slow
  RTX 4090) — the pull, not the boot, is what a cold request waits for
- app route (`/api/audio/voice-change-seedvc`, 1 s wav): 200 in **8.4-8.6 s**
  (fast and quality, warm worker on a 5 Gbps host), 50.5 s when the worker has
  to restart from the exited state, and a `VOICE_ENGINE_WARMING` 503 when the
  request lands on a host that still has to pull the image (>240 s)

### GPU constraint (important)

torch 2.4.0+cu121 ships kernels up to `sm_90`. On Blackwell (`sm_120`: RTX PRO
5000/6000, RTX 50xx, B200) every kernel launch fails with "no kernel image is
available for execution on the device", the PyWorker's benchmark gets a 500 and
the autoscaler kills the worker — exactly the churn this image is meant to stop.
The workergroup search therefore pins pre-Blackwell GPUs, and also requires
`inet_down>=2000` so a virgin pull is not minutes long:

```
gpu_ram>=16 num_gpus=1 rentable=true verified=true rented=false inet_down>=2000
gpu_name in [RTX_4090,L40,L40S,A100,RTX_A6000,RTX_3090,RTX_6000Ada,H100]
```


## Build

CI builds and pushes on every push to `main` (`.github/workflows/image.yml`):

```
ghcr.io/doppler999/seedvc-serverless:latest   # + :<git sha>
```

Locally:

```bash
docker build -t seedvc-serverless .
docker run --gpus all -p 8080:8080 seedvc-serverless
```

### Pull requirement

The Vast workergroup pulls this image **anonymously** — the GHCR package must be
public (Package settings → Change visibility → Public). If it is private, Vast
fails the worker with an image-pull error; there is no registry credential in
the workergroup launch args.

## Runtime

`onstart.sh` is the entrypoint (also used as `--onstart-cmd`):

1. `WORKER_PORT` / `USE_SSL` / `REPORT_ADDR` exported (same defaults as boot.sh)
2. `api_server.py` on `8080`, then wait for `/health` (logs
   `MODEL_SERVER_READY_SECONDS`)
3. request the instance TLS cert from `console.vast.ai` if `/etc/instance.crt`
   is missing
4. `exec worker.py` — the Vast PyWorker on `WORKER_PORT`

Vast workergroup launch args:

```
--image ghcr.io/doppler999/seedvc-serverless:latest --disk 40 \
  --onstart-cmd "bash /workspace/seed-vc/onstart.sh"
```

## API

### Health check
```
GET /health
```
`/health` only answers after FastAPI's startup hook has loaded the models.

### Voice conversion
```
POST /convert
```

- JSON (`application/json`) — the PyWorker path:
  `{"source_audio_b64": "...", "target_audio_b64": "...", "mode": "fast"|"quality",
    "diffusion_steps": 10, "length_adjust": 1.0, "convert_style": false,
    "f0_conditioned": false}` → `{"audio": "<base64 wav>", "duration_s": ...,
    "sample_rate": ..., "mode": ...}`
- multipart (`multipart/form-data`) — app/browser path:
  `source_audio`, `target_audio` files + the same scalar fields, returns
  `audio/wav`

`mode=fast` is Seed-VC V1 (22.05 kHz, 44.1 kHz with `f0_conditioned=true`),
`mode=quality` is Seed-VC V2 (22.05 kHz). Output is peak-trimmed to -0.3 dBFS
when it comes out hot; clean output is returned untouched.

## Contract note

This server has **no auth**. Put it behind a token-checking proxy / the
serverless provider's endpoint auth; never expose it raw.
