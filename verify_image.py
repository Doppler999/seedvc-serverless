"""Build-time verification for the SeedVC baked image.

Runs inside `docker build` on a CPU-only GitHub runner, with the same offline
env the runtime uses. Every check here is a failure mode that would otherwise
show up as a cold worker that never becomes ready:

  1. no unresolved shared-object dependency in the torch CUDA wheel
     (the image uses the plain CUDA 12.1 runtime base, not the -cudnn8 one)
  2. the python dep set imports (dac, hydra, librosa, fastapi, vastai)
  3. every baked artifact resolves from the local cache with HF_HUB_OFFLINE=1 —
     i.e. the cache layout is exactly what the runtime will look for
  4. the V2 (quality) wrapper instantiates and loads its checkpoints offline
  5. the V1 (fast) wrapper instantiates and loads its checkpoints offline,
     including openai/whisper-small weights, both BigVGAN vocoders and RMVPE
  6. api_server.py imports and exposes the routes the PyWorker calls

Usage: python3 verify_image.py
"""
import os
import subprocess
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SEED_VC_DIR = os.environ.get("SEED_VC_DIR", "/workspace/seed-vc")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SEED_VC_DIR)


def log(msg):
    print(f"[verify] {msg}", flush=True)


def fail(msg):
    print(f"[verify] FAIL: {msg}", flush=True)
    raise SystemExit(1)


# ── 1. torch CUDA libs resolve ───────────────────────────────────────────────
def check_torch_libs():
    import torch

    libdir = os.path.join(os.path.dirname(torch.__file__), "lib")
    # libcuda.so.1 / libnvidia-* come from the host driver via the NVIDIA
    # container runtime, never from the image, so they are expected to be
    # unresolved at build time.
    driver_libs = ("libcuda.so", "libnvidia-")
    missing = []
    for name in sorted(os.listdir(libdir)):
        if not name.endswith(".so"):
            continue
        out = subprocess.run(["ldd", os.path.join(libdir, name)],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "not found" not in line:
                continue
            lib = line.strip().split(" ")[0]
            if lib.startswith(driver_libs):
                continue
            missing.append(f"{name}: {line.strip()}")
    if missing:
        fail("unresolved shared libraries:\n  " + "\n  ".join(missing[:20]))
    log(f"torch {torch.__version__} (cuda {torch.version.cuda}) shared libs resolve")


# ── 2. python deps ───────────────────────────────────────────────────────────
def check_imports():
    import dac.nn.quantize  # noqa: F401
    import fastapi  # noqa: F401
    import hydra  # noqa: F401
    import librosa  # noqa: F401
    import soundfile  # noqa: F401
    import torchaudio  # noqa: F401
    import transformers  # noqa: F401
    import vastai  # noqa: F401

    log(f"imports ok: dac, fastapi, hydra, librosa, soundfile, "
        f"torchaudio {torchaudio.__version__}, transformers {transformers.__version__}, vastai")


# ── 3. every artifact resolves offline ───────────────────────────────────────
def check_artifacts_offline():
    from huggingface_hub import hf_hub_download
    from prefetch_weights import ARTIFACTS, CHECKPOINT_DIR, REPO_LOCAL_CACHE_PREFIXES

    total = 0
    for repo_id, revision, filename, _expected in ARTIFACTS:
        cache_dir = CHECKPOINT_DIR if repo_id.startswith(REPO_LOCAL_CACHE_PREFIXES) else None
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        if not os.path.exists(path):
            fail(f"{repo_id}/{filename} did not resolve offline")
        total += os.path.getsize(path)
    log(f"{len(ARTIFACTS)} artifacts resolve offline, {total / 1e9:.2f} GB")


# ── 4. V2 (quality) loads offline ────────────────────────────────────────────
def check_v2():
    os.chdir(SEED_VC_DIR)
    import torch
    import yaml
    from hydra.utils import instantiate
    from omegaconf import DictConfig

    started = time.time()
    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    wrapper = instantiate(cfg)
    wrapper.load_checkpoints()
    n = sum(p.numel() for p in wrapper.parameters())
    log(f"V2 wrapper loads offline: {n / 1e6:.1f} M params in {time.time() - started:.1f}s")
    if n < 10_000_000:
        fail("V2 wrapper has suspiciously few parameters")
    del wrapper
    import gc

    gc.collect()


# ── 5. V1 (fast) loads offline ───────────────────────────────────────────────
def check_v1():
    os.chdir(SEED_VC_DIR)
    import torch
    from seed_vc_wrapper import SeedVCWrapper

    started = time.time()
    v1 = SeedVCWrapper(device=torch.device("cpu"))
    log(
        f"V1 wrapper loads offline on cpu in {time.time() - started:.1f}s: "
        f"sr={v1.sr} sr_f0={v1.sr_f0} has_f0={v1.model_f0 is not None} "
        f"bigvgan_44k={v1.bigvgan_44k_model is not None}"
    )
    if v1.sr != 22050 or v1.sr_f0 != 44100 or v1.model_f0 is None:
        fail("V1 wrapper loaded with unexpected sample rates / missing f0 model")
    del v1
    import gc

    gc.collect()


# ── 6. api_server surface ────────────────────────────────────────────────────
def check_api_server():
    os.chdir(SEED_VC_DIR)
    import api_server

    routes = {getattr(r, "path", None) for r in api_server.app.routes}
    for path in ("/health", "/convert"):
        if path not in routes:
            fail(f"api_server.py is missing route {path} (has {sorted(routes)})")
    if not hasattr(api_server, "run_v1") or not hasattr(api_server, "run_v2"):
        fail("api_server.py is missing run_v1/run_v2")
    log(f"api_server routes ok: {sorted(str(p) for p in routes if p)}")


if __name__ == "__main__":
    check_torch_libs()
    check_imports()
    check_artifacts_offline()
    check_v2()
    check_v1()
    check_api_server()
    log("image verified: both engines load from the baked weights with no network")
