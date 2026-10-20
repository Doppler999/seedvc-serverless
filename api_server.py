"""
FastAPI server for Seed-VC voice conversion
Production-ready API for Vast.ai Serverless

Two engines, dispatched on `mode` (this mirrors the old always-on RunPod
server.py that the app was built against):

  * mode="fast"    -> Seed-VC V1 (seed_vc_wrapper.SeedVCWrapper at the pinned
                      upstream commit): non-autoregressive DiT + length
                      regulator, so it preserves timing (lip sync) and pitch.
                      22.05 kHz base model, 44.1 kHz when f0_conditioned=true
                      (the f0-conditioned singing model).
  * mode="quality" -> Seed-VC V2 (configs/v2/vc_wrapper.yaml): 22.05 kHz,
                      CFM + AR, style conversion off unless asked.

V1 is loaded lazily on the first fast request; V2 loads at startup.

Endpoints
---------
GET  /health   -> liveness + model state
POST /convert  -> voice conversion, two accepted encodings:

  * multipart/form-data (legacy browser/app path):
        source_audio=<file> target_audio=<file>
        mode=fast diffusion_steps=10 convert_style=false
      returns the converted wav as audio/wav (FileResponse).
  * application/json (Vast.ai PyWorker path, and any base64 client):
        {"source_audio_b64": "...", "target_audio_b64": "...",
         "mode": "quality", "diffusion_steps": 50, ...}
      returns {"audio": "<base64 wav>", "duration_s": 3.2, "sample_rate": 22050,
      "mode": "quality"}.

Both paths call the same `run_conversion`, so dispatch and peak_trim are
identical for JSON and multipart.
"""
import base64
import binascii
import os
import tempfile
import threading
import time
import torch
import yaml
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
dtype = torch.float16

# V2's BigVGAN has no output normalisation: every 22.05 kHz output peaks at
# 0.0 dBFS (1-6 clipped samples on speech, 757 on singing). Trim any hot output
# to -0.3 dBFS before returning it. 0.9664 in the brief is the 4-dp rounding of
# -0.3 dBFS; the exact constant is 10 ** (-0.3 / 20) = 0.96605.
TARGET_PEAK_DBFS = -0.3
TARGET_PEAK = 10.0 ** (TARGET_PEAK_DBFS / 20.0)
HOT_PEAK_THRESHOLD = 0.99

# Engine defaults. Fast is V1's trained default (10 steps); quality keeps the
# 50 steps the app sends and the previous server default.
FAST_DEFAULT_STEPS = 10
QUALITY_DEFAULT_STEPS = 50


def peak_trim(audio):
    """Scale float audio down to TARGET_PEAK when it peaks above the threshold.

    Returns (audio, peak_before, peak_after). Non-hot audio is returned
    unchanged (the 44.1 kHz V1 model is already clean at ~-1.1 dBFS, so this
    is a no-op there).
    """
    import numpy as np

    arr = np.asarray(audio)
    if arr.size == 0:
        return arr, 0.0, 0.0
    peak_before = float(np.max(np.abs(arr)))
    if peak_before <= HOT_PEAK_THRESHOLD:
        return arr, peak_before, peak_before
    arr = (arr.astype(np.float32) * (TARGET_PEAK / peak_before)).astype(arr.dtype)
    peak_after = float(np.max(np.abs(arr)))
    return arr, peak_before, peak_after


def _as_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_int(value, default):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _decode_b64_audio(value, field):
    """Decode a base64 string (optionally a data: URL) into wav bytes."""
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=422, detail=f"{field} is required")
    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    try:
        raw = base64.b64decode(value, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"{field} is not valid base64: {exc}")
    if len(raw) < 44:
        raise HTTPException(status_code=422, detail=f"{field} decoded to {len(raw)} bytes, too small for audio")
    return raw


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
vc_wrapper_v2 = None  # V2: voice & style conversion (quality)
vc_wrapper_v1 = None  # V1: voice & singing conversion (fast)
_v1_lock = threading.Lock()
_load_time_v2 = None
_load_time_v1 = None


def load_models():
    """Load the V2 (quality) engine at startup, as before."""
    global vc_wrapper_v2, _load_time_v2
    print(f"Loading Seed-VC V2 models on {device}...", flush=True)
    started = time.time()
    from hydra.utils import instantiate
    from omegaconf import DictConfig

    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper_v2 = instantiate(cfg)
    vc_wrapper_v2.load_checkpoints()
    vc_wrapper_v2.to(device)
    vc_wrapper_v2.eval()
    vc_wrapper_v2.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)
    _load_time_v2 = round(time.time() - started, 2)
    print(f"Models loaded successfully! (V2 in {_load_time_v2}s)", flush=True)


def load_v1():
    """Load V1 (fast) on first use.

    V1 pulls its own DiT checkpoints, both BigVGAN vocoders (22.05 kHz and
    44.1 kHz), CAMPPlus and RMVPE: ~1.9 GB of weights, so it stays out of the
    startup path and out of VRAM until a fast request arrives.
    """
    global vc_wrapper_v1, _load_time_v1
    if vc_wrapper_v1 is not None:
        return vc_wrapper_v1
    with _v1_lock:
        if vc_wrapper_v1 is not None:
            return vc_wrapper_v1
        started = time.time()
        print("Loading Seed-VC V1 (fast) models...", flush=True)
        from seed_vc_wrapper import SeedVCWrapper

        vc_wrapper_v1 = SeedVCWrapper(device=device)
        _load_time_v1 = round(time.time() - started, 2)
        print(f"V1 models loaded in {_load_time_v1}s", flush=True)
        return vc_wrapper_v1


def run_v1(
    src_path,
    tgt_path,
    diffusion_steps,
    length_adjust,
    inference_cfg_rate,
    auto_f0_adjust,
    pitch_shift,
    f0_conditioned,
):
    """Run Seed-VC V1. Returns (samples, sample_rate).

    Upstream (pinned 51383ef) exposes V1 as a generator that yields
    (mp3_bytes, (sr, audio)) and only fills the payload on the last chunk; we
    keep the last non-None payload, which is the whole converted utterance.
    """
    wrapper = load_v1()
    print(
        f"V1 fast: steps={diffusion_steps} cfg={inference_cfg_rate} "
        f"f0_condition={f0_conditioned} auto_f0_adjust={auto_f0_adjust} "
        f"pitch_shift={pitch_shift} length_adjust={length_adjust}",
        flush=True,
    )

    full_audio = None
    for _mp3_bytes, payload in wrapper.convert_voice(
        src_path,
        tgt_path,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        f0_condition=f0_conditioned,
        auto_f0_adjust=auto_f0_adjust,
        pitch_shift=pitch_shift,
        stream_output=True,
    ):
        if payload is not None:
            full_audio = payload

    if full_audio is None:
        raise HTTPException(status_code=500, detail="V1 conversion returned no audio")

    import numpy as np

    sr, audio = full_audio
    if isinstance(audio, np.ndarray):
        samples = audio
    elif hasattr(audio, "cpu"):
        samples = audio.detach().cpu().numpy()
    else:
        samples = np.array(audio)
    if samples.ndim == 2 and samples.shape[0] == 1:
        samples = samples[0]
    return samples, int(sr)


def run_v2(
    src_path,
    tgt_path,
    diffusion_steps,
    length_adjust,
    convert_style,
    intelligibility_cfg_rate,
    similarity_cfg_rate,
    top_p,
    temperature,
    repetition_penalty,
):
    """Run Seed-VC V2 exactly as before. Returns (samples, sample_rate)."""
    print(
        f"V2 quality: steps={diffusion_steps} length_adjust={length_adjust} "
        f"convert_style={convert_style}",
        flush=True,
    )

    full_audio = None
    for mp3_bytes, audio_result in vc_wrapper_v2.convert_voice_with_streaming(
        source_audio_path=src_path,
        target_audio_path=tgt_path,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        intelligebility_cfg_rate=intelligibility_cfg_rate,
        similarity_cfg_rate=similarity_cfg_rate,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        convert_style=convert_style,
        anonymization_only=False,
        device=device,
        dtype=dtype,
        stream_output=True,
    ):
        full_audio = audio_result

    if full_audio is None:
        raise HTTPException(status_code=500, detail="Conversion returned no audio")

    import numpy as np

    if isinstance(full_audio, tuple):
        sr, audio = full_audio
    else:
        sr, audio = 22050, full_audio

    if isinstance(audio, np.ndarray):
        samples = audio
    elif hasattr(audio, "cpu"):
        samples = audio.detach().cpu().numpy()
    else:
        samples = np.array(audio)
    if samples.ndim == 2 and samples.shape[0] == 1:
        samples = samples[0]
    return samples, sr


def run_conversion(
    src_path,
    tgt_path,
    mode="quality",
    diffusion_steps=None,
    length_adjust=1.0,
    convert_style=False,
    intelligibility_cfg_rate=0.7,
    similarity_cfg_rate=0.7,
    top_p=0.7,
    temperature=0.7,
    repetition_penalty=1.5,
    inference_cfg_rate=0.7,
    auto_f0_adjust=True,
    pitch_shift=0,
    f0_conditioned=False,
):
    """Dispatch on mode, then apply the same peak_trim to either engine."""
    import numpy as np

    mode = (mode or "quality").strip().lower()
    if diffusion_steps is None:
        diffusion_steps = FAST_DEFAULT_STEPS if mode == "fast" else QUALITY_DEFAULT_STEPS
    engine = "v1" if mode == "fast" else "v2"

    if engine == "v1":
        samples, sr = run_v1(
            src_path,
            tgt_path,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            inference_cfg_rate=inference_cfg_rate,
            auto_f0_adjust=auto_f0_adjust,
            pitch_shift=pitch_shift,
            f0_conditioned=f0_conditioned,
        )
    else:
        samples, sr = run_v2(
            src_path,
            tgt_path,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            convert_style=convert_style,
            intelligibility_cfg_rate=intelligibility_cfg_rate,
            similarity_cfg_rate=similarity_cfg_rate,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

    samples, peak_before, peak_after = peak_trim(samples)
    db_before = 20.0 * np.log10(peak_before) if peak_before > 0 else float("-inf")
    db_after = 20.0 * np.log10(peak_after) if peak_after > 0 else float("-inf")
    print(
        f"[{engine}] mode={mode} steps={diffusion_steps} sr={sr} "
        f"peak {db_before:.3f} dBFS -> {db_after:.3f} dBFS",
        flush=True,
    )
    return samples, sr, engine


def write_wav(samples, sr):
    import soundfile as sf

    fd, output_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(output_path, samples, sr)
    return output_path


app = FastAPI(title="Seed-VC API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    load_models()
    # Warm V1 in the background: it is ~1.9 GB of weights and should not delay
    # /health (the autoscaler gates worker readiness on it), but it also should
    # not be paid by the first user's fast request. A failure here only logs;
    # run_v1() will retry the load synchronously.
    def _warm_v1():
        try:
            load_v1()
        except Exception:
            import traceback
            print("WARNING: V1 warm-up failed; will retry on the first fast request", flush=True)
            traceback.print_exc()

    threading.Thread(target=_warm_v1, name="v1-warmup", daemon=True).start()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": vc_wrapper_v2 is not None,
        "v1_loaded": vc_wrapper_v1 is not None,
        "device": str(device),
        "load_time_s": {"v2": _load_time_v2, "v1": _load_time_v1},
    }


def _params_from_mapping(get):
    """Shared param parsing for both encodings (form and JSON)."""
    mode = str(get("mode") or "quality").strip().lower()
    return {
        "mode": mode,
        "diffusion_steps": _as_int(get("diffusion_steps"), None),
        "length_adjust": _as_float(get("length_adjust"), 1.0),
        # Fast ignores this; Quality must not run style conversion by default.
        "convert_style": _as_bool(get("convert_style"), False),
        "inference_cfg_rate": _as_float(get("inference_cfg_rate"), 0.7),
        "auto_f0_adjust": _as_bool(get("auto_f0_adjust"), True),
        "pitch_shift": _as_int(get("pitch_shift"), 0),
        "f0_conditioned": _as_bool(get("f0_conditioned"), False),
        "intelligibility_cfg_rate": _as_float(get("intelligibility_cfg_rate"), 0.7),
        "similarity_cfg_rate": _as_float(get("similarity_cfg_rate"), 0.7),
        "top_p": _as_float(get("top_p"), 0.7),
        "temperature": _as_float(get("temperature"), 0.7),
        "repetition_penalty": _as_float(get("repetition_penalty"), 1.5),
    }


async def _convert_multipart(request: Request):
    """Legacy multipart path — wav file back."""
    form = await request.form()
    source_audio = form.get("source_audio")
    target_audio = form.get("target_audio")

    if source_audio is None or target_audio is None:
        raise HTTPException(
            status_code=422,
            detail="source_audio and target_audio files are required",
        )
    if not hasattr(source_audio, "read") or not hasattr(target_audio, "read"):
        raise HTTPException(
            status_code=422,
            detail="source_audio and target_audio must be uploaded as files",
        )

    params = _params_from_mapping(form.get)

    src_path = tempfile.mktemp(suffix=".wav")
    tgt_path = tempfile.mktemp(suffix=".wav")
    output_path = None
    try:
        with open(src_path, "wb") as f:
            f.write(await source_audio.read())
        with open(tgt_path, "wb") as f:
            f.write(await target_audio.read())

        samples, sr, engine = run_conversion(src_path, tgt_path, **params)
        output_path = write_wav(samples, sr)
        print(f"Conversion complete: {output_path}", flush=True)
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="converted.wav",
            headers={
                "X-Mode": params["mode"],
                "X-Engine": engine,
                "X-Sample-Rate": str(sr),
                "X-Diffusion-Steps": str(
                    params["diffusion_steps"]
                    if params["diffusion_steps"] is not None
                    else (FAST_DEFAULT_STEPS if engine == "v1" else QUALITY_DEFAULT_STEPS)
                ),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in (src_path, tgt_path):
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass


async def _convert_json(request: Request):
    """JSON path used by the Vast PyWorker: base64 in, base64 out."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Body is not valid JSON")

    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="JSON body must be an object")

    # The PyWorker posts `payload` verbatim, but accept a wrapped payload too.
    params = body
    if "payload" in body and isinstance(body["payload"], dict) and not (
        "source_audio_b64" in body or "source_audio" in body
    ):
        params = body["payload"]

    source_raw = _decode_b64_audio(
        params.get("source_audio_b64", params.get("source_audio")), "source_audio_b64"
    )
    target_raw = _decode_b64_audio(
        params.get("target_audio_b64", params.get("target_audio")), "target_audio_b64"
    )

    parsed = _params_from_mapping(params.get)

    src_path = tempfile.mktemp(suffix=".wav")
    tgt_path = tempfile.mktemp(suffix=".wav")
    try:
        with open(src_path, "wb") as f:
            f.write(source_raw)
        with open(tgt_path, "wb") as f:
            f.write(target_raw)

        samples, sr, engine = run_conversion(src_path, tgt_path, **parsed)

        import numpy as np
        import soundfile as sf

        # Encode in memory — no temp output file, nothing to clean up.
        import io
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV")
        wav_bytes = buf.getvalue()
        duration_s = round(float(np.asarray(samples).shape[-1]) / float(sr), 3)
        print(
            f"Conversion complete (json): {len(wav_bytes)} bytes, {duration_s}s, "
            f"mode={parsed['mode']} engine={engine}",
            flush=True,
        )
        return JSONResponse(
            {
                "audio": base64.b64encode(wav_bytes).decode("ascii"),
                "duration_s": duration_s,
                "sample_rate": sr,
                "mode": parsed["mode"],
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in (src_path, tgt_path):
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass


@app.post("/convert")
async def convert_voice(request: Request):
    if vc_wrapper_v2 is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        return await _convert_json(request)
    return await _convert_multipart(request)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
