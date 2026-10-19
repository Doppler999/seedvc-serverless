"""
FastAPI server for Seed-VC voice conversion
Production-ready API for Vast.ai Serverless

Endpoints
---------
GET  /health   -> liveness + model state
POST /convert  -> voice conversion, two accepted encodings:

  * multipart/form-data (legacy browser/app path):
        source_audio=<file> target_audio=<file>
        diffusion_steps=50 length_adjust=1.0 convert_style=true
      returns the converted wav as audio/wav (FileResponse).

  * application/json (Vast.ai PyWorker path, and any base64 client):
        {"source_audio_b64": "...", "target_audio_b64": "...",
         "mode": "quality", "diffusion_steps": 50, ...}
      returns {"audio": "<base64 wav>", "duration_s": 3.2, "sample_rate": 22050}.
      The PyWorker forwards the client's `payload` object verbatim as this
      JSON body, so the JSON shape is what the app route sends.

Both paths run the exact same inference and the same peak_trim.
"""
import base64
import binascii
import os
import tempfile
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


def peak_trim(audio):
    """Scale float audio down to TARGET_PEAK when it peaks above the threshold.

    Returns (audio, peak_before, peak_after). Non-hot audio is returned
    unchanged (the 44.1 kHz model is already clean at ~-1.1 dBFS).
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


def run_conversion(
    src_path,
    tgt_path,
    diffusion_steps=50,
    length_adjust=1.0,
    convert_style=True,
    intelligibility_cfg_rate=0.7,
    similarity_cfg_rate=0.7,
    top_p=0.7,
    temperature=0.7,
    repetition_penalty=1.5,
):
    """Run SeedVC V2 on two wav paths. Returns (samples, sample_rate)."""
    print(
        f"Converting: {src_path} -> {tgt_path} (diffusion_steps={diffusion_steps}, "
        f"length_adjust={length_adjust}, convert_style={convert_style})",
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

    # Server-side level trim: V2 always saturates at 0.0 dBFS.
    samples, peak_before, peak_after = peak_trim(samples)
    db_before = 20.0 * np.log10(peak_before) if peak_before > 0 else float("-inf")
    db_after = 20.0 * np.log10(peak_after) if peak_after > 0 else float("-inf")
    print(
        f"Output peak: {db_before:.3f} dBFS -> {db_after:.3f} dBFS "
        f"(target {TARGET_PEAK_DBFS} dBFS)"
    )
    return samples, sr


def write_wav(samples, sr):
    import soundfile as sf

    fd, output_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(output_path, samples, sr)
    return output_path


app = FastAPI(title="Seed-VC API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vc_wrapper_v2 = None

def load_models():
    global vc_wrapper_v2
    print(f"Loading Seed-VC V2 models on {device}...")
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper_v2 = instantiate(cfg)
    vc_wrapper_v2.load_checkpoints()
    vc_wrapper_v2.to(device)
    vc_wrapper_v2.eval()
    vc_wrapper_v2.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)
    print("Models loaded successfully!")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": vc_wrapper_v2 is not None, "device": str(device)}


async def _convert_multipart(request: Request):
    """Legacy multipart path — unchanged semantics, wav file back."""
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

    diffusion_steps = _as_int(form.get("diffusion_steps"), 50)
    length_adjust = _as_float(form.get("length_adjust"), 1.0)
    convert_style = _as_bool(form.get("convert_style"), True)

    src_path = tempfile.mktemp(suffix=".wav")
    tgt_path = tempfile.mktemp(suffix=".wav")
    output_path = None
    try:
        with open(src_path, "wb") as f:
            f.write(await source_audio.read())
        with open(tgt_path, "wb") as f:
            f.write(await target_audio.read())

        samples, sr = run_conversion(
            src_path,
            tgt_path,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            convert_style=convert_style,
        )
        output_path = write_wav(samples, sr)
        print(f"Conversion complete: {output_path}")
        return FileResponse(output_path, media_type="audio/wav", filename="converted.wav")
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

    mode = str(params.get("mode") or "quality")
    diffusion_steps = _as_int(params.get("diffusion_steps"), 50)
    length_adjust = _as_float(params.get("length_adjust"), 1.0)
    convert_style = _as_bool(params.get("convert_style"), True)
    intelligibility_cfg_rate = _as_float(params.get("intelligibility_cfg_rate"), 0.7)
    similarity_cfg_rate = _as_float(params.get("similarity_cfg_rate"), 0.7)
    top_p = _as_float(params.get("top_p"), 0.7)
    temperature = _as_float(params.get("temperature"), 0.7)
    repetition_penalty = _as_float(params.get("repetition_penalty"), 1.5)

    src_path = tempfile.mktemp(suffix=".wav")
    tgt_path = tempfile.mktemp(suffix=".wav")
    try:
        with open(src_path, "wb") as f:
            f.write(source_raw)
        with open(tgt_path, "wb") as f:
            f.write(target_raw)

        samples, sr = run_conversion(
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

        import numpy as np
        import soundfile as sf

        # Encode in memory — no temp output file, nothing to clean up.
        import io
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV")
        wav_bytes = buf.getvalue()
        duration_s = round(float(np.asarray(samples).shape[-1]) / float(sr), 3)
        print(
            f"Conversion complete (json): {len(wav_bytes)} bytes, {duration_s}s, mode={mode}",
            flush=True,
        )
        return JSONResponse(
            {
                "audio": base64.b64encode(wav_bytes).decode("ascii"),
                "duration_s": duration_s,
                "sample_rate": sr,
                "mode": mode,
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
