"""
Vast.ai PyWorker for SeedVC Voice Conversion
This worker integrates with Vast.ai serverless autoscaler
"""
import base64
import io
import math
import struct
import wave

from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    LogActionConfig,
)


def _tone_wav(seconds=0.75, f0=180.0, sr=22050):
    """Small speech-ish wav (harmonic stack + 3 Hz envelope), as base64."""
    n = int(seconds * sr)
    frames = bytearray()
    for i in range(n):
        t = i / sr
        env = 0.5 * (1.0 - math.cos(2.0 * math.pi * 3.0 * t))  # syllable rate
        sample = (
            math.sin(2 * math.pi * f0 * t)
            + 0.5 * math.sin(2 * math.pi * 2 * f0 * t)
            + 0.25 * math.sin(2 * math.pi * 3 * f0 * t)
        )
        frames += struct.pack("<h", int(6000 * env * sample))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))
    return base64.b64encode(buf.getvalue()).decode()


def _benchmark_payload():
    """Payload the autoscaler benchmarks /convert with.

    The benchmark POSTs this to /convert and kills the worker when every
    response is non-200 ("No successful responses from benchmark"), so it has to
    carry real audio. 8 diffusion steps keeps it to a few seconds.
    """
    return {
        "source_audio_b64": _tone_wav(0.75, 180.0),
        "target_audio_b64": _tone_wav(0.75, 240.0),
        "mode": "quality",
        "diffusion_steps": 8,
        "length_adjust": 1.0,
        "convert_style": True,
    }


worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=8080,
    model_log_file="/var/log/seedvc/server.log",
    handlers=[
        HandlerConfig(
            route="/convert",
            allow_parallel_requests=False,  # SeedVC processes one at a time
            max_queue_time=300.0,  # 5 minutes max wait
            workload_calculator=lambda payload: 100.0,  # Fixed cost per request
            benchmark_config=BenchmarkConfig(
                generator=_benchmark_payload,
                runs=1,
                concurrency=1,
            ),
        ),
        HandlerConfig(
            route="/health",
            allow_parallel_requests=True,
            max_queue_time=10.0,
            workload_calculator=lambda payload: 1.0,
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=["Models loaded successfully", "Uvicorn running on"],
        on_error=["Traceback (most recent call last):", "RuntimeError:", "CUDA out of memory"],
        on_info=["Loading Seed-VC", "Converting:"],
    ),
)

if __name__ == "__main__":
    Worker(worker_config).run()
