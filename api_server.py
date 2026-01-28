"""
FastAPI server for Seed-VC voice conversion
Production-ready API for Vast.ai Serverless
"""
import os
import tempfile
import torch
import yaml
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
dtype = torch.float16

app = FastAPI(title="Seed-VC API", version="1.0.0")

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

@app.post("/convert")
async def convert_voice(
    source_audio: UploadFile = File(...),
    target_audio: UploadFile = File(...),
    diffusion_steps: int = Form(30),
    length_adjust: float = Form(1.0),
    convert_style: bool = Form(True),
):
    if vc_wrapper_v2 is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        src_path = tempfile.mktemp(suffix=".wav")
        tgt_path = tempfile.mktemp(suffix=".wav")
        
        with open(src_path, "wb") as f:
            f.write(await source_audio.read())
        with open(tgt_path, "wb") as f:
            f.write(await target_audio.read())
        
        print(f"Converting: {src_path} -> {tgt_path}")
        
        full_audio = None
        for mp3_bytes, audio_result in vc_wrapper_v2.convert_voice_with_streaming(
            source_audio_path=src_path,
            target_audio_path=tgt_path,
            diffusion_steps=min(diffusion_steps, 50),
            length_adjust=length_adjust,
            intelligebility_cfg_rate=0.7,
            similarity_cfg_rate=0.7,
            top_p=0.7,
            temperature=0.7,
            repetition_penalty=1.5,
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
        import soundfile as sf
        
        if isinstance(full_audio, tuple):
            sr, audio = full_audio
        else:
            sr, audio = 22050, full_audio
        
        output_path = tempfile.mktemp(suffix=".wav")
        if isinstance(audio, np.ndarray):
            sf.write(output_path, audio, sr)
        elif hasattr(audio, "cpu"):
            sf.write(output_path, audio.cpu().numpy(), sr)
        else:
            sf.write(output_path, np.array(audio), sr)
        
        os.unlink(src_path)
        os.unlink(tgt_path)
        
        print(f"Conversion complete: {output_path}")
        
        return FileResponse(output_path, media_type="audio/wav", filename="converted.wav")
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
