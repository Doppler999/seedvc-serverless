# SeedVC Serverless

Docker image for Seed-VC voice conversion API, optimized for Vast.ai Serverless.

## Features

- FastAPI server with `/convert` and `/health` endpoints
- Pre-installed Seed-VC V2 model
- CUDA-enabled for GPU acceleration

## Usage

```bash
docker pull ghcr.io/doppler999/seedvc-serverless:latest
docker run -p 8080:8080 --gpus all ghcr.io/doppler999/seedvc-serverless:latest
```

## API Endpoints

### Health Check
```
GET /health
```

### Voice Conversion
```
POST /convert
Content-Type: multipart/form-data

- source_audio: Audio file (wav)
- target_audio: Target voice audio file (wav)
- diffusion_steps: int (default: 30)
- length_adjust: float (default: 1.0)
- convert_style: bool (default: true)
```

## Building

The image is automatically built and pushed to GitHub Container Registry on every push to main.
