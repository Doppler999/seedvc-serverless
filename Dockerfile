FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Clone seed-vc repository
RUN git clone https://github.com/Plachtaa/seed-vc.git

WORKDIR /workspace/seed-vc

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn python-multipart soundfile

# Copy our custom API server
COPY api_server.py /workspace/seed-vc/api_server.py

# Expose port
EXPOSE 8080

# Start the API server
CMD ["python3", "api_server.py"]
