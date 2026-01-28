#!/bin/bash
# SeedVC Serverless Startup Script
# This script starts the API server and then the PyWorker

set -e

# Create log directory
mkdir -p /var/log/seedvc

# Clone seed-vc if not exists
if [ ! -d "/workspace/seed-vc" ]; then
    echo "Cloning seed-vc repository..."
    cd /workspace
    git clone https://github.com/Plachtaa/seed-vc.git
fi

cd /workspace/seed-vc

# Install dependencies if needed
if [ ! -f "/workspace/.deps_installed" ]; then
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
    pip install -q fastapi uvicorn python-multipart soundfile
    touch /workspace/.deps_installed
fi

# Download api_server.py from our repo
echo "Downloading api_server.py..."
curl -sL https://raw.githubusercontent.com/Doppler999/seedvc-serverless/main/api_server.py -o api_server.py

# Start the API server in background, logging to file
echo "Starting SeedVC API server..."
python3 api_server.py > /var/log/seedvc/server.log 2>&1 &

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..120}; do
    if curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 2
done

# Install PyWorker requirements (vast-sdk from GitHub for Worker/WorkerConfig)
echo "Installing PyWorker requirements..."
pip install -q git+https://github.com/vast-ai/vast-sdk.git

# Start the PyWorker (from the same directory as start.sh)
echo "Starting PyWorker..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
python3 worker.py
