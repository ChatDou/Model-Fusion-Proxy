#!/bin/bash
# Model Fusion Proxy - Startup Script
# Used by launchd to start the proxy server on login

PROXY_DIR="/Users/douyuan/.gemini/antigravity-ide/scratch/model_fusion_proxy"
VENV_PYTHON="$PROXY_DIR/.venv/bin/python"
LOG_DIR="$HOME/Library/Logs"

cd "$PROXY_DIR" || exit 1

# Ensure the .env file exists (API keys)
if [ ! -f "$PROXY_DIR/.env" ]; then
    echo "$(date): ERROR - .env file not found, cannot start proxy" >> "$LOG_DIR/model-fusion-proxy.log"
    exit 1
fi

# Start the proxy server
exec "$VENV_PYTHON" -m uvicorn main:app --port 8000 --host 127.0.0.1
