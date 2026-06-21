#!/bin/bash
# ──────────────────────────────────────────────────────────────────
# MLX Model Server — Startup Script
# Used by launchd to start the MLX inference server on login.
#
# Warm-up note: the first request after launch will be slow (~3s)
# while Metal kernels compile.  Model Fusion Proxy's config.yaml
# has a Stage-2 timeout of 2s for the classifier — this is OK
# because the classifier silently degrades to "general" on timeout
# and retry in the next request will hit warm kernels.
# ──────────────────────────────────────────────────────────────────

# ── Configuration ───────────────────────────────────────────────
MODEL_PATH="/Users/douyuan/Documents/Codex/2026-06-04/gemma-4-12b-it-abliterated-gguf/work/models/omlx/Qwen3.5-9B-MLX-4bit"
HOST="127.0.0.1"
PORT="8080"
LOG_DIR="$HOME/Library/Logs"

# You may need to change this to the Python that has mlx_lm installed
# Option A: use the project venv (if mlx_lm was pip-installed there)
PYTHON="$HOME/.gemini/antigravity-ide/scratch/model_fusion_proxy/.venv/bin/python"
# Option B: use system Python (uncomment below, comment above)
# PYTHON="/opt/homebrew/bin/python3"

cd "$HOME" || exit 1

# ── Pre-flight checks ──────────────────────────────────────────
if [ ! -d "$MODEL_PATH" ]; then
    echo "$(date): ERROR - MLX model not found at $MODEL_PATH" >> "$LOG_DIR/mlx-server-error.log"
    exit 1
fi

if [ ! -x "$PYTHON" ]; then
    echo "$(date): ERROR - Python not found at $PYTHON" >> "$LOG_DIR/mlx-server-error.log"
    exit 1
fi

# ── Ensure mlx_lm is installed ──────────────────────────────────
$PYTHON -c "import mlx_lm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "$(date): WARNING - mlx_lm not installed, attempting install..." >> "$LOG_DIR/mlx-server-error.log"
    $PYTHON -m pip install mlx mlx_lm --quiet
fi

# ── Warm-up: pre-compile Metal kernels (optional, uncomment to enable) ──
# The warm-up call forces Metal kernel compilation so the user's first
# real request doesn't time out.  If your model is large and you want
# zero-surprise latency, uncomment the next two lines.
#
# echo "$(date): Warming up Metal kernels..." >> "$LOG_DIR/mlx-server.log"
# $PYTHON -m mlx_lm.generate --model "$MODEL_PATH" --prompt "1+1=" --max-tokens 1 2>/dev/null
# echo "$(date): Warm-up complete." >> "$LOG_DIR/mlx-server.log"

# ── Start the MLX OpenAI-compatible server ──────────────────────
exec $PYTHON -m mlx_lm.server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT"
