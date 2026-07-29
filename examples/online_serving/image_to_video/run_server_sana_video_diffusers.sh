#!/bin/bash
# SANA-Video-2B image-to-video serving through the Diffusers adapter.
# The 480p checkpoint is validated; 720p adapter I2V is not yet claimed.

set -euo pipefail

MODEL="${MODEL:-Efficient-Large-Model/SANA-Video_2B_480p_diffusers}"
PORT="${PORT:-8099}"

vllm serve "$MODEL" \
    --omni \
    --model-class-name SanaImageToVideoPipeline \
    --diffusion-load-format diffusers \
    --diffusion-attention-backend TORCH_SDPA \
    --dtype bfloat16 \
    --port "$PORT"
