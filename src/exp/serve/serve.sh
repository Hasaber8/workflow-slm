#!/bin/bash
set -e

# Serving script for Qwen3-4B with LoRA adapters
# Adapted for 12 GB VRAM (RTX 3060)

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.80}"

ADAPTER_ARGS=""
if [ -d "artifacts/adapters/triage_v1/epoch1" ]; then
  ADAPTER_ARGS="--enable-lora --max-lora-rank 16 --lora-modules ep1=artifacts/adapters/triage_v1/epoch1 ep2=artifacts/adapters/triage_v1/epoch2 ep3=artifacts/adapters/triage_v1/epoch3"
fi

echo "Starting vLLM serving for ${BASE_MODEL} on port ${PORT}..."
exec vllm serve "${BASE_MODEL}" \
  --served-model-name base \
  ${ADAPTER_ARGS} \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --port "${PORT}"
