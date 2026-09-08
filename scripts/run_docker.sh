#!/bin/bash

# ================== Qwen/Qwen3-4B ==================
docker run --gpus '"device=1"' \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      --env "HF_TOKEN=$HF_TOKEN" \
      -p 8002:8000 \
      --ipc=host \
      vllm/vllm-openai:v0.27.1-cu129 \
      --model Qwen/Qwen3-4B
