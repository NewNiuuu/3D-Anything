#!/bin/bash
# ==============================================================
# Qwen3.5-35B-A3B vLLM Serving Script
# ==============================================================
# Usage: bash /root/nyp/3D-anything/start_qwen35.sh
# API will be available at http://localhost:8000
# ==============================================================

# Required environment variables to fix CUDA compilation issues
export CUDA_HOME=/root/miniconda3/envs/vllm_env/lib/python3.10/site-packages/nvidia/cu13
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export HF_HOME=/root/nyp/hf_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_HAS_FLASHINFER_CUBIN=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vllm_env

# Start vLLM OpenAI-compatible API server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --host 0.0.0.0 \
    --port 8000
