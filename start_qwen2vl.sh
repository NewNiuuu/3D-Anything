#!/bin/bash
# ==============================================================
# Qwen2-VL-72B-Instruct vLLM Serving Script
# ==============================================================
# Usage: bash /root/nyp/start_qwen2vl.sh
# API will be available at http://localhost:8000
# ==============================================================

# Required environment variables to fix CUDA compilation issues
export CUDA_HOME=/root/miniconda3/envs/vllm_env/lib/python3.10/site-packages/nvidia/cu13
export HF_HOME=/root/nyp/hf_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_HAS_FLASHINFER_CUBIN=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vllm_env

# Start vLLM OpenAI-compatible API server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-72B-Instruct \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 8000
