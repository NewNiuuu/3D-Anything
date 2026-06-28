#!/usr/bin/env python3
"""
Qwen2-VL-72B-Instruct Inference Example Script

Usage (offline/batch):
    CUDA_HOME=/root/miniconda3/envs/vllm_env/lib/python3.10/site-packages/nvidia/cu13 \
    HF_HOME=/root/nyp/hf_cache \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    VLLM_HAS_FLASHINFER_CUBIN=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    python /root/nyp/inference_qwen2vl.py

Usage (via API, after starting start_qwen2vl.sh):
    See the API examples at the bottom of this file.
"""

from vllm import LLM, SamplingParams
import time


def main():
    print("=" * 60)
    print("Loading Qwen2-VL-72B-Instruct with vLLM")
    print("=" * 60)

    t0 = time.time()
    llm = LLM(
        model="Qwen/Qwen2-VL-72B-Instruct",
        tensor_parallel_size=4,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )
    print(f"✓ Model loaded in {time.time() - t0:.1f}s\n")

    # ========== Text Generation Examples ==========
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        stop=["<|im_end|>"],
    )

    # Using chat template format
    prompts = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n请解释一下什么是Transformer架构？<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWrite a Python function to merge two sorted lists.<|im_end|>\n<|im_start|>assistant\n",
    ]

    print("Generating responses...")
    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t1

    for i, output in enumerate(outputs):
        generated = output.outputs[0].text
        token_count = len(output.outputs[0].token_ids)
        print(f"\n--- Response {i+1} ({token_count} tokens) ---")
        print(generated)

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"\n{'=' * 60}")
    print(f"Total: {total_tokens} tokens in {gen_time:.2f}s ({total_tokens/gen_time:.1f} tok/s)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()


"""
========== API Usage Examples (after starting start_qwen2vl.sh) ==========

# Text completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-72B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "什么是深度学习？"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'

# Vision (image understanding) - base64 encoded image
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-72B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
          {"type": "text", "text": "Describe this image in detail."}
        ]
      }
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'

# Python SDK usage
# pip install openai
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-72B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)
"""
