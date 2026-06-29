#!/usr/bin/env python3
"""Download Qwen3.5-35B-A3B model to HF cache."""
import os
os.environ["HF_HOME"] = "/root/nyp/hf_cache"

from huggingface_hub import snapshot_download

print("=" * 60)
print("Downloading Qwen/Qwen3.5-35B-A3B...")
print("Cache dir: /root/nyp/hf_cache")
print("=" * 60)

path = snapshot_download(
    repo_id="Qwen/Qwen3.5-35B-A3B",
    cache_dir="/root/nyp/hf_cache",
    resume_download=True,
)

print(f"\n✓ Download complete!")
print(f"  Model path: {path}")
