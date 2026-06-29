#!/bin/bash
# ==============================================================
# Qwen2-VL → Qwen3.5 Migration Execution Script
# ==============================================================
# This script completes the migration steps that require manual execution:
# 1. Downloads the new model
# 2. Makes scripts executable
# 3. Verifies the pipeline
# 4. Commits changes
# 5. Deletes old model weights
#
# Usage: bash scripts/migrate_to_qwen35.sh
# ==============================================================

set -e
cd "$(dirname "$0")/.." || exit 1

echo "============================================================"
echo "  Qwen2-VL-72B → Qwen3.5-35B-A3B Migration"
echo "============================================================"
echo ""

# ======================== Step 1: Download Model ========================
echo "[1/5] Downloading Qwen3.5-35B-A3B model..."
echo "      This may take 15-60 minutes depending on network speed."
echo ""

export HF_HOME=/root/nyp/hf_cache

# Use huggingface-cli if available, otherwise use Python
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download Qwen/Qwen3.5-35B-A3B --cache-dir /root/nyp/hf_cache
else
    python -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='Qwen/Qwen3.5-35B-A3B',
    cache_dir='/root/nyp/hf_cache',
    resume_download=True,
)
print(f'Model downloaded to: {path}')
"
fi

# Verify download
echo ""
echo "  Verifying download..."
SHARD_COUNT=$(find /root/nyp/hf_cache -path "*Qwen3.5-35B-A3B*" -name "*.safetensors" | wc -l)
INCOMPLETE=$(find /root/nyp/hf_cache -path "*Qwen3.5-35B-A3B*" -name "*.incomplete" | wc -l)

if [ "$SHARD_COUNT" -lt 14 ]; then
    echo "  ✗ ERROR: Only found $SHARD_COUNT safetensors files (expected 14)"
    echo "  Please re-run the download command."
    exit 1
fi

if [ "$INCOMPLETE" -gt 0 ]; then
    echo "  ✗ ERROR: Found $INCOMPLETE incomplete files"
    echo "  Please re-run the download command."
    exit 1
fi

echo "  ✓ Download verified: $SHARD_COUNT safetensors files, no incomplete files"
echo ""

# ======================== Step 2: Fix Permissions ========================
echo "[2/5] Setting file permissions..."
chmod +x start_qwen35.sh
chmod +x scripts/download_model.py
echo "  ✓ Scripts are executable"
echo ""

# ======================== Step 3: Verify Pipeline ========================
echo "[3/5] Verifying inference pipeline..."
echo ""

export CUDA_HOME=/root/miniconda3/envs/vllm_env/lib/python3.10/site-packages/nvidia/cu13
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_HAS_FLASHINFER_CUBIN=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# Activate conda if not already
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate vllm_env 2>/dev/null || true

echo "  Running offline inference test..."
python inference_qwen35.py

if [ $? -eq 0 ]; then
    echo ""
    echo "  ✓ Offline inference PASSED"
else
    echo ""
    echo "  ✗ Offline inference FAILED"
    echo "  Please check the error above and fix before continuing."
    exit 1
fi
echo ""

# ======================== Step 4: Git Commit ========================
echo "[4/5] Committing changes..."
echo ""

# Remove old deprecated files from git tracking
git rm --cached start_qwen2vl.sh 2>/dev/null || true
git rm --cached inference_qwen2vl.py 2>/dev/null || true
git rm --cached scripts/infer_qwen2vl.py 2>/dev/null || true
git rm --cached doc/qwen2vl_setup_log.md 2>/dev/null || true

# Add all new/modified files
git add start_qwen35.sh
git add inference_qwen35.py
git add scripts/infer_qwen35.py
git add scripts/run_infer.sh
git add scripts/download_model.py
git add README.md
git add requirements.txt
git add doc/qwen35_setup_log.md

# Commit
git commit -m "refactor: migrate from Qwen2-VL-72B to Qwen3.5-35B-A3B

- Replace all model references from Qwen/Qwen2-VL-72B-Instruct to Qwen/Qwen3.5-35B-A3B
- Rename scripts: start_qwen2vl.sh -> start_qwen35.sh, etc.
- Update max-model-len from 8192 to 32768 (Qwen3.5 supports 256K)
- Update README.md with new hardware requirements (reduced from 4x80GB to 1x80GB min)
- Update doc/qwen35_setup_log.md with migration details
- Add scripts/download_model.py utility
- Qwen3.5-35B-A3B: MoE (256 experts, 8 active), multimodal, ~70GB vs 137GB

Co-Authored-By: Claude <noreply@anthropic.com>"

echo "  ✓ Changes committed"
echo ""

# ======================== Step 5: Delete Old Model ========================
echo "[5/5] Deleting old Qwen2-VL-72B model weights..."
echo "      This will free ~137GB of disk space."
echo ""

OLD_MODEL_DIR="/root/nyp/hf_cache/hub/models--Qwen--Qwen2-VL-72B-Instruct"
if [ -d "$OLD_MODEL_DIR" ]; then
    OLD_SIZE=$(du -sh "$OLD_MODEL_DIR" | cut -f1)
    rm -rf "$OLD_MODEL_DIR"
    echo "  ✓ Deleted $OLD_MODEL_DIR ($OLD_SIZE freed)"
else
    echo "  ⓘ Old model directory not found (already deleted?)"
fi

# Also remove deprecated script files from disk
rm -f start_qwen2vl.sh inference_qwen2vl.py scripts/infer_qwen2vl.py
echo "  ✓ Removed deprecated script files"
echo ""

# ======================== Summary ========================
echo "============================================================"
echo "  Migration Complete!"
echo "============================================================"
echo ""
echo "  Model:    Qwen/Qwen3.5-35B-A3B (MoE, 256 experts)"
echo "  Size:     ~70GB (was 137GB)"
echo "  Context:  256K tokens (serving: 32K)"
echo "  Vision:   Supported (built-in encoder)"
echo ""
echo "  New scripts:"
echo "    - start_qwen35.sh         (API server)"
echo "    - inference_qwen35.py     (offline inference)"
echo "    - scripts/infer_qwen35.py (batch multimodal)"
echo "    - scripts/run_infer.sh    (one-click launcher)"
echo ""
echo "  Quick start:"
echo "    bash start_qwen35.sh      # Start API server"
echo "    bash scripts/run_infer.sh # Run batch inference"
echo ""
echo "============================================================"
