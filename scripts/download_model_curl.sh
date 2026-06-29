#!/bin/bash
# ==============================================================
# Download Qwen3.5-35B-A3B model using curl
# ==============================================================
# This script downloads the model files from HuggingFace using curl
# with resume support. Run if huggingface-cli is not available.
#
# Usage: bash scripts/download_model_curl.sh
# ==============================================================

set -e

REPO="Qwen/Qwen3.5-35B-A3B"
REVISION="59d61f3ce65a6d9863b86d2e96597125219dc754"
CACHE_DIR="/root/nyp/hf_cache"
MODEL_DIR="$CACHE_DIR/hub/models--Qwen--Qwen3.5-35B-A3B"
SNAPSHOT_DIR="$MODEL_DIR/snapshots/$REVISION"
BLOBS_DIR="$MODEL_DIR/blobs"
REFS_DIR="$MODEL_DIR/refs"

# Create directory structure matching HF cache format
mkdir -p "$SNAPSHOT_DIR"
mkdir -p "$BLOBS_DIR"
mkdir -p "$REFS_DIR"

# Write refs/main
echo "$REVISION" > "$REFS_DIR/main"

BASE_URL="https://huggingface.co/$REPO/resolve/main"

# Files to download
FILES=(
    "config.json"
    "generation_config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "preprocessor_config.json"
    "video_preprocessor_config.json"
    "chat_template.jinja"
    "merges.txt"
    "vocab.json"
    "model.safetensors.index.json"
    "model.safetensors-00001-of-00014.safetensors"
    "model.safetensors-00002-of-00014.safetensors"
    "model.safetensors-00003-of-00014.safetensors"
    "model.safetensors-00004-of-00014.safetensors"
    "model.safetensors-00005-of-00014.safetensors"
    "model.safetensors-00006-of-00014.safetensors"
    "model.safetensors-00007-of-00014.safetensors"
    "model.safetensors-00008-of-00014.safetensors"
    "model.safetensors-00009-of-00014.safetensors"
    "model.safetensors-00010-of-00014.safetensors"
    "model.safetensors-00011-of-00014.safetensors"
    "model.safetensors-00012-of-00014.safetensors"
    "model.safetensors-00013-of-00014.safetensors"
    "model.safetensors-00014-of-00014.safetensors"
)

TOTAL=${#FILES[@]}
COUNT=0

echo "============================================================"
echo "  Downloading Qwen/Qwen3.5-35B-A3B"
echo "  Target: $SNAPSHOT_DIR"
echo "  Files: $TOTAL"
echo "============================================================"
echo ""

for FILE in "${FILES[@]}"; do
    COUNT=$((COUNT + 1))
    TARGET="$SNAPSHOT_DIR/$FILE"

    if [ -f "$TARGET" ]; then
        echo "[$COUNT/$TOTAL] SKIP (exists): $FILE"
        continue
    fi

    echo "[$COUNT/$TOTAL] Downloading: $FILE"
    curl -L -C - --progress-bar \
        "$BASE_URL/$FILE" \
        -o "$TARGET"

    if [ $? -ne 0 ]; then
        echo "  ✗ FAILED: $FILE"
        rm -f "$TARGET"
        exit 1
    fi
done

echo ""
echo "============================================================"
echo "  Download Complete!"
echo "  Model path: $SNAPSHOT_DIR"
echo "============================================================"

# Verify
SHARD_COUNT=$(find "$SNAPSHOT_DIR" -name "*.safetensors" | wc -l)
echo "  Safetensors files: $SHARD_COUNT/14"
echo "  Total size: $(du -sh "$MODEL_DIR" | cut -f1)"
