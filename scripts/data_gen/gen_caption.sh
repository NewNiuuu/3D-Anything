#!/bin/bash
# ==============================================================
# Caption 数据生成脚本
# ==============================================================
# 任务：模型读图，生成场景描述（caption）
# 输入：图片目录
# 输出：results/caption/caption_output.jsonl
#
# 使用方法: bash scripts/data_gen/gen_caption.sh
# 前提条件: 已启动 API 服务 (bash start_qwen35.sh)
# ==============================================================

# ======================== 配置区域 ============================

# ---------- 输入设置 ----------
IMAGE_DIR="data/DVG_sample"
MAX_IMAGES=0    # 0 = 处理全部

# ---------- 提示词设置 ----------

SYSTEM_PROMPT='You are an AI visual assistant that can analyze images in detail.

Your task is to generate a comprehensive and accurate description of the scene shown in the image.

Requirements:
- Describe the overall scene type and setting (indoor/outdoor, urban/rural, time of day, weather).
- Identify and describe all prominent objects, their attributes (color, size, material), and their spatial positions.
- Describe spatial relationships between objects (relative positions, distances, arrangements).
- Note any actions, activities, or dynamic elements in the scene.
- Mention background elements and environmental context.
- Be specific and factual - only describe what you can actually observe.
- Use natural, flowing language organized from general to specific.

Output format:
Provide a single detailed paragraph (150-300 words) describing the scene comprehensively. Do not use bullet points or numbered lists.'

USER_PROMPT='Please provide a detailed description of this scene. Include the setting, all visible objects with their attributes and positions, spatial relationships between objects, and any notable activities or environmental details.'

# ---------- 生成参数 ----------
TEMPERATURE=0.7
MAX_TOKENS=1024

# ---------- 输出设置 ----------
OUTPUT="results/caption/caption_output.jsonl"

# ---------- API 设置 ----------
API_BASE="http://localhost:8000/v1"

# ======================== 运行区域 ============================

cd "$(dirname "$0")/../.." || exit 1

ARGS=()
ARGS+=(--image-dir "$IMAGE_DIR")

if [ "$MAX_IMAGES" -gt 0 ]; then
    ARGS+=(--max-images "$MAX_IMAGES")
fi

ARGS+=(--system-prompt "$SYSTEM_PROMPT")
ARGS+=(--prompt "$USER_PROMPT")
ARGS+=(--temperature "$TEMPERATURE")
ARGS+=(--max-tokens "$MAX_TOKENS")
ARGS+=(--output "$OUTPUT")
ARGS+=(--api-base "$API_BASE")

echo "============================================================"
echo "  Caption 数据生成"
echo "============================================================"
echo "  输入目录: $IMAGE_DIR"
echo "  输出文件: $OUTPUT"
echo "  温度:     $TEMPERATURE"
echo "  Max Token: $MAX_TOKENS"
echo "============================================================"

python scripts/infer_qwen35.py "${ARGS[@]}"

echo ""
echo "完成! 输出: $OUTPUT"
