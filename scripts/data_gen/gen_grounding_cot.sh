#!/bin/bash
# ==============================================================
# Grounding-CoT 数据生成脚本
# ==============================================================
# 任务：模型依据 Q-jsonl 中的问题，读图并生成包含思维链的定位回答
# 输入：图片目录 + 问题文件（Q-jsonl）
# 输出：results/grounding_cot/grounding_cot_output.jsonl
#
# 流程：
#   Step 1: gen_visual_grounding_qa.sh 生成问题 (Q-jsonl)
#   Step 2: 本脚本消费 Q-jsonl，生成 A-VG-CoT-jsonl
#
# 使用方法: bash scripts/data_gen/gen_grounding_cot.sh
# 前提条件: 已启动 API 服务 (bash start_qwen35.sh)
# ==============================================================

# ======================== 配置区域 ============================

# ---------- 输入设置 ----------
IMAGE_DIR="data/DVG_sample"
MAX_IMAGES=0    # 0 = 处理全部

# 使用第一步生成的 Q-jsonl 中的 question 作为提示词
USE_ANNOTATION=true
# 若 Q-jsonl 已合并到标注 JSON 中，设为 true
# 若使用独立的 Q-jsonl 文件，请指定路径：
Q_JSONL=""  # 留空则使用标注中的 question 字段

# ---------- 提示词设置 ----------

SYSTEM_PROMPT='You are a professional visual grounding expert. Your task is to precisely locate target objects in the image based on the user'"'"'s text description.

You must strictly follow this output format:

<think>
[Observation] Describe the overall scene: scene type, main objects, spatial layout.
[Analysis] Based on the user description, identify the visual features of the target (color, shape, size, category).
[Reasoning] Use spatial relationships (up/down/left/right, near/far, occlusion) to eliminate distractors and pinpoint the target.
[Localization] Estimate the bounding box pixel coordinates of the target.
</think>

<answer>
target_name: [x1, y1, x2, y2]
</answer>

Rules:
- The <think> section must contain a complete reasoning process with specific content for each step.
- Coordinates are pixel values in format [top-left-x, top-left-y, bottom-right-x, bottom-right-y].
- If multiple targets exist, give each one on a separate line.
- Always provide your best estimate even if uncertain, and explain the uncertainty in reasoning.'

USER_PROMPT='Please locate the target object described in the question and provide its bounding box coordinates.'

# ---------- 生成参数 ----------
TEMPERATURE=0.6
MAX_TOKENS=2048

# ---------- 输出设置 ----------
OUTPUT="results/grounding_cot/grounding_cot_output.jsonl"

# ---------- API 设置 ----------
API_BASE="http://localhost:8000/v1"

# ======================== 运行区域 ============================

cd "$(dirname "$0")/../.." || exit 1

ARGS=()

if [ -n "$Q_JSONL" ]; then
    # 如果有独立的 Q-jsonl，逐行读取并配对图片
    echo "使用独立 Q-jsonl 文件: $Q_JSONL"
    ARGS+=(--image-dir "$IMAGE_DIR")
    ARGS+=(--use-annotation)
else
    ARGS+=(--image-dir "$IMAGE_DIR")
    if [ "$USE_ANNOTATION" = true ]; then
        ARGS+=(--use-annotation)
    fi
fi

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
echo "  Grounding-CoT 数据生成"
echo "============================================================"
echo "  输入目录: $IMAGE_DIR"
echo "  输出文件: $OUTPUT"
echo "  温度:     $TEMPERATURE"
echo "  Max Token: $MAX_TOKENS"
echo "============================================================"

python scripts/infer_qwen35.py "${ARGS[@]}"

echo ""
echo "完成! 输出: $OUTPUT"
echo "下一步: 使用合成脚本将 Q-jsonl 和 A-CoT-jsonl 组合为训练数据集"
