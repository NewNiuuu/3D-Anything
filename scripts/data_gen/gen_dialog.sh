#!/bin/bash
# ==============================================================
# Dialog 数据生成脚本
# ==============================================================
# 任务：模型读图，生成人类与机器人助手之间的多轮对话
# 输入：图片目录
# 输出：results/dialog/dialog_output.jsonl
#
# 使用方法: bash scripts/data_gen/gen_dialog.sh
# 前提条件: 已启动 API 服务 (bash start_qwen35.sh)
# ==============================================================

# ======================== 配置区域 ============================

# ---------- 输入设置 ----------
IMAGE_DIR="data/DVGBench/images/visdrone"
MAX_IMAGES=0    # 0 = 处理全部

# ---------- 提示词设置 ----------

SYSTEM_PROMPT='You are a conversation generator for visual scene understanding.

Given an image of a scene, generate a natural multi-round conversation (4-10 rounds) between a human and a robot assistant. The conversation should be about the scene shown in the image.

Requirements:
- The human asks questions about the scene, and the robot provides informative, helpful answers.
- Questions should progressively build on previous answers (not just independent questions).
- Cover diverse aspects: object identification, spatial relationships, attributes, functions, suggestions, planning.
- The robot assistant should be knowledgeable, helpful, and occasionally proactive (offering additional useful information).
- Maintain natural conversation flow - not just Q&A but genuine dialogue with follow-ups.
- The robot can reference spatial locations, object properties, and scene context in its answers.
- Include at least one question about spatial relationships and one about potential actions/tasks.

Output format (JSON array of conversation turns):
[
  {"role": "human", "content": "What do you see on the table?"},
  {"role": "assistant", "content": "I can see a laptop, a coffee mug, and several papers scattered on the wooden desk. The laptop is open and positioned in the center, with the mug to its right side."},
  {"role": "human", "content": "Is there anything near the laptop that might spill on it?"},
  {"role": "assistant", "content": "Yes, the coffee mug is quite close to the laptop, approximately 15cm to its right. It appears to be a full mug without a lid. I would suggest moving it further away or using a coaster with a lid to prevent any accidental spills on the laptop."},
  ...
]

Guidelines for natural dialogue:
- Human can ask follow-up questions based on robot'"'"'s answers
- Robot can volunteer relevant information beyond what was asked
- Include some turns about practical concerns (safety, suggestions, planning)
- Vary question complexity from simple identification to complex reasoning'

USER_PROMPT='Based on this image, generate a natural multi-round conversation (4-10 rounds) between a human and a robot assistant discussing the scene. The conversation should cover different aspects of the scene including objects, spatial relationships, and potential tasks. Output as a JSON array.'

# ---------- 生成参数 ----------
TEMPERATURE=0.8
MAX_TOKENS=2048

# ---------- 输出设置 ----------
OUTPUT="results/dialog/dialog_output.jsonl"

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
echo "  Dialog 数据生成 (Multi-round Conversation)"
echo "============================================================"
echo "  输入目录: $IMAGE_DIR"
echo "  输出文件: $OUTPUT"
echo "  温度:     $TEMPERATURE"
echo "  Max Token: $MAX_TOKENS"
echo "============================================================"

python scripts/infer_qwen35.py "${ARGS[@]}"

echo ""
echo "完成! 输出: $OUTPUT"
