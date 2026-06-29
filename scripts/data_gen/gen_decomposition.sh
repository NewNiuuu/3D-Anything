#!/bin/bash
# ==============================================================
# Decomposition 数据生成脚本
# ==============================================================
# 任务：模型读图，生成高层任务及其低层动作分解
# 输入：图片目录
# 输出：results/decomposition/decomposition_output.jsonl
#
# 使用方法: bash scripts/data_gen/gen_decomposition.sh
# 前提条件: 已启动 API 服务 (bash start_qwen35.sh)
# ==============================================================

# ======================== 配置区域 ============================

# ---------- 输入设置 ----------
IMAGE_DIR="data/DVG_sample"
MAX_IMAGES=0    # 0 = 处理全部

# ---------- 提示词设置 ----------

SYSTEM_PROMPT='You are an AI visual assistant specialized in task decomposition for robotic manipulation and navigation.

Given an image of a scene, your job is to:
1. Identify a meaningful high-level task that a robot could perform in this scene.
2. Decompose that high-level task into a sequence of low-level primitive actions.

Requirements for the high-level task:
- It should be physically achievable by a mobile manipulator robot.
- It should be contextually appropriate for the scene (e.g., "clean the table" for a messy table scene).
- Generate 2-3 different high-level tasks per scene for diversity.

Requirements for low-level actions:
- Each action should be a primitive operation: navigate, pick, place, push, pull, open, close, rotate, pour, etc.
- Actions should reference specific objects visible in the scene.
- Include spatial targets (e.g., "navigate to the table on the left").
- The sequence should be logically ordered and complete.

Output format (JSON):
{
  "tasks": [
    {
      "high_level_task": "Clear the dining table",
      "context": "The table has dirty plates and cups from a meal",
      "low_level_actions": [
        {"step": 1, "action": "navigate", "target": "dining table", "detail": "Move to the dining table in the center of the room"},
        {"step": 2, "action": "pick", "target": "plate", "detail": "Grasp the dirty plate on the left side of the table"},
        {"step": 3, "action": "navigate", "target": "kitchen sink", "detail": "Move to the kitchen sink area"},
        {"step": 4, "action": "place", "target": "sink", "detail": "Place the plate into the sink"},
        {"step": 5, "action": "navigate", "target": "dining table", "detail": "Return to the dining table"},
        {"step": 6, "action": "pick", "target": "cup", "detail": "Grasp the cup near the center of the table"},
        {"step": 7, "action": "navigate", "target": "kitchen sink", "detail": "Move to the kitchen sink"},
        {"step": 8, "action": "place", "target": "sink", "detail": "Place the cup into the sink"}
      ]
    }
  ]
}'

USER_PROMPT='Observe this scene carefully. Identify 2-3 meaningful high-level tasks that a robot could perform here, and decompose each into a sequence of low-level primitive actions. Consider the objects, their states, and spatial layout. Output as JSON.'

# ---------- 生成参数 ----------
TEMPERATURE=0.7
MAX_TOKENS=2048

# ---------- 输出设置 ----------
OUTPUT="results/decomposition/decomposition_output.jsonl"

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
echo "  Decomposition 数据生成 (High-level Task → Low-level Actions)"
echo "============================================================"
echo "  输入目录: $IMAGE_DIR"
echo "  输出文件: $OUTPUT"
echo "  温度:     $TEMPERATURE"
echo "  Max Token: $MAX_TOKENS"
echo "============================================================"

python scripts/infer_qwen35.py "${ARGS[@]}"

echo ""
echo "完成! 输出: $OUTPUT"
