#!/bin/bash
# ==============================================================
# Visual Grounding 问题生成脚本
# ==============================================================
# 任务：模型读图，自动生成 visual grounding 类型的问题
# 输入：图片目录
# 输出：results/grounding_cot/questions_output.jsonl (Q-jsonl)
#
# 这是 Grounding-CoT 流程的第一步：
#   Step 1: 本脚本 → 生成 Q-jsonl
#   Step 2: gen_grounding_cot.sh → 消费 Q-jsonl，生成 A-VG-CoT-jsonl
#
# 使用方法: bash scripts/data_gen/gen_visual_grounding_qa.sh
# 前提条件: 已启动 API 服务 (bash start_qwen35.sh)
# ==============================================================

# ======================== 配置区域 ============================

# ---------- 输入设置 ----------
IMAGE_DIR="data/DVG_sample"
MAX_IMAGES=0    # 0 = 处理全部

# ---------- 提示词设置 ----------

SYSTEM_PROMPT='You are a question generator for visual grounding tasks.

Given an image of a scene, your job is to generate diverse and specific questions that ask about the spatial location of objects in the scene. Each question should be answerable by providing a bounding box coordinate.

Requirements:
- Generate 3-5 questions per image.
- Questions should target different objects or spatial relationships in the scene.
- Questions should be specific enough to have a unique answer (avoid ambiguous references).
- Cover different question types: object identification, spatial relationship, attribute-based, counting+locating.
- Use natural language that a human might ask.

Output format (one JSON array):
[
  {"question": "Where is the red car parked?", "target": "red car", "type": "object"},
  {"question": "Which building is tallest in the background?", "target": "tallest building", "type": "attribute"},
  {"question": "What is to the left of the traffic light?", "target": "object left of traffic light", "type": "spatial"}
]

Question types:
- "object": directly asking to locate a specific object
- "attribute": locating an object by its distinguishing attribute
- "spatial": locating based on spatial relationships
- "counting": locating multiple instances of the same category'

USER_PROMPT='Please observe this image carefully and generate 3-5 visual grounding questions. Each question should ask about the location of a specific object or region in the scene. Output as a JSON array.'

# ---------- 生成参数 ----------
TEMPERATURE=0.8
MAX_TOKENS=1024

# ---------- 输出设置 ----------
OUTPUT="results/grounding_cot/questions_output.jsonl"

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
echo "  Visual Grounding 问题生成 (Q-jsonl)"
echo "============================================================"
echo "  输入目录: $IMAGE_DIR"
echo "  输出文件: $OUTPUT"
echo "  温度:     $TEMPERATURE"
echo "  Max Token: $MAX_TOKENS"
echo "============================================================"

python scripts/infer_qwen35.py "${ARGS[@]}"

echo ""
echo "完成! 问题文件: $OUTPUT"
echo "下一步: 运行 bash scripts/data_gen/gen_grounding_cot.sh 生成思维链回答"
