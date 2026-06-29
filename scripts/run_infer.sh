#!/bin/bash
# ==============================================================
# Qwen3.5-35B-A3B 推理启动脚本
# ==============================================================
# 使用方法: bash scripts/run_infer.sh
# 只需修改下方配置区域，无需改动 Python 代码
# ==============================================================

# ======================== 配置区域 ============================

# ---------- 输入设置 (二选一，注释掉不用的那个) ----------

# 方式A: 单张图片
IMAGE="data/DVG_sample/0000000_00098_d_0000001.jpg"

# 方式B: 批量目录 (取消注释下面两行，并注释掉上面的 IMAGE)
# IMAGE_DIR="data/DVG_sample"
# MAX_IMAGES=10

# ---------- 标注设置 ----------
# 设为 true 时，使用同名 JSON 标注中的 question 作为用户提示词
# 此时下方的 USER_PROMPT 会被忽略
USE_ANNOTATION=false

# ---------- 提示词设置 ----------

SYSTEM_PROMPT='你是一个专业的视觉定位(Visual Grounding)专家。你的任务是根据用户的文字描述，在图片中精确定位目标物体。

你必须严格按照以下格式输出，不得省略任何部分：

<think>
[观察] 对图片整体内容进行描述，包括场景类型、主要物体、空间布局等。
[分析] 根据用户的文字描述，分析目标物体应具备的视觉特征（颜色、形状、大小、类别等）。
[推理] 结合空间关系（上下左右、远近、遮挡等），逐步排除干扰物体，锁定目标位置。
[定位] 估算目标物体的边界框像素坐标。
</think>

<answer>
[x1, y1, x2, y2]
</answer>

重要规则：
- <think>部分必须包含完整的推理过程，每个步骤都要有具体内容
- 坐标为像素值，格式为 [左上角x, 左上角y, 右下角x, 右下角y]
- 如果有多个目标，每个目标单独一行给出坐标
- 不确定时也要给出最佳估计，并在推理中说明不确定性'

USER_PROMPT='请定位图中的主要目标物体，给出每个物体的类别和边界框坐标。'

# ---------- 生成参数 ----------
TEMPERATURE=0.7
MAX_TOKENS=2048

# ---------- 输出设置 ----------
OUTPUT="results/infer_output.jsonl"

# ---------- API 设置 ----------
API_BASE="http://localhost:8000/v1"

# ======================== 运行区域 ============================
# 以下内容无需修改

cd "$(dirname "$0")/.." || exit 1

# 构建命令参数
ARGS=()

if [ -n "$IMAGE" ] && [ -z "$IMAGE_DIR" ]; then
    ARGS+=(--image "$IMAGE")
elif [ -n "$IMAGE_DIR" ]; then
    ARGS+=(--image-dir "$IMAGE_DIR")
    if [ -n "$MAX_IMAGES" ] && [ "$MAX_IMAGES" -gt 0 ]; then
        ARGS+=(--max-images "$MAX_IMAGES")
    fi
fi

if [ "$USE_ANNOTATION" = true ]; then
    ARGS+=(--use-annotation)
fi

ARGS+=(--system-prompt "$SYSTEM_PROMPT")
ARGS+=(--prompt "$USER_PROMPT")
ARGS+=(--temperature "$TEMPERATURE")
ARGS+=(--max-tokens "$MAX_TOKENS")
ARGS+=(--output "$OUTPUT")
ARGS+=(--api-base "$API_BASE")

echo "============================================================"
echo "Qwen3.5 推理启动"
echo "============================================================"
echo "  输入:   ${IMAGE:-$IMAGE_DIR (max: ${MAX_IMAGES:-all})}"
echo "  输出:   $OUTPUT"
echo "  温度:   $TEMPERATURE"
echo "  Token:  $MAX_TOKENS"
echo "============================================================"

python scripts/infer_qwen35.py "${ARGS[@]}"
