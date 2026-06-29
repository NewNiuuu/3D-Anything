"""
Qwen2-VL-72B-Instruct 多模态推理脚本
=====================================
用途：通过 vLLM OpenAI 兼容 API，批量对图片进行多模态推理。
      适用于 visual grounding CoT 数据生成等任务。

使用前提：
    先启动 vLLM 服务：bash start_qwen2vl.sh
    确认服务就绪：curl http://localhost:8000/v1/models

用法：
    # 单张图片测试
    python scripts/infer_qwen2vl.py --image data/DVG_sample/0000000_00098_d_0000001.jpg

    # 指定提示词
    python scripts/infer_qwen2vl.py --image data/DVG_sample/0000000_00098_d_0000001.jpg \
        --prompt "请描述图中所有车辆的位置。"

    # 批量处理整个目录
    python scripts/infer_qwen2vl.py --image-dir data/DVG_sample --max-images 10

    # 结合 JSON 标注信息（将标注中的 question 作为提示词）
    python scripts/infer_qwen2vl.py --image-dir data/DVG_sample --use-annotation --max-images 5

    # 自定义输出路径
    python scripts/infer_qwen2vl.py --image-dir data/DVG_sample --output results/output.jsonl
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# ============================================================
# 配置区域（可根据任务需要修改）
# ============================================================

API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "dummy"  # vLLM 不校验 key，任意值即可
MODEL_NAME = "Qwen/Qwen2-VL-72B-Instruct"

# 默认系统提示词 —— 针对 visual grounding CoT 任务设计，可按需替换
DEFAULT_SYSTEM_PROMPT = """你是一个专业的视觉定位(Visual Grounding)专家。你的任务是根据用户的文字描述，在图片中精确定位目标物体。

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
- 不确定时也要给出最佳估计，并在推理中说明不确定性"""

# 默认用户提示词 —— 当未通过 --prompt 或 --use-annotation 指定时使用
DEFAULT_USER_PROMPT = (
    "请定位图中的主要目标物体，给出每个物体的类别和边界框坐标。"
)

# 生成参数
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048


# ============================================================
# 核心函数
# ============================================================


def encode_image_to_base64(image_path: str) -> str:
    """将本地图片编码为 base64 字符串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """根据文件扩展名返回 MIME 类型。"""
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }
    return media_types.get(ext, "image/jpeg")


def load_annotation(image_path: str) -> dict | None:
    """加载与图片同名的 JSON 标注文件（如果存在）。"""
    json_path = Path(image_path).with_suffix(".json")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def infer_single_image(
    client: OpenAI,
    image_path: str,
    user_prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """
    对单张图片进行推理。

    Returns:
        dict: 包含 image_path, prompt, response, usage, elapsed_s 等字段
    """
    # 编码图片
    image_base64 = encode_image_to_base64(image_path)
    media_type = get_image_media_type(image_path)
    image_url = f"data:{media_type};base64,{image_base64}"

    # 构造消息
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    # 调用 API
    start_time = time.time()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - start_time

    # 提取结果
    choice = response.choices[0]
    usage = response.usage

    result = {
        "image_path": image_path,
        "prompt": user_prompt,
        "system_prompt": system_prompt,
        "response": choice.message.content,
        "finish_reason": choice.finish_reason,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
        "elapsed_s": round(elapsed, 2),
    }
    return result


def collect_image_tasks(
    image_path: str | None,
    image_dir: str | None,
    user_prompt: str,
    use_annotation: bool,
    max_images: int,
) -> list[dict]:
    """
    收集所有待推理的 (image_path, prompt) 任务列表。

    Returns:
        list[dict]: 每个元素包含 image_path 和 prompt
    """
    tasks = []

    if image_path:
        # 单张图片模式
        prompt = user_prompt
        if use_annotation:
            ann = load_annotation(image_path)
            if ann and "question" in ann:
                prompt = ann["question"]
        tasks.append({"image_path": image_path, "prompt": prompt})

    elif image_dir:
        # 批量目录模式
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_files = sorted(
            p
            for p in Path(image_dir).iterdir()
            if p.suffix.lower() in image_extensions
        )

        if max_images > 0:
            image_files = image_files[:max_images]

        for img_file in image_files:
            prompt = user_prompt
            if use_annotation:
                ann = load_annotation(str(img_file))
                if ann and "question" in ann:
                    prompt = ann["question"]
            tasks.append({"image_path": str(img_file), "prompt": prompt})

    return tasks


def save_result(result: dict, output_path: str):
    """将单条结果追加写入 JSONL 文件。"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def print_result_summary(result: dict, index: int, total: int):
    """打印单条推理结果的摘要。"""
    print(f"\n{'='*60}")
    print(f"[{index}/{total}] {Path(result['image_path']).name}")
    print(f"{'='*60}")
    print(f"  Prompt:   {result['prompt'][:80]}{'...' if len(result['prompt']) > 80 else ''}")
    print(f"  Time:     {result['elapsed_s']}s")
    print(f"  Tokens:   {result['usage']['prompt_tokens']} prompt + "
          f"{result['usage']['completion_tokens']} completion = "
          f"{result['usage']['total_tokens']} total")
    print(f"  Response: {result['response'][:200]}{'...' if len(result['response']) > 200 else ''}")


# ============================================================
# 主入口
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2-VL 多模态推理脚本 — 支持单图/批量模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单张图片快速测试
  python scripts/infer_qwen2vl.py \\
      --image data/DVG_sample/0000000_00098_d_0000001.jpg

  # 批量处理 + 使用标注中的 question 作为提示词
  python scripts/infer_qwen2vl.py \\
      --image-dir data/DVG_sample \\
      --use-annotation \\
      --max-images 10 \\
      --output results/vg_cot_output.jsonl
        """,
    )

    # 输入源（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", type=str, help="单张图片路径"
    )
    input_group.add_argument(
        "--image-dir", type=str, help="图片目录路径（批量模式）"
    )

    # 提示词
    parser.add_argument(
        "--prompt", type=str, default=DEFAULT_USER_PROMPT,
        help="用户提示词（默认使用 visual grounding 提示）"
    )
    parser.add_argument(
        "--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
        help="系统提示词"
    )
    parser.add_argument(
        "--use-annotation", action="store_true",
        help="使用同名 JSON 标注文件中的 question 字段作为提示词"
    )

    # 生成参数
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"生成温度 (默认: {DEFAULT_TEMPERATURE})"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"最大生成 token 数 (默认: {DEFAULT_MAX_TOKENS})"
    )

    # 批量控制
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="最多处理的图片数量，0 表示全部 (默认: 0)"
    )

    # 输出
    parser.add_argument(
        "--output", type=str, default="results/infer_output.jsonl",
        help="输出 JSONL 文件路径 (默认: results/infer_output.jsonl)"
    )

    # API 配置
    parser.add_argument(
        "--api-base", type=str, default=API_BASE_URL,
        help=f"API 地址 (默认: {API_BASE_URL})"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 初始化客户端
    client = OpenAI(base_url=args.api_base, api_key=API_KEY)

    # 验证服务连接
    print("正在连接 vLLM 服务...")
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"  服务正常，可用模型: {available}")
        if MODEL_NAME not in available:
            print(f"  ⚠ 警告: 目标模型 {MODEL_NAME} 不在列表中，可能无法推理")
    except Exception as e:
        print(f"  ✗ 无法连接到 vLLM 服务 ({args.api_base}): {e}")
        print("  请确认已运行: bash start_qwen2vl.sh")
        sys.exit(1)

    # 收集任务
    tasks = collect_image_tasks(
        image_path=args.image,
        image_dir=args.image_dir,
        user_prompt=args.prompt,
        use_annotation=args.use_annotation,
        max_images=args.max_images,
    )

    if not tasks:
        print("未找到任何图片，请检查路径。")
        sys.exit(1)

    print(f"\n共 {len(tasks)} 张图片待推理")
    print(f"输出文件: {args.output}")
    print(f"生成参数: temperature={args.temperature}, max_tokens={args.max_tokens}")

    # 清空输出文件（如已存在）
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    # 逐张推理
    total = len(tasks)
    success_count = 0
    fail_count = 0
    total_start = time.time()

    for i, task in enumerate(tasks, 1):
        try:
            result = infer_single_image(
                client=client,
                image_path=task["image_path"],
                user_prompt=task["prompt"],
                system_prompt=args.system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            # 如果使用了标注，把标注信息也保存进去
            if args.use_annotation:
                ann = load_annotation(task["image_path"])
                if ann:
                    result["annotation"] = ann

            save_result(result, output_path)
            print_result_summary(result, i, total)
            success_count += 1

        except Exception as e:
            print(f"\n[{i}/{total}] ✗ {Path(task['image_path']).name} 失败: {e}")
            fail_count += 1
            # 记录失败信息
            error_result = {
                "image_path": task["image_path"],
                "prompt": task["prompt"],
                "error": str(e),
            }
            save_result(error_result, output_path)

    # 汇总
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print("推理完成!")
    print(f"  成功: {success_count}/{total}")
    print(f"  失败: {fail_count}/{total}")
    print(f"  总耗时: {total_elapsed:.1f}s")
    print(f"  平均每张: {total_elapsed/total:.1f}s")
    print(f"  结果保存至: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
