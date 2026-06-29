# Qwen2-VL-72B-Instruct 部署与推理

基于 vLLM 框架部署 [Qwen2-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct) 多模态大语言模型，支持文本生成、图像理解、视频理解等任务。

---

## 硬件需求

> **在开始之前，请确认你的硬件满足以下最低要求。**

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **GPU 显存** | 4 × 40GB (如 A100-40GB) | 4 × 80GB (如 A100-80GB / H100) |
| **GPU 架构** | Ampere (SM 80+) | Ampere / Hopper |
| **系统内存** | 128GB RAM | 256GB+ RAM |
| **磁盘空间** | 200GB 可用空间 | 300GB+ SSD |
| **CUDA 版本** | CUDA 12.0+ | CUDA 13.0 |
| **操作系统** | Linux (Ubuntu 20.04+) | Ubuntu 22.04 / 24.04 |

**显存预估说明**：
- 模型权重 (bfloat16): ~137GB，4 卡 TP 分片后每卡 ~34GB
- KV Cache + 运行时开销: 每卡额外 ~30-40GB
- `gpu_memory_utilization=0.9` 时，4×80GB 可支持 max_model_len=8192
- 若仅有 4×40GB 显存，需使用 AWQ/GPTQ 量化版本（如 Qwen2-VL-72B-Instruct-AWQ）

**不同 GPU 配置参考**：

| GPU 配置 | 是否可行 | 备注 |
|----------|---------|------|
| 4 × A100-80GB | 推荐 | 本项目验证环境 |
| 8 × A100-40GB | 可行 | 需设置 `tensor_parallel_size=8` |
| 4 × A100-40GB | 不可行 | 显存不足，需使用量化版本 |
| 2 × H100-80GB | 可行 | 需设置 `tensor_parallel_size=2` |
| 8 × V100-32GB | 不可行 | 架构不支持 bfloat16，显存不足 |

---

## 1. 创建虚拟环境

### 方式 A：使用 Conda（推荐）

```bash
# 创建 Python 3.10 环境
conda create -n vllm_env python=3.10 -y
conda activate vllm_env

# 安装核心依赖
pip install -r requirements.txt

# 安装系统编译依赖 (需要 root 权限)
sudo apt-get update && sudo apt-get install -y gcc g++
```

### 方式 B：使用 venv

```bash
python3.10 -m venv vllm_env
source vllm_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

sudo apt-get update && sudo apt-get install -y gcc g++
```

### 验证安装

```bash
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"
```

预期输出：
```
vLLM version: 0.23.0
PyTorch: 2.11.0, CUDA: 13.0, GPUs: 4
```

---

## 2. 下载模型

### 配置下载路径

```bash
# 设置 HuggingFace 缓存目录（模型将下载到此路径）
export HF_HOME=./hf_cache
```

### 方式 A：使用 hf CLI 下载（推荐）

```bash
# 登录 HuggingFace（可选，登录后下载速度更快）
hf auth login

# 下载完整模型（约 137GB，预计耗时 30-60 分钟，取决于网速）
hf download Qwen/Qwen2-VL-72B-Instruct --cache-dir ./hf_cache
```

### 方式 B：使用 Python 下载

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2-VL-72B-Instruct",
    cache_dir="./hf_cache",
    resume_download=True,  # 支持断点续传
)
```

### 方式 C：使用 huggingface-cli（旧版本）

```bash
huggingface-cli download Qwen/Qwen2-VL-72B-Instruct --cache-dir ./hf_cache
```

### 验证下载完整性

```bash
# 检查模型分片数量（应有 38 个 safetensors 文件）
find ./hf_cache -name "model-*.safetensors" | wc -l

# 检查无未完成下载
find ./hf_cache -name "*.incomplete" | wc -l  # 应输出 0

# 检查总大小（应约 137GB）
du -sh ./hf_cache/hub/models--Qwen--Qwen2-VL-72B-Instruct/
```

---

## 3. 使用模型

### 必需环境变量

**每次启动前必须设置**（建议写入 `.bashrc` 或启动脚本）：

```bash
export CUDA_HOME=/path/to/your/conda/envs/vllm_env/lib/python3.10/site-packages/nvidia/cu13
export HF_HOME=./hf_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_HAS_FLASHINFER_CUBIN=1
export VLLM_USE_FLASHINFER_SAMPLER=0
```

> **CUDA_HOME 路径说明**：指向 conda 环境内 pip 安装的 nvidia-cu13 包。若你的 conda 环境路径不同，请执行以下命令获取实际路径：
> ```bash
> python -c "import nvidia.cu13; import os; print(os.path.dirname(nvidia.cu13.__file__))"
> ```

| 变量 | 作用 |
|------|------|
| `CUDA_HOME` | 指向 CUDA toolkit，使 Triton/FlashInfer 能找到 nvcc |
| `HF_HOME` | 模型缓存路径，避免重复下载 |
| `CUDA_VISIBLE_DEVICES` | 控制使用哪些 GPU |
| `VLLM_HAS_FLASHINFER_CUBIN` | 启用预编译 cubin，跳过 JIT 编译 |
| `VLLM_USE_FLASHINFER_SAMPLER` | 禁用 flashinfer sampler JIT（规避 CUDA 版本不兼容） |

---

### 3.1 启动 OpenAI 兼容 API 服务（推荐）

```bash
bash start_qwen2vl.sh
```

或手动启动：

```bash
conda activate vllm_env

export CUDA_HOME=/path/to/conda/envs/vllm_env/lib/python3.10/site-packages/nvidia/cu13
export HF_HOME=./hf_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_HAS_FLASHINFER_CUBIN=1
export VLLM_USE_FLASHINFER_SAMPLER=0

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-72B-Instruct \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 8000
```

服务就绪后 API 地址为 `http://localhost:8000/v1`。

---

### 3.2 API 调用示例

#### 文本对话

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-72B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "什么是深度学习？"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

#### 图像理解

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-72B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
          {"type": "text", "text": "请描述这张图片的内容。"}
        ]
      }
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

#### Python SDK 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-72B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "请解释什么是Transformer架构？"},
    ],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

---

### 3.3 离线批量推理（无需启动 API 服务）

```python
from vllm import LLM, SamplingParams
import os

# 环境变量（若未在 shell 中 export，需在代码中设置）
os.environ["CUDA_HOME"] = "/path/to/conda/envs/vllm_env/lib/python3.10/site-packages/nvidia/cu13"
os.environ["HF_HOME"] = "./hf_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["VLLM_HAS_FLASHINFER_CUBIN"] = "1"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# 加载模型
llm = LLM(
    model="Qwen/Qwen2-VL-72B-Instruct",
    tensor_parallel_size=4,
    dtype="bfloat16",
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
)

# 使用 Qwen chat template
prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n你好，请介绍一下自己。<|im_end|>\n"
    "<|im_start|>assistant\n"
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=512, stop=["<|im_end|>"])
outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

---

## 性能参考

在 4 × A100-80GB PCIe 环境下的实测性能：

| 指标 | 数值 |
|------|------|
| 模型加载时间 | ~153 秒 |
| 文本生成速度 | ~17-18 tok/s |
| 中文生成速度 | ~18 tok/s |
| 代码生成速度 | ~18 tok/s |
| CUDA Graph 捕获 | ~12 秒 (35 graphs) |

---

## 项目结构

```
3D-anything/
├── README.md                 # 本文件
├── requirements.txt          # Python 依赖清单
├── start_qwen2vl.sh          # API 服务启动脚本
├── inference_qwen2vl.py      # 离线推理示例 (纯文本验证)
├── scripts/
│   ├── infer_qwen2vl.py      # 多模态推理脚本 (支持单图/批量/标注关联)
│   └── run_infer.sh          # 推理启动脚本 (在此修改提示词和参数)
├── data/
│   └── DVG_sample/           # 示例图片及标注 (jpg + json)
├── results/                  # 推理输出目录 (JSONL 格式)
├── hf_cache/                 # 模型权重缓存 (不上传至 git)
│   └── hub/
│       └── models--Qwen--Qwen2-VL-72B-Instruct/
└── doc/
    └── qwen2vl_setup_log.md  # 部署过程详细日志
```

---

### 3.4 多模态推理脚本（支持批量 & CoT 数据生成）

用于 visual grounding CoT 数据生成任务。通过 bash 启动脚本直接管理提示词和参数，无需修改 Python 代码。

**前提**：先启动 API 服务（`bash start_qwen2vl.sh`），确认服务就绪。

#### 推荐用法：通过 bash 脚本一键启动

```bash
# 编辑提示词和参数
vim scripts/run_infer.sh

# 一键运行
bash scripts/run_infer.sh
```

`run_infer.sh` 中可直接修改的配置项：

```bash
IMAGE="..."           # 单张图片路径 (与 IMAGE_DIR 二选一)
IMAGE_DIR="..."       # 批量目录路径
MAX_IMAGES=10         # 最多处理数量
SYSTEM_PROMPT='...'   # 系统提示词 (控制 CoT 格式)
USER_PROMPT='...'     # 用户提示词
TEMPERATURE=0.7       # 生成温度
MAX_TOKENS=2048       # 最大生成 token 数
OUTPUT="..."          # 输出文件路径
```

#### 也可直接调用 Python 脚本

```bash
# 单张图片
python scripts/infer_qwen2vl.py \
    --image data/DVG_sample/0000000_00098_d_0000001.jpg

# 自定义提示词
python scripts/infer_qwen2vl.py \
    --image data/DVG_sample/0000000_00098_d_0000001.jpg \
    --prompt "请找出图中所有白色飞机，用 [x1, y1, x2, y2] 标注它们的位置。"

# 批量处理 + 使用标注中的 question 作为提示词
python scripts/infer_qwen2vl.py \
    --image-dir data/DVG_sample \
    --use-annotation \
    --max-images 10 \
    --output results/vg_cot_output.jsonl
```

#### 完整参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--image` | 单张图片路径（与 `--image-dir` 二选一） | — |
| `--image-dir` | 图片目录路径，批量模式（与 `--image` 二选一） | — |
| `--prompt` | 用户提示词 | visual grounding 定位提示 |
| `--system-prompt` | 系统提示词（控制 CoT 输出格式） | VG CoT 模板 |
| `--use-annotation` | 使用同名 JSON 标注文件的 question 作为提示词 | 关闭 |
| `--temperature` | 生成温度 | 0.7 |
| `--max-tokens` | 最大生成 token 数 | 2048 |
| `--max-images` | 最多处理图片数，0 = 全部 | 0 |
| `--output` | 输出 JSONL 文件路径 | `results/infer_output.jsonl` |
| `--api-base` | API 地址 | `http://localhost:8000/v1` |

#### CoT 输出格式

模型输出遵循 `<think>...</think>` + `<answer>...</answer>` 结构：

```
<think>
[观察] 图片整体场景描述...
[分析] 目标特征分析...
[推理] 空间推理与排除过程...
[定位] 坐标估算...
</think>

<answer>
目标物体: [x1, y1, x2, y2]
</answer>
```

输出为 JSONL 格式，每行一条记录，包含 `image_path`、`prompt`、`response`、`usage`、`elapsed_s` 等字段。使用 `--use-annotation` 时会额外保存完整标注信息（bbox/obb/poly）。

---

## 常见问题

### Q: 报错 `Failed to find C compiler`

安装 gcc：
```bash
sudo apt-get install -y gcc g++
```

### Q: 报错 `Could not find nvcc`

确认设置了 `CUDA_HOME` 环境变量。获取正确路径：
```bash
python -c "import nvidia.cu13; import os; print(os.path.dirname(nvidia.cu13.__file__))"
```

### Q: 报错 `CUDA compiler and CUDA toolkit headers are incompatible`

设置以下环境变量绕过 JIT 编译：
```bash
export VLLM_HAS_FLASHINFER_CUBIN=1
export VLLM_USE_FLASHINFER_SAMPLER=0
```

### Q: 报错 `Engine core initialization failed`

这是上述三个问题的最终表现，按顺序排查：gcc → CUDA_HOME → FLASHINFER_SAMPLER。

### Q: 显存不足 (OOM)

- 减小 `--max-model-len`（如 4096 → 2048）
- 减小 `--gpu-memory-utilization`（如 0.9 → 0.85）
- 使用量化版本模型：`Qwen/Qwen2-VL-72B-Instruct-AWQ`

### Q: 下载中断

`hf download` 和 `snapshot_download` 均支持断点续传，重新执行相同命令即可继续。

---

## .gitignore 建议

由于模型文件体积巨大 (137GB)，**不应上传至 Git**：

```gitignore
# Model weights (137GB+)
hf_cache/

# Python
__pycache__/
*.pyc
.venv/
vllm_env/

# System
.DS_Store
*.log
```

---

## License

本项目的代码部分为 MIT License。模型权重遵循 [Qwen License](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct/blob/main/LICENSE)。
