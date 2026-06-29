# Qwen3.5-35B-A3B 部署与推理

基于 vLLM 框架部署 [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) 多模态大语言模型（MoE 架构，256 experts / 8 active），支持文本生成、图像理解、视频理解等任务。

---

## 硬件需求

> **在开始之前，请确认你的硬件满足以下最低要求。**

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **GPU 显存** | 1 × 80GB (如 A100-80GB) | 4 × 80GB (如 A100-80GB / H100) |
| **GPU 架构** | Ampere (SM 80+) | Ampere / Hopper |
| **系统内存** | 64GB RAM | 128GB+ RAM |
| **磁盘空间** | 100GB 可用空间 | 150GB+ SSD |
| **CUDA 版本** | CUDA 12.0+ | CUDA 13.0 |
| **操作系统** | Linux (Ubuntu 20.04+) | Ubuntu 22.04 / 24.04 |

**显存预估说明**：
- 模型权重 (bfloat16): ~70GB，4 卡 TP 分片后每卡 ~18GB
- MoE 架构: 256 experts, 每 token 激活 8 个, 实际推理显存需求较低
- KV Cache + 运行时开销: 每卡额外 ~15-25GB
- `gpu_memory_utilization=0.9` 时，4×80GB 可支持 max_model_len=32768
- 单卡 80GB 即可运行（使用 `tensor_parallel_size=1`）

**不同 GPU 配置参考**：

| GPU 配置 | 是否可行 | 备注 |
|----------|---------|------|
| 4 × A100-80GB | 推荐 | 本项目验证环境，最佳吞吐 |
| 2 × A100-80GB | 可行 | 设置 `tensor_parallel_size=2` |
| 1 × A100-80GB | 可行 | 设置 `tensor_parallel_size=1`，上下文长度需缩小 |
| 4 × A100-40GB | 可行 | 设置 `tensor_parallel_size=4` |
| 2 × H100-80GB | 推荐 | 设置 `tensor_parallel_size=2`，性能最佳 |
| 8 × V100-32GB | 不可行 | 架构不支持 bfloat16 |

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

# 下载完整模型（约 70GB，预计耗时 15-30 分钟，取决于网速）
hf download Qwen/Qwen3.5-35B-A3B --cache-dir ./hf_cache
```

### 方式 B：使用 Python 下载

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-35B-A3B",
    cache_dir="./hf_cache",
    resume_download=True,  # 支持断点续传
)
```

### 方式 C：使用项目内下载脚本

```bash
python scripts/download_model.py
```

### 验证下载完整性

```bash
# 检查模型分片数量（应有 14 个 safetensors 文件）
find ./hf_cache -name "model-*.safetensors" | wc -l

# 检查无未完成下载
find ./hf_cache -name "*.incomplete" | wc -l  # 应输出 0

# 检查总大小（应约 70GB）
du -sh ./hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/
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
bash start_qwen35.sh
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
    --model Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
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
    "model": "Qwen/Qwen3.5-35B-A3B",
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
    "model": "Qwen/Qwen3.5-35B-A3B",
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
    model="Qwen/Qwen3.5-35B-A3B",
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
    model="Qwen/Qwen3.5-35B-A3B",
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

在 4 × A100-80GB PCIe 环境下的预期性能（MoE 模型推理效率高）：

| 指标 | 数值 |
|------|------|
| 模型加载时间 | ~60-90 秒 |
| 文本生成速度 | ~40-60 tok/s |
| 模型大小 (bf16) | ~70GB |
| 每卡显存占用 (tp=4) | ~18GB weights + KV cache |
| 支持上下文长度 | 256K (服务设为 32K) |

---

## 项目结构

```
3D-anything/
├── README.md                     # 本文件
├── requirements.txt              # Python 依赖清单
├── start_qwen35.sh               # API 服务启动脚本
├── inference_qwen35.py           # 离线推理示例 (纯文本验证)
├── scripts/
│   ├── infer_qwen35.py           # 多模态推理脚本 (支持单图/批量/标注关联)
│   ├── run_infer.sh              # 通用推理启动脚本
│   ├── download_model.py         # 模型下载脚本 (Python)
│   ├── download_model_curl.sh    # 模型下载脚本 (curl)
│   └── data_gen/                 # 数据生成脚本集
│       ├── gen_visual_grounding_qa.sh   # [Step 1] 问题生成 (Q-jsonl)
│       ├── gen_grounding_cot.sh         # [Step 2] Grounding-CoT 生成
│       ├── gen_caption.sh               # Caption 场景描述生成
│       ├── gen_decomposition.sh         # Task Decomposition 生成
│       └── gen_dialog.sh               # Multi-round Dialog 生成
├── data/
│   ├── DVG_sample/               # DVGBench 切片数据 (100 张, 含标注 json)
│   └── DVGBench/                 # DVGBench 完整数据集 (2242 张)
│       ├── dvg_test.jsonl        # 872 条标注 (bbox/obb/poly/question)
│       └── images/
│           ├── era/              # 759 张 ERA 遥感图片
│           └── visdrone/         # 1483 张 VisDrone 无人机图片
├── results/                      # 推理输出目录
│   ├── grounding_cot/            # Grounding-CoT 输出
│   ├── caption/                  # Caption 输出
│   ├── decomposition/            # Decomposition 输出
│   └── dialog/                   # Dialog 输出
├── hf_cache/                     # 模型权重缓存 (不上传至 git)
└── docs/
    └── qwen35_setup_log.md       # 部署过程详细日志
```

---

### 3.4 数据生成流水线

项目提供 4 类数据生成脚本，用于构建多模态训练数据集：

| 任务 | 脚本 | 输出目录 | 说明 |
|------|------|----------|------|
| **Grounding Q** | `gen_visual_grounding_qa.sh` | `results/grounding_cot/` | 模型读图生成定位问题 (Q-jsonl) |
| **Grounding CoT** | `gen_grounding_cot.sh` | `results/grounding_cot/` | 依据问题生成思维链定位回答 |
| **Caption** | `gen_caption.sh` | `results/caption/` | 生成场景详细描述 |
| **Decomposition** | `gen_decomposition.sh` | `results/decomposition/` | 生成高层任务→低层动作分解 |
| **Dialog** | `gen_dialog.sh` | `results/dialog/` | 生成 4-10 轮人机多轮对话 |

#### 使用方法

```bash
# 前提：先启动 API 服务
bash start_qwen35.sh

# 等待服务就绪后，运行各数据生成脚本
bash scripts/data_gen/gen_caption.sh
bash scripts/data_gen/gen_decomposition.sh
bash scripts/data_gen/gen_dialog.sh

# Grounding-CoT 分两步执行
bash scripts/data_gen/gen_visual_grounding_qa.sh   # Step 1: 生成问题
bash scripts/data_gen/gen_grounding_cot.sh         # Step 2: 生成思维链回答
```

#### Grounding-CoT 两步流程

```
┌─────────────────┐     Q-jsonl      ┌─────────────────┐
│  gen_visual_    │ ───────────────►  │  gen_grounding_ │
│  grounding_qa   │   questions       │  cot            │
│  (模型读图提问) │                   │  (模型读图+CoT) │
└─────────────────┘                   └─────────────────┘
        │                                      │
        ▼                                      ▼
  questions_output.jsonl              grounding_cot_output.jsonl
```

#### 配置说明

每个脚本顶部的「配置区域」可直接修改：

```bash
IMAGE_DIR="data/DVG_sample"   # 输入图片目录
MAX_IMAGES=0                   # 处理数量限制（0=全部）
TEMPERATURE=0.7                # 生成温度
MAX_TOKENS=2048                # 最大 token 数
OUTPUT="results/xxx/output.jsonl"  # 输出路径
API_BASE="http://localhost:8000/v1"  # API 地址
```

---

## 模型架构说明

Qwen3.5-35B-A3B 采用混合专家 (MoE) 架构：

| 特性 | 数值 |
|------|------|
| 总参数量 | ~35B |
| 激活参数量/token | ~3B (8/256 experts) |
| 专家数量 | 256 |
| 每 token 激活专家数 | 8 |
| 隐藏层数 | 40 |
| 注意力机制 | Linear + Full Attention 混合 |
| 上下文窗口 | 262,144 tokens (256K) |
| 视觉编码器 | Patch size 16, hidden 1152 |
| 支持模态 | 文本 + 图像 + 视频 |

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

- 减小 `--max-model-len`（如 32768 → 16384 → 8192）
- 减小 `--gpu-memory-utilization`（如 0.9 → 0.85）
- 增加 `--tensor-parallel-size`（使用更多 GPU）

### Q: 下载中断

`hf download` 和 `snapshot_download` 均支持断点续传，重新执行相同命令即可继续。

---

## .gitignore 建议

由于模型文件体积较大 (~70GB)，**不应上传至 Git**：

```gitignore
# Model weights (~70GB)
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

本项目的代码部分为 MIT License。模型权重遵循 [Qwen License](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/LICENSE)。
