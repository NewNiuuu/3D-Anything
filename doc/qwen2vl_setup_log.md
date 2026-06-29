# Qwen2-VL-72B-Instruct 部署日志

## 任务信息

- **日期**: 2026-06-28
- **目标**: 在 `/root/nyp/hf_cache` 路径下配置 Qwen2-VL-72B-Instruct 模型，使其可正常推理
- **机器**: debug-48gpu-cluster-v1-7
- **操作者环境**: tmux 终端，conda 环境 `vllm_env`

---

## 0. 虚拟环境创建记录

### 创建命令
```bash
conda create -n vllm_env python=3.10 -y
conda activate vllm_env
pip install vllm==0.23.0
# vllm 会自动安装大部分依赖（torch, flashinfer, transformers 等）
```

### 系统依赖安装
```bash
apt-get update && apt-get install -y gcc g++
# gcc 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1)
```

### 关键包版本（完整清单见 requirements.txt）
```
vllm==0.23.0
torch==2.11.0 (CUDA 13.0)
torchvision==0.26.0
flashinfer-python==0.6.12
flashinfer-cubin==0.6.12
transformers==5.12.1
huggingface_hub==1.21.0
safetensors==0.8.0
triton==3.6.0
numpy==2.2.6
```

---

## 1. 环境探查

### 硬件配置
```
GPU: 4 × NVIDIA A100 80GB PCIe (共 320GB VRAM)
Driver: NVIDIA-SMI 580.105.08
CUDA Version: 13.0
磁盘: 782GB 可用 (overlay 1008G)
```

### 软件环境
```
Python: 3.10 (miniconda3/envs/vllm_env)
vLLM: 0.23.0
huggingface_hub: 1.21.0
PyTorch CUDA: 13.0
flashinfer-python: 0.6.12
flashinfer-cubin: 0.6.12 (预编译 cubin 15558 个)
nvcc (conda env): 13.2 (V13.2.78)
```

---

## 2. 模型下载状态

检查发现模型已经**完整下载**：

```
路径: /root/nyp/hf_cache/hub/models--Qwen--Qwen2-VL-72B-Instruct/
Commit: 2ac26c967836fbb5729c709ad8f8b5548e1f88aa
总大小: 137GB
分片数: 38 个 safetensors 文件
配置文件: config.json, tokenizer.json, preprocessor_config.json 等
状态: 所有 symlink 有效，无 .incomplete 文件
```

---

## 3. 问题诊断过程

### 3.1 原始报错

```
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

此错误信息不明确，需要深入追踪。

### 3.2 第一次复现 — 缺少 C 编译器

```
RuntimeError: Worker failed with error 'Failed to find C compiler.
Please specify via CC environment variable or set triton.knobs.build.impl'
```

**原因**: flashinfer / triton 需要 JIT 编译 GPU kernel，但系统未安装 gcc。

**修复**:
```bash
apt-get update && apt-get install -y gcc g++
```

安装结果：gcc 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1)

### 3.3 第二次复现 — 缺少 nvcc

```
RuntimeError: Worker failed with error 'Could not find nvcc and
default cuda_home='/usr/local/cuda' doesn't exist'
```

**原因**: 系统没有全局 CUDA toolkit 安装 (`/usr/local/cuda` 不存在)，nvcc 仅存在于 conda 环境内。

**修复**: 设置 `CUDA_HOME` 指向 conda 包中的 CUDA toolkit：
```bash
export CUDA_HOME=/root/miniconda3/envs/vllm_env/lib/python3.10/site-packages/nvidia/cu13
```

### 3.4 第三次复现 — CUDA 版本不兼容

```
error: #error "CUDA compiler and CUDA toolkit headers are incompatible,
please check your include paths"
```

**原因**: flashinfer 的 CCCL headers 要求 CUDA 13.0，但 conda 环境内的 nvcc 是 13.2。flashinfer 尝试 JIT 编译 sampling kernel 时触发此错误。

**分析**:
- PyTorch CUDA runtime: 13.0
- Driver CUDA: 13.0
- nvcc (nvidia-cu13 pip package): 13.2
- flashinfer CCCL headers 内部版本检查失败

### 3.5 最终解决方案

```bash
export VLLM_HAS_FLASHINFER_CUBIN=1    # 使用预编译的 cubin 文件
export VLLM_USE_FLASHINFER_SAMPLER=0   # 禁用需要 JIT 编译的 sampler
```

设置 `VLLM_USE_FLASHINFER_SAMPLER=0` 后，flashinfer 的 sampling kernel 不再 JIT 编译，从而完全绕过 CUDA 版本不兼容问题。预编译的 cubin 文件 (15558 个) 已由 `flashinfer-cubin` 包提供。

---

## 4. 验证结果

### 成功启动日志
```
Model loaded in 152.6s
Capturing CUDA graphs (decode, FULL): 100% | 35/35
```

### 推理测试

| 测试项 | 输入 | Token 数 | 耗时 | 速度 |
|--------|------|----------|------|------|
| 英文生成 | "Hello, who are you?" | 200 | 11.73s | 17.0 tok/s |
| 中文生成 | "什么是深度学习？" | 70 | 3.85s | 18.2 tok/s |
| 代码生成 | "fibonacci function" | 57 | 3.20s | 17.8 tok/s |

### 生成示例

**中文回答**:
> 深度学习是一种人工智能技术，它使用多层神经网络来模拟人类大脑的工作方式，通过大量的数据训练来自动提取特征和模式，并用于解决各种问题，如图像识别、语音识别、自然语言处理等。

**代码生成**:
```python
def fibonacci(n):
    fib = [0, 1] + [0] * (n-1)
    for i in range(2, n+1):
        fib[i] = fib[i-1] + fib[i-2]
    return fib[n]
```

---

## 5. 最终配置汇总

### 必需环境变量
```bash
export CUDA_HOME=/root/miniconda3/envs/vllm_env/lib/python3.10/site-packages/nvidia/cu13
export HF_HOME=/root/nyp/hf_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_HAS_FLASHINFER_CUBIN=1
export VLLM_USE_FLASHINFER_SAMPLER=0
```

### 系统依赖（已安装）
```bash
apt-get install -y gcc g++
```

### 生成的脚本文件
- `/root/nyp/3D-anything/start_qwen2vl.sh` — OpenAI 兼容 API 服务启动脚本
- `/root/nyp/3D-anything/inference_qwen2vl.py` — 离线推理示例（纯文本验证）
- `/root/nyp/3D-anything/scripts/infer_qwen2vl.py` — 多模态推理脚本（支持图片输入、批量处理、标注关联）
- `/root/nyp/3D-anything/scripts/run_infer.sh` — 推理启动入口（在此修改提示词和参数）

---

## 6. Visual Grounding CoT 数据生成

### 使用流程

```
1. 启动 API 服务:     bash start_qwen2vl.sh (终端 1，保持运行)
2. 编辑提示词/参数:   vim scripts/run_infer.sh
3. 运行推理:          bash scripts/run_infer.sh (终端 2)
4. 查看结果:          results/infer_output.jsonl
```

### CoT 输出格式

通过系统提示词强制模型输出结构化思维链：

```
<think>
[观察] 图片整体内容描述...
[分析] 目标视觉特征分析...
[推理] 空间推理与排除...
[定位] 坐标估算...
</think>

<answer>
[x1, y1, x2, y2]
</answer>
```

### 关键设计决策

- **系统提示词使用中文** + 结构化标签 (`<think>/<answer>`)，确保模型不跳过推理步骤
- **坐标要求像素值**，避免归一化坐标歧义
- **max_tokens 设为 2048**，CoT 输出较长需要足够空间
- **JSONL 输出格式**，便于后续清洗和训练数据组装

---

## 7. 已知限制与建议

1. **nvcc 版本不匹配**: conda 环境中 nvidia-cu13 包自带 nvcc 13.2 但 flashinfer headers 预期 13.0。当前通过禁用 JIT sampler 规避，若后续需要 flashinfer 的全部功能，需降级 nvcc 或升级 flashinfer。
2. **HuggingFace 未登录**: 模型下载可匿名完成，但会有速率限制。如需更快下载建议 `hf auth login`。
3. **建议长期方案**: 若环境重建，建议安装匹配版本的全局 CUDA toolkit (`/usr/local/cuda`) 以彻底解决编译问题。
