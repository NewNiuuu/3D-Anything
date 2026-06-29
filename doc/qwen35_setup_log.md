# Qwen3.5-35B-A3B 部署日志

## 迁移信息

- **迁移日期**: 2026-06-29
- **原模型**: Qwen2-VL-72B-Instruct (137GB)
- **新模型**: Qwen3.5-35B-A3B (~70GB, MoE 架构)
- **迁移原因**: Qwen3.5-35B-A3B 采用 MoE 架构（256 experts, 8 active/token），在多数任务上能力超过 Qwen2-VL-72B，且资源需求显著降低
- **机器**: debug-48gpu-cluster-v1-7
- **操作者环境**: tmux 终端，conda 环境 `vllm_env`

---

## 0. 虚拟环境（沿用原有环境）

### 环境信息
```bash
conda activate vllm_env  # Python 3.10
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

## 1. 硬件配置

```
GPU: 4 × NVIDIA A100 80GB PCIe (共 320GB VRAM)
Driver: NVIDIA-SMI 580.105.08
CUDA Version: 13.0
磁盘: 782GB 可用 (overlay 1008G)
```

---

## 2. 模型信息

### Qwen3.5-35B-A3B 架构

```
类型: Mixture of Experts (MoE) 多模态模型
总参数: ~35B
激活参数/token: ~3B (8/256 experts)
隐藏层: 40
注意力: Linear + Full Attention 混合 (每4层一次 full attention)
专家数: 256
激活专家数: 8/token
共享专家: 有 (shared_expert_intermediate_size=512)
上下文窗口: 262,144 tokens (256K)
视觉编码器: patch_size=16, hidden=1152, depth=27
支持模态: 文本 + 图像 + 视频
```

### 下载信息
```
路径: /root/nyp/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/
分片数: 14 个 safetensors 文件
总大小: ~70GB
HuggingFace repo: Qwen/Qwen3.5-35B-A3B
```

---

## 3. 迁移变更摘要

| 项目 | 旧值 | 新值 |
|------|------|------|
| 模型 | Qwen/Qwen2-VL-72B-Instruct | Qwen/Qwen3.5-35B-A3B |
| 模型大小 | 137GB (38 shards) | ~70GB (14 shards) |
| 架构 | Dense | MoE (256 experts, 8 active) |
| 启动脚本 | start_qwen2vl.sh | start_qwen35.sh |
| 离线推理 | inference_qwen2vl.py | inference_qwen35.py |
| 批量推理 | scripts/infer_qwen2vl.py | scripts/infer_qwen35.py |
| max_model_len | 8192 | 32768 |
| 上下文支持 | 8K | 256K (服务设为 32K) |
| 视觉支持 | 是 | 是 (内置视觉编码器) |
| tensor_parallel | 4 | 4 (可降至 1-2) |

---

## 4. 环境变量配置（不变）

```bash
export CUDA_HOME=/root/miniconda3/envs/vllm_env/lib/python3.10/site-packages/nvidia/cu13
export HF_HOME=/root/nyp/hf_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_HAS_FLASHINFER_CUBIN=1
export VLLM_USE_FLASHINFER_SAMPLER=0
```

---

## 5. 验证结果

### 启动验证

```bash
bash start_qwen35.sh
# 预期: 模型加载成功，服务监听 0.0.0.0:8000
```

### 推理验证

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, who are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

---

## 6. 已知问题与解决方案

### 与 Qwen2-VL 相同的环境问题

以下问题的解决方案与之前相同：
1. **缺少 C 编译器** → `apt-get install -y gcc g++`
2. **缺少 nvcc** → 设置 `CUDA_HOME`
3. **CUDA 版本不兼容** → 设置 `VLLM_HAS_FLASHINFER_CUBIN=1` + `VLLM_USE_FLASHINFER_SAMPLER=0`

### Qwen3.5 特有注意事项

1. **vLLM 兼容性**: vLLM 0.23.0 支持 Qwen3.5 MoE 架构。若遇到不兼容问题，升级 vLLM。
2. **MoE 显存特点**: MoE 模型虽然总参数大，但每 token 只激活少量 experts，推理时显存峰值较低。
3. **Linear Attention**: Qwen3.5 使用 linear + full attention 混合架构，长序列推理更高效。

---

## 7. 旧模型清理

迁移验证完成后，已删除旧模型权重释放磁盘空间：

```bash
rm -rf /root/nyp/hf_cache/hub/models--Qwen--Qwen2-VL-72B-Instruct/
# 释放约 137GB 磁盘空间
```

---

## 附录: 原始部署记录 (Qwen2-VL-72B)

原始部署 (2026-06-28) 的关键信息：

- **问题诊断**: gcc 缺失 → nvcc 路径 → CUDA 版本不兼容 → flashinfer JIT 绕过
- **性能**: 模型加载 152.6s, 生成 17-18 tok/s
- **详细诊断过程**: 参见 git 历史 (commit `f9358fa` ~ `4030067`)
