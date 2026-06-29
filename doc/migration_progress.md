# 迁移进度文档 — Qwen2-VL-72B → Qwen3.5-35B-A3B

> **下次打开 Claude 时，请让它先阅读此文件恢复上下文。**
> 路径：`/root/nyp/3D-anything/doc/migration_progress.md`

---

## 当前状态：Step 3 卡住（推理验证失败）

### 已完成 ✅

1. **代码重构** — 所有脚本文件已就绪
2. **模型下载** — Qwen3.5-35B-A3B 已下载至 `/root/nyp/hf_cache/models--Qwen--Qwen3.5-35B-A3B`（67GB）
3. **文件权限** — 已 chmod +x
4. **缓存路径修复** — 已创建符号链接 `/root/nyp/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B` → 实际路径

### 未完成 ❌

5. **离线推理验证** — 失败（见下方"当前阻塞问题"）
6. **API 服务验证** — 未执行
7. **Git 提交** — 未执行
8. **删除旧模型** — 未执行（旧模型仍在 `/root/nyp/hf_cache/hub/models--Qwen--Qwen2-VL-72B-Instruct/`，137GB）

---

## 当前阻塞问题

### 错误信息

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

出现在 `flashinfer.sampling.top_k_mask_logits` → `get_sampling_module()` → JIT build 阶段。

### 已尝试但无效的方案

| # | 方案 | 结果 |
|---|------|------|
| 1 | `VLLM_USE_FLASHINFER_SAMPLER=0` | 对旧 Qwen2-VL 有效，但 Qwen3.5 走 vLLM v1 引擎路径，该变量无效 |
| 2 | `CUDA_HOME=.../nvidia/cu13` 环境变量 | Worker 子进程没继承到该变量（multiprocessing fork） |
| 3 | `ln -s .../nvidia/cu13 /usr/local/cuda` | 用户尝试后仍报错（未确认是新错误还是同一错误，需要重新测试获取新日志） |

### 根因分析

vLLM v1 引擎中的 `topk_topp_sampler.py` 调用 `flashinfer.sampling.top_k_top_p_sampling_from_logits`，这触发了 `flashinfer/jit/cpp_ext.py:get_cuda_path()` 的 JIT 编译流程。该函数：

1. 先查 `CUDA_HOME` 环境变量
2. 再查 `CUDA_PATH` 环境变量
3. 最后 fallback 到 `/usr/local/cuda`
4. 以上都找不到就报错

问题是 vLLM 用 multiprocessing 启动 worker 子进程，`CUDA_HOME` 可能没被子进程继承。

### 下一步应尝试的方案（按优先级）

**方案 A：确保环境变量对所有子进程可见**

```bash
export CUDA_HOME=/root/miniconda3/envs/vllm_env/lib/python3.10/site-packages/nvidia/cu13
export CUDA_PATH=/root/miniconda3/envs/vllm_env/lib/python3.10/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
cd /root/nyp/3D-anything
HF_HOME=/root/nyp/hf_cache CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_HAS_FLASHINFER_CUBIN=1 VLLM_USE_FLASHINFER_SAMPLER=0 python inference_qwen35.py
```

关键：用 `export` 而非行内变量，确保 fork 出的子进程也能看到。

**方案 B：创建 /usr/local/cuda 符号链接（如果方案 A 无效）**

```bash
rm -f /usr/local/cuda
ln -s /root/miniconda3/envs/vllm_env/lib/python3.10/site-packages/nvidia/cu13 /usr/local/cuda
```

然后确认链接有效：`ls /usr/local/cuda/bin/nvcc`

**方案 C：使用 vLLM v0 引擎绕过（如果方案 A/B 无效）**

```bash
export VLLM_USE_V1=0
```

v0 引擎的 sampler 路径不强制使用 flashinfer JIT sampling。

**方案 D：如果报 "CUDA compiler and CUDA toolkit headers are incompatible"**

这是创建了 /usr/local/cuda 后的第二层问题（nvcc 13.2 vs flashinfer headers 预期 13.0）。此时只能用方案 C（回退 v0 引擎）或升级 flashinfer。

---

## 项目文件结构

```
/root/nyp/3D-anything/
├── start_qwen35.sh              # [NEW] API 服务启动脚本
├── start_qwen2vl.sh             # [DEPRECATED] 已改为退出提示
├── inference_qwen35.py          # [NEW] 离线推理脚本
├── inference_qwen2vl.py         # [DEPRECATED] 已改为退出提示
├── README.md                    # [UPDATED] 已更新为 Qwen3.5
├── requirements.txt             # [UPDATED] 注释已更新
├── scripts/
│   ├── infer_qwen35.py          # [NEW] 多模态批量推理
│   ├── infer_qwen2vl.py         # [DEPRECATED] 已改为退出提示
│   ├── run_infer.sh             # [UPDATED] 指向 infer_qwen35.py
│   ├── download_model.py        # [NEW] Python 下载脚本
│   ├── download_model_curl.sh   # [NEW] Curl 下载脚本
│   └── migrate_to_qwen35.sh    # [NEW] 一键迁移脚本（需更新）
├── doc/
│   ├── migration_commands.md    # [NEW] 手动执行命令手册
│   ├── migration_progress.md    # [NEW] 本文件
│   ├── qwen35_setup_log.md      # [NEW] 新部署日志
│   └── qwen2vl_setup_log.md    # [DEPRECATED] 已加废弃标注
├── docs/superpowers/plans/
│   └── 2026-06-29-migrate-qwen2vl-to-qwen35.md  # 实施计划
├── data/DVG_sample/             # 测试数据（jpg + json）
├── results/                     # 推理输出目录
└── .gitignore                   # 不需要修改
```

## 模型路径

```
新模型实际位置: /root/nyp/hf_cache/models--Qwen--Qwen3.5-35B-A3B/
新模型符号链接: /root/nyp/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B -> 实际位置
旧模型位置:     /root/nyp/hf_cache/hub/models--Qwen--Qwen2-VL-72B-Instruct/ (137GB, 待删除)
```

## 模型配置关键信息

```json
{
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "model_type": "qwen3_5_moe",
  "text_config": {
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 40,
    "hidden_size": 2048,
    "max_position_embeddings": 262144
  },
  "vision_config": {
    "depth": 27,
    "hidden_size": 1152,
    "patch_size": 16
  }
}
```

模型支持多模态（文本+图像+视频），Chat template 与 Qwen2 相同（ChatML 格式）。

## 环境信息

```
机器: debug-48gpu-cluster-v1-7
GPU: 4 × NVIDIA A100 80GB PCIe
Driver: NVIDIA-SMI 580.105.08, CUDA 13.0
Python: 3.10 (conda env: vllm_env)
vLLM: 0.23.0
flashinfer-python: 0.6.12
flashinfer-cubin: 0.6.12
nvcc (conda): 13.2
```

## Git 状态

**未提交**。所有变更在工作区（unstaged）：
- 7 个已修改文件
- 多个新文件待 add

提交命令（验证通过后执行）：

```bash
cd /root/nyp/3D-anything
git add start_qwen35.sh inference_qwen35.py scripts/infer_qwen35.py scripts/run_infer.sh scripts/download_model.py scripts/download_model_curl.sh scripts/migrate_to_qwen35.sh README.md requirements.txt doc/qwen35_setup_log.md doc/migration_commands.md doc/migration_progress.md
git rm start_qwen2vl.sh inference_qwen2vl.py scripts/infer_qwen2vl.py doc/qwen2vl_setup_log.md
git commit -m "refactor: migrate from Qwen2-VL-72B to Qwen3.5-35B-A3B

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## 验证通过后的最后一步

```bash
rm -rf /root/nyp/hf_cache/hub/models--Qwen--Qwen2-VL-72B-Instruct/
```

---

## 恢复任务清单

下次 Claude session 应：
1. 阅读本文件恢复上下文
2. 按"下一步应尝试的方案"顺序修复 flashinfer/nvcc 问题
3. 验证离线推理成功
4. 验证 API 服务启动成功
5. 执行 git commit
6. 删除旧模型权重
7. 更新本文件标记完成
