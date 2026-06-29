# 迁移进度文档 — Qwen2-VL-72B → Qwen3.5-35B-A3B

> **迁移已完成** ✅ (2026-06-29)

---

## 当前状态：✅ 迁移完成

### 已完成 ✅

1. **代码重构** — 所有脚本文件已就绪
2. **模型下载** — Qwen3.5-35B-A3B 已下载至 `/root/nyp/hf_cache/models--Qwen--Qwen3.5-35B-A3B`（67GB）
3. **文件权限** — 已 chmod +x
4. **缓存路径修复** — 已创建符号链接 `/root/nyp/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B` → 实际路径
5. **离线推理验证** — ✅ 通过（1024 tokens / 7.88s = 130.0 tok/s）
6. **API 服务验证** — ✅ 通过（vLLM OpenAI-compatible server 正常响应）
7. **Git 提交** — ✅ 已提交 (commit `92926a9`)
8. **删除旧模型** — ⚠️ 需手动执行：`rm -rf /root/nyp/hf_cache/hub/models--Qwen--Qwen2-VL-72B-Instruct/`

---

## 已解决的阻塞问题

### 原错误

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

### 解决方案（方案 A+B 组合）

1. 创建 `/usr/local/cuda` → `/root/miniconda3/envs/vllm_env/lib/python3.10/site-packages/nvidia/cu13` 符号链接
2. 在脚本中 `export CUDA_HOME` 和 `CUDA_PATH`，确保 multiprocessing spawn 子进程继承
3. vLLM 使用 `spawn` 模式（非 fork），环境变量通过 `export` 正确传播

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

## 验证结果

### 离线推理 (inference_qwen35.py)

```
✓ Model loaded (4×A100, tensor_parallel_size=4)
✓ 2 prompts generated: 1024 tokens in 7.88s (130.0 tok/s)
✓ 中文回答质量正常（Transformer 架构解释）
✓ 英文代码生成正常（merge sorted lists）
```

### API 服务 (start_qwen35.sh)

```
✓ vLLM OpenAI-compatible server started on 0.0.0.0:8000
✓ GET /v1/models → 200 OK, model: Qwen/Qwen3.5-35B-A3B
✓ POST /v1/chat/completions → 200 OK, 正常生成
✓ system_fingerprint: vllm-0.23.0-tp4-36d6207c
```

---

## 使用方法

### 离线推理

```bash
conda activate vllm_env
cd /root/nyp/3D-anything
bash start_qwen35.sh  # 或直接运行 inference_qwen35.py
```

### API 服务

```bash
conda activate vllm_env
cd /root/nyp/3D-anything
bash start_qwen35.sh
# 然后用 curl 或 OpenAI SDK 访问 http://localhost:8000/v1/
```

---

## 恢复任务清单

~~下次 Claude session 应：~~
1. ~~阅读本文件恢复上下文~~ ✅
2. ~~按"下一步应尝试的方案"顺序修复 flashinfer/nvcc 问题~~ ✅ (方案 A+B)
3. ~~验证离线推理成功~~ ✅ (130 tok/s)
4. ~~验证 API 服务启动成功~~ ✅
5. ~~执行 git commit~~ ✅ (commit 92926a9)
6. ~~删除旧模型权重~~ ⚠️ 需手动执行
7. ~~更新本文件标记完成~~ ✅

**所有迁移步骤完成。** 唯一剩余操作：手动删除旧模型文件释放 137GB 磁盘空间。
