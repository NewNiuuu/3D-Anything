# Qwen2-VL → Qwen3.5 迁移执行手册

> 按顺序复制粘贴以下命令即可完成迁移。每一步都有预期输出，请确认后再进行下一步。

---

## Step 1: 下载新模型

```bash
cd /root/nyp/3D-anything && HF_HOME=/root/nyp/hf_cache huggingface-cli download Qwen/Qwen3.5-35B-A3B --cache-dir /root/nyp/hf_cache
```

预计耗时 15-60 分钟（取决于网速），模型约 70GB。支持断点续传，中断后重新运行同一命令即可。

### 验证下载

```bash
find /root/nyp/hf_cache -path "*Qwen3.5-35B-A3B*" -name "*.safetensors" | wc -l
```

预期输出：`14`

```bash
find /root/nyp/hf_cache -path "*Qwen3.5-35B-A3B*" -name "*.incomplete" | wc -l
```

预期输出：`0`

```bash
du -sh /root/nyp/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B/
```

预期输出：约 `70G`

---

## Step 1.5: 修复缓存路径（关键！）

`huggingface-cli download --cache-dir` 把模型放在了 `/root/nyp/hf_cache/models--Qwen--Qwen3.5-35B-A3B/`，但 vLLM 设置 `HF_HOME=/root/nyp/hf_cache` 后会在 `hub/` 子目录下查找模型。需要创建软链接：

```bash
ln -s /root/nyp/hf_cache/models--Qwen--Qwen3.5-35B-A3B /root/nyp/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B
```

### 验证软链接

```bash
ls -la /root/nyp/hf_cache/hub/ | grep Qwen3.5
```

预期输出：显示 `models--Qwen--Qwen3.5-35B-A3B -> /root/nyp/hf_cache/models--Qwen--Qwen3.5-35B-A3B`

---

## Step 2: 设置文件权限

```bash
cd /root/nyp/3D-anything && chmod +x start_qwen35.sh scripts/download_model.py scripts/download_model_curl.sh scripts/migrate_to_qwen35.sh
```

---

## Step 3: 验证离线推理

```bash
cd /root/nyp/3D-anything && \
CUDA_HOME=/root/miniconda3/envs/vllm_env/lib/python3.10/site-packages/nvidia/cu13 \
HF_HOME=/root/nyp/hf_cache \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_HAS_FLASHINFER_CUBIN=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
python inference_qwen35.py
```

预期输出：模型加载成功，生成两段回复，打印 token/s 速度。类似：

```
============================================================
Loading Qwen3.5-35B-A3B with vLLM
============================================================
✓ Model loaded in XX.Xs

--- Response 1 (xxx tokens) ---
...
--- Response 2 (xxx tokens) ---
...
============================================================
Total: xxx tokens in X.XXs (XX.X tok/s)
============================================================
```

---

## Step 4: 验证 API 服务

### 终端 1 — 启动服务（保持运行）

```bash
cd /root/nyp/3D-anything && bash start_qwen35.sh
```

等待出现 `Uvicorn running on http://0.0.0.0:8000` 后，在另一个终端执行：

### 终端 2 — 测试 API

```bash
curl http://localhost:8000/v1/models
```

预期输出：JSON 中包含 `"id": "Qwen/Qwen3.5-35B-A3B"`

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "什么是深度学习？请用两句话回答。"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

预期输出：JSON 响应，包含模型的中文回答。

### 测试多模态推理（可选）

```bash
cd /root/nyp/3D-anything && python scripts/infer_qwen35.py \
    --image data/DVG_sample/0000000_00098_d_0000001.jpg \
    --max-tokens 512
```

预期输出：模型对图片进行分析并输出 `<think>...</think><answer>...</answer>` 格式的结果。

测试完毕后可以 Ctrl+C 关闭终端 1 的 API 服务。

---

## Step 5: Git 提交

```bash
cd /root/nyp/3D-anything && \
git add start_qwen35.sh inference_qwen35.py scripts/infer_qwen35.py \
  scripts/run_infer.sh scripts/download_model.py scripts/download_model_curl.sh \
  scripts/migrate_to_qwen35.sh README.md requirements.txt doc/qwen35_setup_log.md && \
git rm start_qwen2vl.sh inference_qwen2vl.py scripts/infer_qwen2vl.py doc/qwen2vl_setup_log.md && \
git commit -m "refactor: migrate from Qwen2-VL-72B to Qwen3.5-35B-A3B

- Replace all model references: Qwen/Qwen2-VL-72B-Instruct -> Qwen/Qwen3.5-35B-A3B
- Rename scripts: start_qwen2vl.sh -> start_qwen35.sh, etc.
- Update max-model-len: 8192 -> 32768 (Qwen3.5 supports 256K)
- Update README.md with reduced hardware requirements
- Add migration docs in doc/qwen35_setup_log.md
- Qwen3.5-35B-A3B: MoE (256 experts, 8 active), multimodal, ~70GB vs 137GB

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Step 6: 删除旧模型权重

> **重要**：请确认 Step 3 和 Step 4 均通过后再执行此步！删除后无法恢复。

```bash
du -sh /root/nyp/hf_cache/hub/models--Qwen--Qwen2-VL-72B-Instruct/
```

预期输出：`137G`（确认是旧模型目录）

```bash
rm -rf /root/nyp/hf_cache/hub/models--Qwen--Qwen2-VL-72B-Instruct/
```

### 确认空间释放

```bash
du -sh /root/nyp/hf_cache/
```

预期输出：约 `70G`（仅剩新模型）

```bash
df -h /root
```

确认可用空间增加了约 137GB。

---

## 迁移完成后的日常使用

### 启动 API 服务

```bash
cd /root/nyp/3D-anything && bash start_qwen35.sh
```

### 单张图片推理

```bash
python scripts/infer_qwen35.py --image data/DVG_sample/0000000_00098_d_0000001.jpg
```

### 批量推理

```bash
bash scripts/run_infer.sh
```

### 离线推理（不需要 API 服务）

```bash
CUDA_HOME=/root/miniconda3/envs/vllm_env/lib/python3.10/site-packages/nvidia/cu13 \
HF_HOME=/root/nyp/hf_cache \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_HAS_FLASHINFER_CUBIN=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
python inference_qwen35.py
```
