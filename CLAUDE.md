# CLAUDE.md — 项目上下文与规则

> Claude 每次启动新 session 时应首先阅读此文件恢复上下文。

---

## 项目概述

**3D-Anything** — 基于 Qwen3.5-35B-A3B 多模态大模型的数据标注生成项目。通过 vLLM 部署模型 API，批量处理遥感/无人机图片，生成 4 类训练数据：

1. **Grounding-CoT** — 视觉定位 + 思维链（两步流程：先生成问题，再生成 CoT 回答）
2. **Caption** — 场景详细描述
3. **Decomposition** — 高层任务→低层动作分解
4. **Dialog** — 4-10 轮人机多轮对话

---

## 关键规则 ⚠️

### 1. 每次完成新需求后，必须更新文档

完成任何功能开发、bug 修复或流程变更后，**必须同步更新以下文件**：

| 文件 | 更新内容 |
|------|----------|
| `CLAUDE.md`（本文件） | 项目状态、架构变更、新增文件/目录、新的注意事项 |
| `docs/progress_log.md` | 任务完成记录、测试结果、下一步待办 |
| `README.md` | 仅当用户可见的使用方式发生变化时更新 |

### 2. 提交规范

- 使用语义化 commit message：`feat:` / `fix:` / `docs:` / `chore:` / `refactor:`
- 每次 commit 末尾添加 `Co-Authored-By: Claude <noreply@anthropic.com>`
- 推送前确认 `git status` 干净

### 3. 推送到 GitHub

- Remote: `https://github.com/NewNiuuu/3D-Anything.git`
- 推送时临时设置 token URL，推送后立即恢复为无 token 的 URL
- SSH key 不可用（Permission denied），只能用 HTTPS + token

### 4. 不要做的事

- 不要提交 `results/` 下的 jsonl 输出文件（已 gitignore）
- 不要提交 `data/DVGBench/images/` 下的图片（已 gitignore，691MB）
- 不要提交 `docs/superpowers/` 目录
- 不要提交 `hf_cache/` 目录
- 不要修改 `.gitignore` 中对大文件的排除规则除非用户明确要求

---

## 当前项目状态

### 模型服务

- **模型**: Qwen/Qwen3.5-35B-A3B (MoE, 256 experts, 8 active)
- **部署**: vLLM 0.23.0, tensor_parallel_size=4, 4×A100-80GB
- **启动**: `bash start_qwen35.sh` → API at `http://localhost:8000/v1`
- **环境**: conda env `vllm_env`, Python 3.10
- **关键环境变量**: `CUDA_HOME`, `CUDA_PATH`, `HF_HOME`, `VLLM_HAS_FLASHINFER_CUBIN=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`

### 数据

- **DVGBench 完整数据集**: `data/DVGBench/` (2242 张: ERA 759 + VisDrone 1483)
- **标注文件**: `data/DVGBench/dvg_test.jsonl` (872 条)
- **切片样本**: `data/DVG_sample/` (100 张，含配对 json 标注)

### 数据生成脚本

```
scripts/data_gen/
├── gen_visual_grounding_qa.sh   # Step 1: 生成 VG 问题
├── gen_grounding_cot.sh         # Step 2: 生成 CoT 定位回答
├── gen_caption.sh               # 场景描述
├── gen_decomposition.sh         # 任务分解
└── gen_dialog.sh                # 多轮对话
```

默认输入: `data/DVGBench/images/visdrone`
输出目录: `results/{grounding_cot,caption,decomposition,dialog}/`

### 测试状态

所有 5 个脚本已通过 10 张图片测试（2026-06-29），全部 10/10 成功。

### 已知问题

- 模型输出包含 thinking 前缀（"The user wants..."），可能需要后处理去除
- Grounding CoT 第二步依赖标注 json 中的 question 字段，DVGBench 原始图片目录下没有配对 json

---

## 文件结构

```
3D-anything/
├── CLAUDE.md                     # ← 本文件
├── README.md                     # 用户文档
├── start_qwen35.sh               # API 服务启动
├── inference_qwen35.py           # 离线推理验证
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── infer_qwen35.py           # 多模态推理核心脚本
│   ├── run_infer.sh              # 通用推理入口
│   ├── download_model.py
│   ├── download_model_curl.sh
│   └── data_gen/                 # 数据生成脚本集
├── data/
│   ├── DVG_sample/               # 切片数据 (100张, git tracked)
│   └── DVGBench/                 # 完整数据集 (2242张, images/ gitignored)
│       ├── dvg_test.jsonl        # 标注 (git tracked)
│       └── images/{era,visdrone}/
├── results/                      # 生成输出 (gitignored)
│   ├── grounding_cot/
│   ├── caption/
│   ├── decomposition/
│   └── dialog/
├── hf_cache/                     # 模型缓存 (gitignored)
└── docs/
    ├── progress_log.md           # 进度日志
    └── qwen35_setup_log.md       # 部署日志
```

---

## 常用命令

```bash
# 启动模型服务
bash start_qwen35.sh

# 检查服务状态
curl http://localhost:8000/v1/models

# 运行数据生成（先启动服务）
bash scripts/data_gen/gen_caption.sh
bash scripts/data_gen/gen_decomposition.sh
bash scripts/data_gen/gen_dialog.sh
bash scripts/data_gen/gen_visual_grounding_qa.sh
bash scripts/data_gen/gen_grounding_cot.sh

# 推送到 GitHub
git remote set-url origin https://<TOKEN>@github.com/NewNiuuu/3D-Anything.git
git push origin main
git remote set-url origin https://github.com/NewNiuuu/3D-Anything.git
```

---

## 变更历史

| 日期 | 变更 |
|------|------|
| 2026-06-29 | 初始部署 Qwen3.5-35B-A3B，迁移自 Qwen2-VL-72B |
| 2026-06-29 | 创建 4 类数据生成脚本，10 张测试通过 |
| 2026-06-29 | 准备 DVGBench 完整数据集 (2242 张) |
| 2026-06-29 | 创建 CLAUDE.md 和 progress_log.md |
