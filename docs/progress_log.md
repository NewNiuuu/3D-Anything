# 项目进度日志

> 最后更新：2026-06-29

---

## 当前状态：数据生成流水线已就绪，等待正式批量运行

---

## 已完成 ✅

### 1. 模型迁移 (2026-06-29)

- Qwen2-VL-72B-Instruct → Qwen3.5-35B-A3B
- 4×A100-80GB tensor parallel, vLLM 0.23.0
- 离线推理验证：1024 tokens / 7.88s = 130 tok/s
- API 服务验证：vLLM OpenAI-compatible server 正常
- 旧模型已删除，释放 137GB

### 2. 数据生成脚本 (2026-06-29)

已创建 5 个数据生成脚本 (`scripts/data_gen/`)：

| 脚本 | 任务 | 测试状态 |
|------|------|----------|
| `gen_visual_grounding_qa.sh` | 模型读图→生成VG问题 (Q-jsonl) | ✅ 10张测试通过 |
| `gen_grounding_cot.sh` | 消费问题→生成思维链定位 (A-CoT-jsonl) | ✅ 10张测试通过 |
| `gen_caption.sh` | 生成场景详细描述 | ✅ 10张测试通过 |
| `gen_decomposition.sh` | 生成高层任务→低层动作分解 | ✅ 10张测试通过 |
| `gen_dialog.sh` | 生成4-10轮人机对话 | ✅ 10张测试通过 |

### 3. 数据集准备 (2026-06-29)

- DVGBench 完整数据集已解压至 `data/DVGBench/`
- 2242 张图片：ERA 759 张 + VisDrone 1483 张
- 标注文件：`dvg_test.jsonl` (872 条，含 bbox/obb/poly/question)

---

## 测试结果 (10张/任务)

| 任务 | 成功率 | 平均耗时/张 | 平均输出 tokens |
|------|--------|-------------|----------------|
| Visual Grounding QA | 10/10 | 6.6s | ~1024 |
| Grounding CoT | 10/10 | 9.4s | ~700 |
| Caption | 10/10 | 7.0s | ~1024 |
| Decomposition | 10/10 | 13.8s | ~2048 |
| Dialog | 10/10 | 11.6s | ~1600 |

---

## 待办 / 下一步

- [ ] 审核 10 张测试输出的质量，调整提示词
- [ ] 模型输出含 thinking 前缀（"The user wants..."），需决定是否后处理去除
- [ ] 正式批量运行全部 1483 张 VisDrone 图片
- [ ] 批量运行 759 张 ERA 图片
- [ ] 合成标注数据集（将 Q-jsonl + A-CoT-jsonl 合并为训练格式）
- [ ] 数据质量抽检与清洗

---

## 环境信息

```
机器: debug-48gpu-cluster-v1-7
GPU: 4 × NVIDIA A100 80GB PCIe
模型: Qwen/Qwen3.5-35B-A3B (MoE, 35B params, 3B active)
推理框架: vLLM 0.23.0, tensor_parallel_size=4
数据集: DVGBench (erenzhou/DVGBench on HuggingFace)
```

---

## Git 提交历史

```
22bb5e8 feat: add DVGBench full dataset and update data pipeline
7ff2a30 feat: add data generation pipeline for 4 annotation types
f125820 chore: remove deprecated Qwen2-VL files and clean up project
cc7e67c docs: update migration progress to completed status
92926a9 refactor: migrate from Qwen2-VL-72B to Qwen3.5-35B-A3B
4030067 feat: 添加多模态推理脚本及 CoT 数据生成支持
f9358fa Initial commit: Qwen2-VL-72B-Instruct deployment project
```
