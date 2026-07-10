<!-- en -->
**Root cause of acc_norm PARTIAL: inference engine mismatch**

| | Paper | This reproduction |
|---|---|---|
| Inference engine | vLLM | HuggingFace direct load |
| Passed to lm-eval | model path (string) | pre-instantiated model object |

When lm-eval receives an already-instantiated model it skips several initialization steps (log warning: `Many other model arguments may be ignored`), affecting log-likelihood computation. The BF16 baseline is itself off by 1.55 (74.96 vs 76.51), ruling out the quantization implementation — the gap is entirely in the eval infrastructure.

**PPL unaffected**: teacher-forcing needs no cross-option log-likelihood comparison and is insensitive to the inference engine → 5/5 MATCH (max deviation ±0.06).

**Fix**: `lm_eval --model vllm --model_args pretrained=<path>`

<!-- zh -->
**acc_norm PARTIAL 的根因：推理引擎差异**

| | 论文 | 本次复现 |
|---|---|---|
| 推理引擎 | vLLM | HuggingFace direct load |
| 传给 lm-eval | model path (string) | pre-instantiated model object |

lm-eval 接收已实例化 model 时跳过部分初始化（日志警告：`Many other model arguments may be ignored`），影响 log-likelihood 计算。BF16 基线本身就偏低 1.55（74.96 vs 76.51），排除了量化实现的责任——差距完全来自评测基础设施。

**PPL 不受影响**：teacher-forcing 无需跨选项对比 log-likelihood，对推理引擎不敏感 → 5/5 MATCH（最大偏差 ±0.06）。

**修复方向**：`lm_eval --model vllm --model_args pretrained=<path>`
