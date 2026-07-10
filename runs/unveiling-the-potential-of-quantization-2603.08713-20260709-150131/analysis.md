<!-- en -->
**Root cause of acc_norm PARTIAL: inference engine mismatch**

| | Paper | This reproduction |
|---|---|---|
| Inference engine | vLLM | HuggingFace direct load |
| Passed to lm-eval | model path (string) | pre-instantiated model object |

When lm-eval receives an already-instantiated model it skips several initialization steps (log warning: `Many other model arguments may be ignored`), affecting log-likelihood computation. The BF16 baseline is itself off by 1.55 (74.96 vs 76.51), ruling out the quantization implementation — the gap is entirely in the eval infrastructure.

**PPL unaffected**: teacher-forcing needs no cross-option log-likelihood comparison and is insensitive to the inference engine → 5/5 MATCH (max deviation ±0.06).

**Fix**: `lm_eval --model vllm --model_args pretrained=<path>`

**MXFP4-16 scale mapping (fixed)**: plain MXFP4-16 must use the MX/OCP (4,8] overflow scale at block size 16 (paper §4.1), NOT the non-saturating (3,6] scale — the latter is an ingredient of OAS (§4.2). An earlier build used (3,6] for MXFP4-16, so it reproduced the paper's *OAS* numbers (ppl 13.65) instead of its own; after the fix, ppl = 15.15 (paper 15.15, MATCH). The remaining acc_norm gap (−1.83) is the same eval-engine offset as every other config.

<!-- zh -->
**acc_norm PARTIAL 的根因：推理引擎差异**

| | 论文 | 本次复现 |
|---|---|---|
| 推理引擎 | vLLM | HuggingFace direct load |
| 传给 lm-eval | model path (string) | pre-instantiated model object |

lm-eval 接收已实例化 model 时跳过部分初始化（日志警告：`Many other model arguments may be ignored`），影响 log-likelihood 计算。BF16 基线本身就偏低 1.55（74.96 vs 76.51），排除了量化实现的责任——差距完全来自评测基础设施。

**PPL 不受影响**：teacher-forcing 无需跨选项对比 log-likelihood，对推理引擎不敏感 → 5/5 MATCH（最大偏差 ±0.06）。

**修复方向**：`lm_eval --model vllm --model_args pretrained=<path>`

**MXFP4-16 标度映射（已修复）**：plain MXFP4-16 应使用 block size 16 上的 MX/OCP (4,8] 溢出标度（论文 §4.1），而非非饱和的 (3,6] 标度——后者是 OAS 的组件（§4.2）。此前实现误用 (3,6]，使 MXFP4-16 复现出论文的 *OAS* 数值（ppl 13.65）而非自身数值；修复后 ppl = 15.15（论文 15.15，MATCH）。剩余的 acc_norm 差距（−1.83）与其它 config 一样源自评测引擎偏移。
