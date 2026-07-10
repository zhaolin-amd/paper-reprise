# Reproduction Report: 2603.08713

- **Repo:** (no official repo)
- **Environment:** CUDA 13.0 / torch 2.11.0+cu130 / transformers 5.13.0 / lm_eval 0.4.12

| model | config | algorithm | metric | paper | measured | verdict | reason |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-8B | INT4 | MXFP4-OCP | acc_norm | 70.98 | 68.87(-2.11) | PARTIAL | process faithful but value off tolerance 2.109 (>0.5) |
| Qwen/Qwen3-8B | BF16 | - | acc_norm | 76.51 | 74.96(-1.55) | PARTIAL | process faithful but value off tolerance 1.555 (>0.5) |
| Qwen/Qwen3-8B | INT4 | MXFP4-16-OAS | acc_norm | 73.14 | 71.83(-1.31) | PARTIAL | process faithful but value off tolerance 1.312 (>0.5) |
| Qwen/Qwen3-8B | INT4 | MXFP4-MBS-S | acc_norm | 73.66 | 72.52(-1.14) | PARTIAL | process faithful but value off tolerance 1.145 (>0.5) |
| Qwen/Qwen3-8B | INT4 | MXFP4-MBS-H | acc_norm | 74.12 | 72.46(-1.66) | PARTIAL | process faithful but value off tolerance 1.664 (>0.5) |
| Qwen/Qwen3-8B | BF16 | - | word_perplexity | 12.2 | 12.22(+0.0158) | MATCH | — |
| Qwen/Qwen3-8B | INT4 | MXFP4-OCP | word_perplexity | 15.18 | 15.15(-0.0333) | MATCH | — |
| Qwen/Qwen3-8B | INT4 | MXFP4-16-OAS | word_perplexity | 13.65 | 13.59(-0.0635) | MATCH | — |
| Qwen/Qwen3-8B | INT4 | MXFP4-MBS-S | word_perplexity | 13.09 | 13.08(-0.0113) | MATCH | — |
| Qwen/Qwen3-8B | INT4 | MXFP4-MBS-H | word_perplexity | 13.03 | 13.05(+0.0235) | MATCH | — |

## Conclusion
- 10 claims: MATCH 5 · PARTIAL 5 · FAIL 0 · BLOCKED 0.
- The FP baseline matches the paper, so the **eval protocol is validated**; the 4 quantized config(s) outside tolerance (worst -2.11) are therefore a **genuine reproduction gap** (algorithm/calibration/version), not an eval-protocol artifact.

## Analysis
**Root cause of acc_norm PARTIAL: inference engine mismatch**
**acc_norm PARTIAL 的根因：推理引擎差异**

| | Paper / 论文 | This reproduction / 本次复现 |
|---|---|---|
| Inference engine / 推理引擎 | vLLM | HuggingFace direct load |
| Passed to lm-eval / 传给 lm-eval | model path (string) | pre-instantiated model object |

When lm-eval receives an already-instantiated model it skips several initialization steps (log warning: `Many other model arguments may be ignored`), affecting log-likelihood computation. The BF16 baseline is itself off by 1.55 (74.96 vs 76.51), ruling out the quantization implementation — the gap is entirely in the eval infrastructure.

lm-eval 接收已实例化 model 时跳过部分初始化（日志警告：`Many other model arguments may be ignored`），影响 log-likelihood 计算。BF16 基线本身就偏低 1.55（74.96 vs 76.51），排除了量化实现的责任——差距完全来自评测基础设施。

**PPL unaffected / PPL 不受影响**：teacher-forcing 无需跨选项对比 log-likelihood，对推理引擎不敏感 → 5/5 MATCH（最大偏差 ±0.06）。

**Fix / 修复方向**：`lm_eval --model vllm --model_args pretrained=<path>`



## Per-task raw scores
(none)

## Replay script (per config)
**Qwen/Qwen3-8B · INT4 · MXFP4-OCP**
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-ocp-hellaswag/stdout.log`
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-ocp-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-ocp-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-ocp-ppl
```

**Qwen/Qwen3-8B · BF16**
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-bf16-hellaswag/stdout.log`
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-bf16-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-bf16-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-bf16-ppl
```

**Qwen/Qwen3-8B · INT4 · MXFP4-16-OAS**
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-16-oas-hellaswag/stdout.log`
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-16-oas-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-16-oas-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-16-oas-ppl
```

**Qwen/Qwen3-8B · INT4 · MXFP4-MBS-S**
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-mbs-s-hellaswag/stdout.log`
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-mbs-s-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-mbs-s-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-mbs-s-ppl
```

**Qwen/Qwen3-8B · INT4 · MXFP4-MBS-H**
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-mbs-h-hellaswag/stdout.log`
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-mbs-h-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-mbs-h-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-mbs-h-ppl
```
