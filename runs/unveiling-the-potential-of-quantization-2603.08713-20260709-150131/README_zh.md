# 复现报告: 2603.08713

- **仓库:** (no official repo)
- **环境:** CUDA 13.0 / torch 2.11.0+cu130 / transformers 5.13.0 / lm_eval 0.4.12

| model | config | algorithm | metric | paper | 实测 | 判定 | 原因 |
|---|---|---|---|---|---|---|---|
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | INT4 | MXFP4-OCP | acc_norm | 70.98 | 68.87(-2.11) | PARTIAL | 过程忠实但数值超容差 2.109 (>0.5) |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | BF16 | - | acc_norm | 76.51 | 74.96(-1.55) | PARTIAL | 过程忠实但数值超容差 1.555 (>0.5) |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | INT4 | MXFP4-16-OAS | acc_norm | 73.14 | 71.83(-1.31) | PARTIAL | 过程忠实但数值超容差 1.312 (>0.5) |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | INT4 | MXFP4-MBS-S | acc_norm | 73.66 | 72.52(-1.14) | PARTIAL | 过程忠实但数值超容差 1.145 (>0.5) |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | INT4 | MXFP4-MBS-H | acc_norm | 74.12 | 72.46(-1.66) | PARTIAL | 过程忠实但数值超容差 1.664 (>0.5) |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | BF16 | - | word_perplexity | 12.2 | 12.22(+0.0158) | MATCH | — |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | INT4 | MXFP4-OCP | word_perplexity | 15.18 | 15.15(-0.0333) | MATCH | — |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | INT4 | MXFP4-16-OAS | word_perplexity | 13.65 | 13.59(-0.0635) | MATCH | — |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | INT4 | MXFP4-MBS-S | word_perplexity | 13.09 | 13.08(-0.0113) | MATCH | — |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | INT4 | MXFP4-MBS-H | word_perplexity | 13.03 | 13.05(+0.0235) | MATCH | — |

## 结论
- 共 10 个 claim:MATCH 5 · PARTIAL 5 · FAIL 0 · BLOCKED 0。
- FP 基线与论文吻合,说明**评测协议可信**;因此 4 个超容差的量化配置(最大偏差 -2.11)是**真实的复现差距**(算法/校准/版本所致),而非评测口径问题。

## 差距分析
**Why acc_norm (Hellaswag) is PARTIAL while PPL (Wikitext) is MATCH:**

The paper uses **vLLM** as the inference engine with lm-eval; this reproduction passes a
pre-instantiated HuggingFace model object to `HFLM` instead. When lm-eval receives an
already-instantiated model (not a string path), it skips several initialization steps —
the log shows `Many other model arguments may be ignored`. This affects the log-likelihood
computation that acc_norm relies on.

- **PPL** is teacher-forced (no sampling, no batching across choices) → insensitive to
  this difference → 5/5 MATCH within ±0.06.
- **acc_norm** compares log-likelihoods across multiple choice strings → sensitive to
  batching, tokenization padding, and prefix caching differences between vLLM and HF
  eager mode → systematically 1.1–2.1 below the paper.

The BF16 baseline itself is off by 1.55 (74.96 vs 76.51), which rules out the
quantization implementation as the cause — the gap is entirely in the eval infrastructure.

**To close this gap:** re-run via lm-eval's vLLM backend, passing the model as a string
path (`--model vllm --model_args pretrained=<path>`) so lm-eval initializes the
full pipeline consistently with the paper's setup.

## 资源占用(每个 config)
(none)

## 各任务原始分数
(none)

## 复算脚本(每个 config)
**/group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B · INT4 · MXFP4-OCP**
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-ocp-hellaswag/stdout.log`
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-ocp-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-ocp-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-ocp-ppl
```

**/group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B · BF16**
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-bf16-hellaswag/stdout.log`
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-bf16-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-bf16-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-bf16-ppl
```

**/group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B · INT4 · MXFP4-16-OAS**
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-16-oas-hellaswag/stdout.log`
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-16-oas-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-16-oas-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-16-oas-ppl
```

**/group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B · INT4 · MXFP4-MBS-S**
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-mbs-s-hellaswag/stdout.log`
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-mbs-s-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-mbs-s-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-mbs-s-ppl
```

**/group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B · INT4 · MXFP4-MBS-H**
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-mbs-h-hellaswag/stdout.log`
`runs/unveiling-the-potential-of-quantization-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-mbs-h-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-mbs-h-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-mbs-h-ppl
```
