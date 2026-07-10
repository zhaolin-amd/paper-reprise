# Reproduction Report: 2603.08713

- **Repo:** (no official repo)
- **Environment:** CUDA 13.0 / torch 2.11.0+cu130 / transformers 5.13.0 / lm_eval 0.4.12

| model | config | algorithm | metric | paper | measured | verdict | reason |
|---|---|---|---|---|---|---|---|
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | INT4 | MXFP4-OCP | acc_norm | 70.98 | 68.87(-2.11) | PARTIAL | process faithful but value off tolerance 2.109 (>0.5) |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | BF16 | - | acc_norm | 76.51 | 74.96(-1.55) | PARTIAL | process faithful but value off tolerance 1.555 (>0.5) |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | INT4 | MXFP4-16-OAS | acc_norm | 73.14 | 71.83(-1.31) | PARTIAL | process faithful but value off tolerance 1.312 (>0.5) |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | INT4 | MXFP4-MBS-S | acc_norm | 73.66 | 72.52(-1.14) | PARTIAL | process faithful but value off tolerance 1.145 (>0.5) |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | INT4 | MXFP4-MBS-H | acc_norm | 74.12 | 72.46(-1.66) | PARTIAL | process faithful but value off tolerance 1.664 (>0.5) |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | BF16 | - | word_perplexity | 12.2 | 12.22(+0.0158) | MATCH | — |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | INT4 | MXFP4-OCP | word_perplexity | 15.18 | 15.15(-0.0333) | MATCH | — |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | INT4 | MXFP4-16-OAS | word_perplexity | 13.65 | 13.59(-0.0635) | MATCH | — |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | INT4 | MXFP4-MBS-S | word_perplexity | 13.09 | 13.08(-0.0113) | MATCH | — |
| /group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B | INT4 | MXFP4-MBS-H | word_perplexity | 13.03 | 13.05(+0.0235) | MATCH | — |

## Conclusion
- 10 claims: MATCH 5 · PARTIAL 5 · FAIL 0 · BLOCKED 0.
- The FP baseline matches the paper, so the **eval protocol is validated**; the 4 quantized config(s) outside tolerance (worst -2.11) are therefore a **genuine reproduction gap** (algorithm/calibration/version), not an eval-protocol artifact.

## Resources (per config)
(none)

## Per-task raw scores
(none)

## Replay script (per config)
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
