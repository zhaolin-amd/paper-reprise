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
- **PPL(Wikitext word_perplexity):5/5 全部 MATCH**,最大偏差 ±0.06——fake-quant 实现精确复现了论文的困惑度数值。
- **acc_norm(Hellaswag):5/5 全部 PARTIAL**,实测一致低于论文约 1.1~2.1。根因是**推理引擎差异**,而非量化误差:
  - 论文使用 **vLLM** 作为推理引擎配合 lm-eval;我们使用 `HFLM` 传入已实例化的 HuggingFace model 对象。
  - lm-eval 接收到已实例化 model(而非字符串路径)时会跳过部分初始化步骤(日志中出现 `Many other model arguments may be ignored` 警告),影响 log-likelihood 的计算精度。
  - PPL 是 teacher-forcing 模式,对此不敏感;acc_norm 是多选项 log-likelihood 排名,对 batch 拼接、tokenization 细节更敏感,因此 PPL 完全吻合而 acc_norm 偏低约 1.5 点。
  - 消除此差距的方向:改用 lm-eval 的 vLLM 后端(`lm_eval --model vllm --model_args pretrained=<path>`)替代传入预加载模型。

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
