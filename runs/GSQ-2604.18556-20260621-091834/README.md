# Reproduction Report: GSQ-2604.18556

- **Paper:** [GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling](https://arxiv.org/abs/2604.18556) (arXiv:2604.18556)
- **Repo:** https://github.com/IST-DASLab/GSQ@194281e25c93c6eb916784db049c536c6996451f
- **Environment:** CUDA 13.0 / torch 2.11.0+cu130 / transformers 5.8.1

| model | config | algorithm | metric | paper | measured | verdict | reason |
|---|---|---|---|---|---|---|---|
| meta-llama/Llama-3.1-8B-Instruct | BF16 | - | acc_norm_avg | 73.71 | 73.79(+0.08) | MATCH | — |
| meta-llama/Llama-3.1-8B-Instruct | INT2 G128 | GSQ | acc_norm_avg | 68.55 | 66.51(-2.04) | PARTIAL | process faithful but value off tolerance 2.04 (>0.5) |

## Conclusion
- 2 claims: MATCH 1 · PARTIAL 1 · FAIL 0 · BLOCKED 0.
- The FP baseline matches the paper, so the **eval protocol is validated**; the 1 quantized config(s) outside tolerance (worst -2.04) are therefore a **genuine reproduction gap** (algorithm/calibration/version), not an eval-protocol artifact.

## Algorithm overview

![GSQ algorithm overview](figures/gsq_overview.png)

GSQ is a post-training scalar-quantization method that makes the **discrete** grid-level assignment differentiable. For each coordinate it learns assignment logits `ℓ` (and per-group scales `s`), and instead of hard-selecting a grid level it takes a **Gumbel-Softmax** soft weighted sum over the K candidate levels: `p_i = softmax((κℓ_i + g_i)/τ)`, `w̃ = s·Σ p_i d_i`. These parameters are optimized block-wise (Lion optimizer, calibration data) to minimize the output reconstruction error `‖f(X;w) − f(X;w̃)‖²_F`. Annealing the temperature `τ → 0` collapses the soft assignment onto a single hard grid level, giving `ŵ = s·q_hard` — a fully discrete, deploy-ready layer (GGUF K-Quant compatible). Keeping the relaxation cardinality small (K = 3–8 for ternary / low bpp) is what makes the discrete optimization tractable.

## Per-task raw scores
**llama3-8b-baseline**

|    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge|      1|none  |     0|acc     |↑  |0.5179|±  |0.0146|
|             |       |none  |     0|acc_norm|↑  |0.5512|±  |0.0145|
|arc_easy     |      1|none  |     0|acc     |↑  |0.8178|±  |0.0079|
|             |       |none  |     0|acc_norm|↑  |0.7980|±  |0.0082|
|hellaswag    |      1|none  |     0|acc     |↑  |0.5910|±  |0.0049|
|             |       |none  |     0|acc_norm|↑  |0.7924|±  |0.0040|
|piqa         |      1|none  |     0|acc     |↑  |0.8003|±  |0.0093|
|             |       |none  |     0|acc_norm|↑  |0.8085|±  |0.0092|
|winogrande   |      1|none  |     0|acc     |↑  |0.7395|±  |0.0123|

**llama3-8b-2bit-avg-acc**

|    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge|      1|none  |     0|acc     |↑  |0.4241|±  |0.0144|
|             |       |none  |     0|acc_norm|↑  |0.4590|±  |0.0146|
|arc_easy     |      1|none  |     0|acc     |↑  |0.7593|±  |0.0088|
|             |       |none  |     0|acc_norm|↑  |0.7285|±  |0.0091|
|hellaswag    |      1|none  |     0|acc     |↑  |0.5077|±  |0.0050|
|             |       |none  |     0|acc_norm|↑  |0.6884|±  |0.0046|
|piqa         |      1|none  |     0|acc     |↑  |0.7584|±  |0.0100|
|             |       |none  |     0|acc_norm|↑  |0.7573|±  |0.0100|
|winogrande   |      1|none  |     0|acc     |↑  |0.6922|±  |0.0130|


## Replay script (per config)
**meta-llama/Llama-3.1-8B-Instruct · BF16**
`runs/GSQ-2604.18556-20260621-091834/claims/llama3-8b-baseline/stdout.log`

```bash
export SLURM_JOB_ID=local
VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 EVAL=1 KEEP_SERVING=0 \
  MODEL_PATH=$PAPER_REPRISE_MODEL \
  EVAL_TASKS=${PAPER_REPRISE_TASKS:-arc_challenge,arc_easy,hellaswag,winogrande,piqa} \
  TP_SIZE=${PAPER_REPRISE_GPUS:-8} \
  bash scripts/serve_model.sh
```

**meta-llama/Llama-3.1-8B-Instruct · INT2 G128 · GSQ**
`runs/GSQ-2604.18556-20260621-091834/claims/llama3-8b-2bit-avg-acc/stdout.log`

```bash
bash scripts/serve_model.sh && python eval_model.py --config configs/local/config.yaml
  --base-url http://localhost:8000/v1/completions
  --tasks arc_challenge,arc_easy,hellaswag,winogrande,piqa
```
