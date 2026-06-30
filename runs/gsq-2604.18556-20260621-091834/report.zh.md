# 复现报告: 2604.18556

- **仓库:** https://github.com/IST-DASLab/GSQ@194281e25c93c6eb916784db049c536c6996451f
- **环境:** torch 2.11.0+cu130 / transformers 5.8.1 / CUDA 13.0

| model | config | algorithm | metric | paper | 实测 | 判定 | 原因 |
|---|---|---|---|---|---|---|---|
| meta-llama/Llama-3.1-8B-Instruct | BF16 | - | acc_norm_avg | 73.71 | 73.79(+0.08) | MATCH | — |
| meta-llama/Llama-3.1-8B-Instruct | INT2 G128 | GSQ | acc_norm_avg | 68.55 | 66.51(-2.04) | PARTIAL | 过程忠实但数值超容差 2.04 (>0.5) |

## 结论
- 共 2 个 claim:MATCH 1 · PARTIAL 1 · FAIL 0 · BLOCKED 0。

## 资源占用(每个 config)
(none)

## 各任务原始分数
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

## 复算脚本(每个 config)
**meta-llama/Llama-3.1-8B-Instruct · BF16**
`runs/gsq-2604.18556-20260621-091834/claims/llama3-8b-baseline/stdout.log`

```bash
export SLURM_JOB_ID=local
VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 EVAL=1 KEEP_SERVING=0 \
  MODEL_PATH=$PAPER_REPRISE_MODEL \
  EVAL_TASKS=${PAPER_REPRISE_TASKS:-arc_challenge,arc_easy,hellaswag,winogrande,piqa} \
  TP_SIZE=${PAPER_REPRISE_GPUS:-8} \
  bash scripts/serve_model.sh
```

**meta-llama/Llama-3.1-8B-Instruct · INT2 G128 · GSQ**
`runs/gsq-2604.18556-20260621-091834/claims/llama3-8b-2bit-avg-acc/stdout.log`

```bash
bash scripts/serve_model.sh && python eval_model.py --config configs/local/config.yaml
  --base-url http://localhost:8000/v1/completions
  --tasks arc_challenge,arc_easy,hellaswag,winogrande,piqa
```
