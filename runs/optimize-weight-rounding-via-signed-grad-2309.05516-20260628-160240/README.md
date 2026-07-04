# Reproduction Report: 2309.05516

- **Repo:** https://github.com/intel/auto-round@9468e52
- **Environment:** torch 2.12.1 / transformers 5.12.1 / lm_eval 0.4.12 / CUDA 13.0

| model | config | algorithm | metric | paper | measured | verdict | reason |
|---|---|---|---|---|---|---|---|
| mistralai/Mistral-7B-v0.1 | BF16 | - | avg_accuracy_11tasks | 63.3 | 64.52(+1.22) | PARTIAL | process faithful but value off tolerance 1.218 (>0.5) |
| mistralai/Mistral-7B-v0.1 | INT4 G128 | signround | avg_accuracy_11tasks | 62.62 | 63.91(+1.29) | PARTIAL | process faithful but value off tolerance 1.289 (>0.5) |
| mistralai/Mistral-7B-v0.1 | INT2 G128 | signround | avg_accuracy_11tasks | 52.71 | 53.86(+1.15) | PARTIAL | process faithful but value off tolerance 1.153 (>0.5) |
| meta-llama/Llama-2-7b-hf | BF16 | - | avg_accuracy_11tasks | 57.98 | 59.47(+1.49) | PARTIAL | process faithful but value off tolerance 1.485 (>0.5) |
| meta-llama/Llama-2-7b-hf | INT4 G128 | signround | avg_accuracy_11tasks | 57.57 | 59.08(+1.51) | PARTIAL | process faithful but value off tolerance 1.513 (>0.5) |
| meta-llama/Llama-2-7b-hf | INT2 G128 | signround | avg_accuracy_11tasks | 48.64 | 51.03(+2.39) | PARTIAL | process faithful but value off tolerance 2.395 (>0.5) |

## Conclusion
- 6 claims: MATCH 0 · PARTIAL 6 · FAIL 0 · BLOCKED 0.
- Measured is **consistently above** the paper (Δ +1.15…+2.39) — a systematic eval/setup offset (e.g. lm-eval/library version drift), not per-config noise.

## Resources (per config)
| model | config | time | peak VRAM |
|---|---|---|---|
| mistralai/Mistral-7B-v0.1 | BF16 | 6.7 min | — |
| mistralai/Mistral-7B-v0.1 | INT4 G128 | 7.5 min | 32.4 GB |
| mistralai/Mistral-7B-v0.1 | INT2 G128 | 7.3 min | 32.5 GB |
| meta-llama/Llama-2-7b-hf | BF16 | 7.0 min | — |
| meta-llama/Llama-2-7b-hf | INT4 G128 | 7.1 min | 32.8 GB |
| meta-llama/Llama-2-7b-hf | INT2 G128 | 7.2 min | 32.8 GB |

## Per-task raw scores
**mistral-7b-fp16-baseline**

|                 Tasks                 |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|----------|---|-----:|---|-----:|
|mmlu                                   |      2|none  |      |acc       |   |0.5975|±  |0.0039|
|arc_challenge                          |      1|none  |     0|acc       |↑  |0.5034|±  |0.0146|
|                                       |       |none  |     0|acc_norm  |↑  |0.5418|±  |0.0146|
|arc_easy                               |      1|none  |     0|acc       |↑  |0.8081|±  |0.0081|
|                                       |       |none  |     0|acc_norm  |↑  |0.7950|±  |0.0083|
|boolq                                  |      2|none  |     0|acc       |↑  |0.8364|±  |0.0065|
|hellaswag                              |      1|none  |     0|acc       |↑  |0.6135|±  |0.0049|
|                                       |       |none  |     0|acc_norm  |↑  |0.8109|±  |0.0039|
|lambada_openai                         |      1|none  |     0|acc       |↑  |0.7565|±  |0.0060|
|                                       |       |none  |     0|perplexity|↓  |3.1802|±  |0.0583|
|openbookqa                             |      1|none  |     0|acc       |↑  |0.3260|±  |0.0210|
|                                       |       |none  |     0|acc_norm  |↑  |0.4400|±  |0.0222|
|piqa                                   |      1|none  |     0|acc       |↑  |0.8085|±  |0.0092|
|                                       |       |none  |     0|acc_norm  |↑  |0.8221|±  |0.0089|
|rte                                    |      1|none  |     0|acc       |↑  |0.6751|±  |0.0282|
|truthfulqa_mc2                         |      3|none  |     0|acc       |↑  |0.4261|±  |0.0142|
|winogrande                             |      1|none  |     0|acc       |↑  |0.7459|±  |0.0122|

**mistral-7b-w4g128-claim**

|                 Tasks                 |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|----------|---|-----:|---|-----:|
|mmlu                                   |      2|none  |      |acc       |   |0.5902|±  |0.0039|
|arc_challenge                          |      1|none  |     0|acc       |↑  |0.4932|±  |0.0146|
|                                       |       |none  |     0|acc_norm  |↑  |0.5333|±  |0.0146|
|arc_easy                               |      1|none  |     0|acc       |↑  |0.7997|±  |0.0082|
|                                       |       |none  |     0|acc_norm  |↑  |0.7896|±  |0.0084|
|boolq                                  |      2|none  |     0|acc       |↑  |0.8321|±  |0.0065|
|hellaswag                              |      1|none  |     0|acc       |↑  |0.6071|±  |0.0049|
|                                       |       |none  |     0|acc_norm  |↑  |0.8036|±  |0.0040|
|lambada_openai                         |      1|none  |     0|acc       |↑  |0.7566|±  |0.0060|
|                                       |       |none  |     0|perplexity|↓  |3.1996|±  |0.0595|
|openbookqa                             |      1|none  |     0|acc       |↑  |0.3320|±  |0.0211|
|                                       |       |none  |     0|acc_norm  |↑  |0.4460|±  |0.0223|
|piqa                                   |      1|none  |     0|acc       |↑  |0.8036|±  |0.0093|
|                                       |       |none  |     0|acc_norm  |↑  |0.8150|±  |0.0091|
|rte                                    |      1|none  |     0|acc       |↑  |0.6462|±  |0.0288|
|truthfulqa_mc2                         |      3|none  |     0|acc       |↑  |0.4156|±  |0.0141|
|winogrande                             |      1|none  |     0|acc       |↑  |0.7537|±  |0.0121|

**mistral-7b-w2g128-claim**

|                 Tasks                 |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|----------|---|-----:|---|-----:|
|mmlu                                   |      2|none  |      |acc       |   |0.3716|±  |0.0040|
|arc_challenge                          |      1|none  |     0|acc       |↑  |0.3788|±  |0.0142|
|                                       |       |none  |     0|acc_norm  |↑  |0.4053|±  |0.0143|
|arc_easy                               |      1|none  |     0|acc       |↑  |0.7142|±  |0.0093|
|                                       |       |none  |     0|acc_norm  |↑  |0.6730|±  |0.0096|
|boolq                                  |      2|none  |     0|acc       |↑  |0.7627|±  |0.0074|
|hellaswag                              |      1|none  |     0|acc       |↑  |0.5183|±  |0.0050|
|                                       |       |none  |     0|acc_norm  |↑  |0.6840|±  |0.0046|
|lambada_openai                         |      1|none  |     0|acc       |↑  |0.5766|±  |0.0069|
|                                       |       |none  |     0|perplexity|↓  |6.6018|±  |0.1881|
|openbookqa                             |      1|none  |     0|acc       |↑  |0.2600|±  |0.0196|
|                                       |       |none  |     0|acc_norm  |↑  |0.3740|±  |0.0217|
|piqa                                   |      1|none  |     0|acc       |↑  |0.7486|±  |0.0101|
|                                       |       |none  |     0|acc_norm  |↑  |0.7677|±  |0.0099|
|rte                                    |      1|none  |     0|acc       |↑  |0.5560|±  |0.0299|
|truthfulqa_mc2                         |      3|none  |     0|acc       |↑  |0.3909|±  |0.0142|
|winogrande                             |      1|none  |     0|acc       |↑  |0.6472|±  |0.0134|

**llama2-7b-fp16-baseline**

|                 Tasks                 |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|----------|---|-----:|---|-----:|
|mmlu                                   |      2|none  |      |acc       |   |0.4180|±  |0.0041|
|arc_challenge                          |      1|none  |     0|acc       |↑  |0.4300|±  |0.0145|
|                                       |       |none  |     0|acc_norm  |↑  |0.4505|±  |0.0145|
|arc_easy                               |      1|none  |     0|acc       |↑  |0.7551|±  |0.0088|
|                                       |       |none  |     0|acc_norm  |↑  |0.7382|±  |0.0090|
|boolq                                  |      2|none  |     0|acc       |↑  |0.7930|±  |0.0071|
|hellaswag                              |      1|none  |     0|acc       |↑  |0.5711|±  |0.0049|
|                                       |       |none  |     0|acc_norm  |↑  |0.7619|±  |0.0043|
|lambada_openai                         |      1|none  |     0|acc       |↑  |0.7361|±  |0.0061|
|                                       |       |none  |     0|perplexity|↓  |3.4146|±  |0.0668|
|openbookqa                             |      1|none  |     0|acc       |↑  |0.3340|±  |0.0211|
|                                       |       |none  |     0|acc_norm  |↑  |0.4400|±  |0.0222|
|piqa                                   |      1|none  |     0|acc       |↑  |0.7813|±  |0.0096|
|                                       |       |none  |     0|acc_norm  |↑  |0.7873|±  |0.0095|
|rte                                    |      1|none  |     0|acc       |↑  |0.6390|±  |0.0289|
|truthfulqa_mc2                         |      3|none  |     0|acc       |↑  |0.3875|±  |0.0135|
|winogrande                             |      1|none  |     0|acc       |↑  |0.6961|±  |0.0129|

**llama2-7b-w4g128-claim**

|                 Tasks                 |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|----------|---|-----:|---|-----:|
|mmlu                                   |      2|none  |      |acc       |   |0.4200|±  |0.0041|
|arc_challenge                          |      1|none  |     0|acc       |↑  |0.4309|±  |0.0145|
|                                       |       |none  |     0|acc_norm  |↑  |0.4505|±  |0.0145|
|arc_easy                               |      1|none  |     0|acc       |↑  |0.7500|±  |0.0089|
|                                       |       |none  |     0|acc_norm  |↑  |0.7243|±  |0.0092|
|boolq                                  |      2|none  |     0|acc       |↑  |0.7878|±  |0.0072|
|hellaswag                              |      1|none  |     0|acc       |↑  |0.5642|±  |0.0049|
|                                       |       |none  |     0|acc_norm  |↑  |0.7523|±  |0.0043|
|lambada_openai                         |      1|none  |     0|acc       |↑  |0.7367|±  |0.0061|
|                                       |       |none  |     0|perplexity|↓  |3.4340|±  |0.0681|
|openbookqa                             |      1|none  |     0|acc       |↑  |0.3220|±  |0.0209|
|                                       |       |none  |     0|acc_norm  |↑  |0.4240|±  |0.0221|
|piqa                                   |      1|none  |     0|acc       |↑  |0.7835|±  |0.0096|
|                                       |       |none  |     0|acc_norm  |↑  |0.7873|±  |0.0095|
|rte                                    |      1|none  |     0|acc       |↑  |0.6137|±  |0.0293|
|truthfulqa_mc2                         |      3|none  |     0|acc       |↑  |0.3957|±  |0.0136|
|winogrande                             |      1|none  |     0|acc       |↑  |0.6946|±  |0.0129|

**llama2-7b-w2g128-claim**

|                 Tasks                 |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|----------|---|-----:|---|-----:|
|mmlu                                   |      2|none  |      |acc       |   |0.2525|±  |0.0037|
|arc_challenge                          |      1|none  |     0|acc       |↑  |0.3515|±  |0.0140|
|                                       |       |none  |     0|acc_norm  |↑  |0.3643|±  |0.0141|
|arc_easy                               |      1|none  |     0|acc       |↑  |0.6755|±  |0.0096|
|                                       |       |none  |     0|acc_norm  |↑  |0.6359|±  |0.0099|
|boolq                                  |      2|none  |     0|acc       |↑  |0.7162|±  |0.0079|
|hellaswag                              |      1|none  |     0|acc       |↑  |0.4794|±  |0.0050|
|                                       |       |none  |     0|acc_norm  |↑  |0.6293|±  |0.0048|
|lambada_openai                         |      1|none  |     0|acc       |↑  |0.5678|±  |0.0069|
|                                       |       |none  |     0|perplexity|↓  |6.4696|±  |0.1712|
|openbookqa                             |      1|none  |     0|acc       |↑  |0.2840|±  |0.0202|
|                                       |       |none  |     0|acc_norm  |↑  |0.3860|±  |0.0218|
|piqa                                   |      1|none  |     0|acc       |↑  |0.7361|±  |0.0103|
|                                       |       |none  |     0|acc_norm  |↑  |0.7410|±  |0.0102|
|rte                                    |      1|none  |     0|acc       |↑  |0.5848|±  |0.0297|
|truthfulqa_mc2                         |      3|none  |     0|acc       |↑  |0.3417|±  |0.0133|
|winogrande                             |      1|none  |     0|acc       |↑  |0.6243|±  |0.0136|

## Replay script (per config)
**mistralai/Mistral-7B-v0.1 · BF16**
`runs/optimize-weight-rounding-via-signed-grad-2309.05516-20260628-160240/claims/mistral-7b-fp16-baseline/stdout.log`

```bash
auto-round --model $PAPER_REPRISE_MODEL --eval --tasks ${PAPER_REPRISE_TASKS:-mmlu,lambada_openai,hellaswag,winogrande,piqa,truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge}
```

**mistralai/Mistral-7B-v0.1 · INT4 G128 · signround**
`runs/optimize-weight-rounding-via-signed-grad-2309.05516-20260628-160240/claims/mistral-7b-w4g128-claim/stdout.log`

```bash
auto-round --model $PAPER_REPRISE_MODEL --bits 4 --group_size 128 --iters 200 --nsamples 512 --seqlen 2048 --dataset NeelNanda/pile-10k --format auto_round --output_dir ./tmp_model --tasks ${PAPER_REPRISE_TASKS:-mmlu,lambada_openai,hellaswag,winogrande,piqa,truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge}
```

**mistralai/Mistral-7B-v0.1 · INT2 G128 · signround**
`runs/optimize-weight-rounding-via-signed-grad-2309.05516-20260628-160240/claims/mistral-7b-w2g128-claim/stdout.log`

```bash
auto-round --model $PAPER_REPRISE_MODEL --bits 2 --group_size 128 --iters 200 --nsamples 512 --seqlen 2048 --dataset NeelNanda/pile-10k --format auto_round --output_dir ./tmp_model --tasks ${PAPER_REPRISE_TASKS:-mmlu,lambada_openai,hellaswag,winogrande,piqa,truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge}
```

**meta-llama/Llama-2-7b-hf · BF16**
`runs/optimize-weight-rounding-via-signed-grad-2309.05516-20260628-160240/claims/llama2-7b-fp16-baseline/stdout.log`

```bash
auto-round --model $PAPER_REPRISE_MODEL --eval --tasks ${PAPER_REPRISE_TASKS:-mmlu,lambada_openai,hellaswag,winogrande,piqa,truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge}
```

**meta-llama/Llama-2-7b-hf · INT4 G128 · signround**
`runs/optimize-weight-rounding-via-signed-grad-2309.05516-20260628-160240/claims/llama2-7b-w4g128-claim/stdout.log`

```bash
auto-round --model $PAPER_REPRISE_MODEL --bits 4 --group_size 128 --iters 200 --nsamples 512 --seqlen 2048 --dataset NeelNanda/pile-10k --format auto_round --output_dir ./tmp_model --tasks ${PAPER_REPRISE_TASKS:-mmlu,lambada_openai,hellaswag,winogrande,piqa,truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge}
```

**meta-llama/Llama-2-7b-hf · INT2 G128 · signround**
`runs/optimize-weight-rounding-via-signed-grad-2309.05516-20260628-160240/claims/llama2-7b-w2g128-claim/stdout.log`

```bash
auto-round --model $PAPER_REPRISE_MODEL --bits 2 --group_size 128 --iters 200 --nsamples 512 --seqlen 2048 --dataset NeelNanda/pile-10k --format auto_round --output_dir ./tmp_model --tasks ${PAPER_REPRISE_TASKS:-mmlu,lambada_openai,hellaswag,winogrande,piqa,truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge}
```
