# 复现报告: TurboQuant-2504.19874

- **论文:** [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (arXiv:2504.19874)
- **仓库:** (no official repo)

| model | config | algorithm | metric | paper | 实测 | 判定 | 原因 |
|---|---|---|---|---|---|---|---|
| Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M | INT1 | TurboQuant_mse | mse_distortion | 0.36 | 0.3633(+0.003309) | MATCH | — |
| Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M | INT2 | TurboQuant_mse | mse_distortion | 0.117 | 0.1174(+0.000423) | MATCH | — |
| Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M | INT3 | TurboQuant_mse | mse_distortion | 0.03 | 0.03452(+0.004519) | MATCH | — |
| Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M | INT4 | TurboQuant_mse | mse_distortion | 0.009 | 0.00949(+0.00049) | MATCH | — |
| Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M | INT1 | TurboQuant_prod | ip_distortion | 0.0010221 | 0.001033(+1.1e-05) | MATCH | — |
| Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M | INT2 | TurboQuant_prod | ip_distortion | 0.00036458 | 0.000373(+8.29e-06) | MATCH | — |
| Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M | INT3 | TurboQuant_prod | ip_distortion | 0.00011719 | 0.000121(+3.36e-06) | MATCH | — |
| Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M | INT4 | TurboQuant_prod | ip_distortion | 3.0599e-05 | 3.54e-05(+4.85e-06) | MATCH | — |
| Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M | INT1 | TurboQuant_mse | ip_ratio | 0.6366 | 0.637(+0.0004) | MATCH | — |
| Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M | INT2 | TurboQuant_prod | ip_ratio | 1 | 0.991(-0.008966) | MATCH | — |

## 结论
- 共 10 个 claim:MATCH 10 · PARTIAL 0 · FAIL 0 · BLOCKED 0。





## 复算脚本(每个 config)
**Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M · INT1 · TurboQuant_mse**
`runs/turboquant-2504.19874-20260626-053253/claims/mse-distortion-b1/stdout.log`
`runs/turboquant-2504.19874-20260626-053253/claims/mse-bias-b1/stdout.log`

```bash
bash impl/run_eval.sh mse-distortion-b1
```

```bash
bash impl/run_eval.sh mse-bias-b1
```

**Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M · INT2 · TurboQuant_mse**
`runs/turboquant-2504.19874-20260626-053253/claims/mse-distortion-b2/stdout.log`

```bash
bash impl/run_eval.sh mse-distortion-b2
```

**Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M · INT3 · TurboQuant_mse**
`runs/turboquant-2504.19874-20260626-053253/claims/mse-distortion-b3/stdout.log`

```bash
bash impl/run_eval.sh mse-distortion-b3
```

**Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M · INT4 · TurboQuant_mse**
`runs/turboquant-2504.19874-20260626-053253/claims/mse-distortion-b4/stdout.log`

```bash
bash impl/run_eval.sh mse-distortion-b4
```

**Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M · INT1 · TurboQuant_prod**
`runs/turboquant-2504.19874-20260626-053253/claims/prod-distortion-b1/stdout.log`

```bash
bash impl/run_eval.sh prod-distortion-b1
```

**Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M · INT2 · TurboQuant_prod**
`runs/turboquant-2504.19874-20260626-053253/claims/prod-distortion-b2/stdout.log`
`runs/turboquant-2504.19874-20260626-053253/claims/prod-unbiased-b2/stdout.log`

```bash
bash impl/run_eval.sh prod-distortion-b2
```

```bash
bash impl/run_eval.sh prod-unbiased-b2
```

**Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M · INT3 · TurboQuant_prod**
`runs/turboquant-2504.19874-20260626-053253/claims/prod-distortion-b3/stdout.log`

```bash
bash impl/run_eval.sh prod-distortion-b3
```

**Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M · INT4 · TurboQuant_prod**
`runs/turboquant-2504.19874-20260626-053253/claims/prod-distortion-b4/stdout.log`

```bash
bash impl/run_eval.sh prod-distortion-b4
```
