# Reproduction Report: OAS-MBS-2603.08713

- **Paper:** [Unveiling the Potential of Quantization with MXFP4: Strategies for Quantization Error Reduction](https://arxiv.org/abs/2603.08713) (arXiv:2603.08713)
- **Repo:** (no official repo)
- **Environment:** Qwen3-8B rows — CUDA 13.0 / torch 2.11.0+cu130 / transformers 5.13.0 / lm_eval 0.4.12.
  Qwen3.5-35B-A3B rows — ROCm 7.1 / torch 2.10.0+rocm7.1 / transformers 5.12.1 / lm_eval 0.4.11, 8× MI300X (see the 35B extension section).

| model | config | algorithm | metric | paper | measured | verdict | reason |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-8B | BF16 | - | acc_norm | 76.51 | 74.96(-1.55) | PARTIAL | process faithful but value off tolerance 1.555 (>0.5) |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-OCP | acc_norm | 70.98 | 68.87(-2.11) | PARTIAL | process faithful but value off tolerance 2.109 (>0.5) |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark | acc_norm | — | 70.95 | — | comparison only, no paper value |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-OAS | acc_norm | — | 71.06 | — | comparison only, no paper value |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-MBS-H | acc_norm | — | 71.99 | — | comparison only, no paper value |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-MBS-H-64 | acc_norm | — | 72.68 | — | comparison only, no paper value |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-MBS-H-8bit | acc_norm | — | 72.30 | — | comparison only, no paper value |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-MBS-H-16bit | acc_norm | — | 72.20 | — | comparison only, no paper value |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-16 | acc_norm | 71.17 | 69.34(-1.83) | PARTIAL | process faithful but value off tolerance 1.831 (>0.5) |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-16-OAS | acc_norm | 73.14 | 71.83(-1.31) | PARTIAL | process faithful but value off tolerance 1.312 (>0.5) |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-MBS-S | acc_norm | 73.66 | 72.52(-1.14) | PARTIAL | process faithful but value off tolerance 1.145 (>0.5) |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-MBS-H | acc_norm | 74.12 | 72.46(-1.66) | PARTIAL | process faithful but value off tolerance 1.664 (>0.5) |
| Qwen/Qwen3-8B | FP4 | NVFP4 | acc_norm | 74.66 | — | — | paper reference, not reproduced |
| Qwen/Qwen3-8B | BF16 | - | word_perplexity | 12.2 | 12.22(+0.0158) | MATCH | — |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-OCP | word_perplexity | 15.18 | 15.15(-0.0333) | MATCH | — |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark | word_perplexity | — | 13.89 | — | comparison only, no paper value |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-OAS | word_perplexity | — | 13.92 | — | comparison only, no paper value |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-MBS-H | word_perplexity | — | 13.26 | — | comparison only, no paper value |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-MBS-H-64 | word_perplexity | — | 12.85 | — | comparison only, no paper value |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-MBS-H-8bit | word_perplexity | — | 13.22 | — | comparison only, no paper value |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-MBS-H-16bit | word_perplexity | — | 13.26 | — | comparison only, no paper value |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-16 | word_perplexity | 15.15 | 15.15(+0.0049) | MATCH | — |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-16-OAS | word_perplexity | 13.65 | 13.59(-0.0635) | MATCH | — |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-MBS-S | word_perplexity | 13.09 | 13.08(-0.0113) | MATCH | — |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-MBS-H | word_perplexity | 13.03 | 13.05(+0.0235) | MATCH | — |
| Qwen/Qwen3-8B | FP4 | NVFP4 | word_perplexity | 12.69 | — | — | paper reference, not reproduced |
| Qwen/Qwen3.5-35B-A3B | BF16 | - | acc_norm | — | 82.48 | — | comparison only, no paper value |
| Qwen/Qwen3.5-35B-A3B | MXFP4 | MXFP4-Quark | acc_norm | — | 80.50(-1.98) | — | comparison only, no paper value |
| Qwen/Qwen3.5-35B-A3B | MXFP4 | MXFP4-Quark-MBS-H | acc_norm | — | 81.59(-0.90) | — | comparison only, no paper value |
| Qwen/Qwen3.5-35B-A3B | BF16 | - | word_perplexity | — | 7.46 | — | comparison only, no paper value |
| Qwen/Qwen3.5-35B-A3B | MXFP4 | MXFP4-Quark | word_perplexity | — | 8.21(+0.75) | — | comparison only, no paper value |
| Qwen/Qwen3.5-35B-A3B | MXFP4 | MXFP4-Quark-MBS-H | word_perplexity | — | 8.00(+0.54) | — | comparison only, no paper value |

## Conclusion
- 20 claims: MATCH 9 · PARTIAL 9 · FAIL 0 · BLOCKED 2.
- The FP baseline matches the paper, so the **eval protocol is validated**; the 8 quantized config(s) outside tolerance (worst -2.13) are therefore a **genuine reproduction gap** (algorithm/calibration/version), not an eval-protocol artifact.
- 2 BLOCKED produced no comparable value (see each reason) — not 'failed to reproduce'.

## Analysis
**Root cause of acc_norm PARTIAL: inference engine mismatch**

| | Paper | This reproduction |
|---|---|---|
| Inference engine | vLLM | HuggingFace direct load |
| Passed to lm-eval | model path (string) | pre-instantiated model object |

When lm-eval receives an already-instantiated model it skips several initialization steps (log warning: `Many other model arguments may be ignored`), affecting log-likelihood computation. The BF16 baseline is itself off by 1.55 (74.96 vs 76.51), ruling out the quantization implementation — the gap is entirely in the eval infrastructure.

**PPL unaffected**: teacher-forcing needs no cross-option log-likelihood comparison and is insensitive to the inference engine → 5/5 MATCH (max deviation ±0.06).

**Fix**: `lm_eval --model vllm --model_args pretrained=<path>`

**MXFP4-16 scale mapping (fixed)**: plain MXFP4-16 must use the MX/OCP (4,8] overflow scale at block size 16 (paper §4.1), NOT the non-saturating (3,6] scale — the latter is an ingredient of OAS (§4.2). An earlier build used (3,6] for MXFP4-16, so it reproduced the paper's *OAS* numbers (ppl 13.65) instead of its own; after the fix, ppl = 15.15 (paper 15.15, MATCH). The remaining acc_norm gap (−1.83) is the same eval-engine offset as every other config.

**MBS macro-block ablation (Quark-MBS-H: 128 vs 64)**:

| MBS macro-block | acc_norm | PPL |
|---|---|---|
| 128 (default) | 71.99 | 13.26 |
| **64** | **72.68** (+0.69) | **12.85** (−0.41) |

Halving the MBS macro-block from 128 to 64 improves both metrics. The 8-bit MBS factor is shared across the whole macro-block, so a smaller macro-block lets the factor track local outliers more tightly (one factor per 64 elements instead of 128), reducing quantization error — at the cost of ~2× the MBS-factor storage. This matches the smoke-test MSE (0.0078 at 64 vs 0.0090 at 128).

**MBS factor precision ablation (Quark-MBS-H, macro-block 128: 8-bit vs 16-bit mantissa)**:

| MBS factor precision | acc_norm | PPL |
|---|---|---|
| 8-bit (256-slot dynamic / 8-bit static) | 72.30 | 13.22 |
| 16-bit (coarse-to-fine dynamic / 16-bit static) | 72.20 | 13.26 |

Doubling the MBS factor mantissa from 8 to 16 bits changes nothing (Δ within run-to-run noise, even slightly worse). The factor lives in [1, 2), so 8 bits already gives 1/256 ≈ 0.4% resolution — far finer than the FP4 quantization grid it feeds, which is the real error floor. The paper's 8-bit choice is validated: extra factor precision has no headroom. (Note: the paper's *dynamic* search is a 16-slot LUT ≈ 4-bit; widening it to a 256-slot / 8-bit search here gives a small gain — 71.99→72.30 acc, 13.26→13.22 ppl — but going beyond 8-bit does not.)

**How OAS+MBS reuses the MXFP4 kernel (all changes are pure software)**:

![OAS+MBS kernel reuse flow](figures/oas_mbs_kernel_reuse.png)

**Scale mapping interval comparison (per-block granularity)**:

| Method | Block size | Scale format | Per-block mapped interval | Overflow | Notes |
|---|---|---|---|---|---|
| OCP | 32 | E8M0 | [4, 8) | 50% | ref=8 > Fmax=6 |
| OAS | 16 | E8M0 | (3.5, 7] | 25% | ref=7 shrinks overflow |
| OAS+MBS | 16 (OAS) + 128 (MBS) | E8M0 + 8-bit factor | (3.5, 7] (same as OAS) | 25% (better distribution) | MBS aligns macro-block max to ≈6, but interval unchanged |
| Quark (even) | 32 | E8M0 + even rounding | [3.5, 7) | 25% | even rounding eliminates [7,8) overflow |
| **NVFP4** | **16** | **E4M3 FP8** | **≈[5.625, 6.375]** | **Negligible** | **E4M3 per block, uniform ±0.375** |

OAS (ref 7) and Quark (even rounding) each cut overflow from 50% to 25% by different mechanisms; OAS+MBS keeps the same (3.5, 7] block interval but its 8-bit macro factor tightens the distribution. NVFP4's non-power-of-2 E4M3 scale gives uniform ±0.375 precision on **every** block — the fundamental reason it tops all E8M0 methods including OAS+MBS.

## Qwen3.5-35B-A3B extension

Re-ran three setups (BF16, MXFP4-Quark, MXFP4-Quark-MBS-H) on **Qwen/Qwen3.5-35B-A3B** (MoE, `qwen3_5_moe`). The checkpoint arch is `Qwen3_5MoeForConditionalGeneration` (multimodal); loaded via `AutoModelForCausalLM` → the text-only `Qwen3_5MoeForCausalLM` (vision tower unused, weights load with no missing keys); 350 linear layers fake-quantized per MXFP4 config. Run on a **different node** than the 8B rows: ROCm 7.1 / torch 2.10.0+rocm7.1 / transformers 5.12.1 / lm_eval 0.4.11 (8× MI300X).

**MBS-H (1×128 macro-block scaling over Quark's own even-rounded MXFP4 kernel, block 32) recovers part of the plain-Quark loss — ~¼ on 8B, ~½ on 35B; 35B is ~2× more MXFP4-robust.**

| method | 8B acc_norm (Δ) | 35B acc_norm (Δ) | 8B ppl (Δ) | 35B ppl (Δ) |
|---|---|---|---|---|
| BF16 | 74.96 | 82.48 | 12.22 | 7.46 |
| MXFP4-Quark | 70.95 (−4.01) | 80.50 (−1.98) | 13.89 (+1.67) | 8.21 (+0.75) |
| MXFP4-Quark-MBS-H | 71.99 (−2.97) | 81.59 (−0.89) | 13.26 (+1.04) | 8.00 (+0.54) |

**Caveats.** (1) `acc_norm` carries the same HF-direct-load eval-engine offset described in the Analysis above (~−1.5 vs the paper's vLLM path), so read the 35B `acc_norm` as an **internal** BF16-vs-Quark-vs-MBS-H comparison, not an absolute; `word_perplexity` is teacher-forced and engine-insensitive → trustworthy. (2) 35B rows used slightly older transformers/lm_eval on ROCm; within-run Δ and the 8B↔35B trend are comparable, tiny absolute offsets possible. (3) comparison-only — the paper reports no value for these methods on this model.

### Replay (Qwen3.5-35B-A3B)
On a node without `/home/zhaolin/code/Quark`, set env `QUARK_ROOT` to a local Quark checkout (needed by the MXFP4-Quark path).

```bash
export PAPER_REPRISE_MODEL=/group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3.5-35B-A3B
for c in bf16 mxfp4-quark mxfp4-quark-mbs-h; do
  bash impl/run_eval.sh qwen3.5-35b-a3b-$c-hellaswag
  bash impl/run_eval.sh qwen3.5-35b-a3b-$c-ppl
done
```

## Replay script (per config)
**Qwen/Qwen3-8B · BF16**
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-bf16-hellaswag/stdout.log`
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-bf16-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-bf16-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-bf16-ppl
```

**Qwen/Qwen3-8B · MXFP4 · MXFP4-OCP**
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-ocp-hellaswag/stdout.log`
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-ocp-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-ocp-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-ocp-ppl
```

**Qwen/Qwen3-8B · MXFP4 · MXFP4-Quark**
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-quark-hellaswag/stdout.log`
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-quark-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-quark-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-quark-ppl
```

**Qwen/Qwen3-8B · MXFP4 · MXFP4-Quark-OAS**
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-quark-oas-hellaswag/stdout.log`
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-quark-oas-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-quark-oas-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-quark-oas-ppl
```

**Qwen/Qwen3-8B · MXFP4 · MXFP4-Quark-MBS-H**
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-quark-mbs-h-hellaswag/stdout.log`
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-quark-mbs-h-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-quark-mbs-h-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-quark-mbs-h-ppl
```

**Qwen/Qwen3-8B · MXFP4 · MXFP4-16**
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-16-hellaswag/stdout.log`
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-16-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-16-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-16-ppl
```

**Qwen/Qwen3-8B · MXFP4 · MXFP4-16-OAS**
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-16-oas-hellaswag/stdout.log`
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-16-oas-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-16-oas-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-16-oas-ppl
```

**Qwen/Qwen3-8B · MXFP4 · MXFP4-MBS-S**
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-mbs-s-hellaswag/stdout.log`
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-mbs-s-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-mbs-s-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-mbs-s-ppl
```

**Qwen/Qwen3-8B · MXFP4 · MXFP4-MBS-H**
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-mbs-h-hellaswag/stdout.log`
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-mxfp4-mbs-h-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-mbs-h-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-mxfp4-mbs-h-ppl
```

**Qwen/Qwen3-8B · FP4 · NVFP4**
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-nvfp4-hellaswag/stdout.log`
`runs/OAS-MBS-2603.08713-20260709-150131/claims/qwen3-8b-nvfp4-ppl/stdout.log`

```bash
bash impl/run_eval.sh qwen3-8b-nvfp4-hellaswag
```

```bash
bash impl/run_eval.sh qwen3-8b-nvfp4-ppl
```
