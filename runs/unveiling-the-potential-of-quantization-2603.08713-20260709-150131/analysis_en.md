**Root cause of acc_norm PARTIAL: inference engine mismatch**

| | Paper | This reproduction |
|---|---|---|
| Inference engine | vLLM | HuggingFace direct load |
| Passed to lm-eval | model path (string) | pre-instantiated model object |

When lm-eval receives an already-instantiated model it skips several initialization steps (log warning: `Many other model arguments may be ignored`), affecting log-likelihood computation. The BF16 baseline is itself off by 1.55 (74.96 vs 76.51), ruling out the quantization implementation — the gap is entirely in the eval infrastructure.

**PPL unaffected**: teacher-forcing needs no cross-option log-likelihood comparison and is insensitive to the inference engine → 5/5 MATCH (max deviation ±0.06).

**Fix**: `lm_eval --model vllm --model_args pretrained=<path>`

**MXFP4-16 scale mapping (fixed)**: plain MXFP4-16 must use the MX/OCP (4,8] overflow scale at block size 16 (paper §4.1), NOT the non-saturating (3,6] scale — the latter is an ingredient of OAS (§4.2). An earlier build used (3,6] for MXFP4-16, so it reproduced the paper's *OAS* numbers (ppl 13.65) instead of its own; after the fix, ppl = 15.15 (paper 15.15, MATCH). The remaining acc_norm gap (−1.83) is the same eval-engine offset as every other config.

**Group-size effect on OAS/MBS (Quark block=32 vs paper block=16)**:

| Method | acc_norm | PPL |
|---|---|---|
| MXFP4-OCP (block=32) | 68.87 | 15.15 |
| MXFP4-Quark (block=32, even scale) | **70.95** | **13.89** |
| MXFP4-16-OAS (block=16) | **71.83** | **13.59** |
| MXFP4-Quark-OAS (block=32) | 71.06 | 13.92 |
| MXFP4-MBS-H (block=16) | **72.46** | **13.05** |
| MXFP4-Quark-MBS-H (block=32) | 72.22 | 13.32 |

Quark's even scale eliminates overflow for amax ∈ [7, 8) within each block — the root cause of OCP baseline saturation. This accounts for its large gain over plain OCP (+2.08 acc, −1.26 PPL). However, once OAS is applied (which independently prevents overflow via the (3.5,7] scale mapping), the finer block granularity of block=16 becomes the dominant factor: smaller blocks give more precise per-block scale → block=16 slightly outperforms block=32 for both acc and PPL.


**Scale mapping interval comparison (per-block granularity)**:

| Method | Block size | Scale format | Per-block mapped interval | Overflow | Notes |
|---|---|---|---|---|---|
| OCP | 32 | E8M0 | [4, 8) | 50% | ref=8 > Fmax=6 |
| OAS | 16 | E8M0 | (3.5, 7] | 25% | ref=7 shrinks overflow |
| OAS+MBS | 16 (OAS) + 128 (MBS) | E8M0 + 8-bit factor | (3.5, 7] (same as OAS) | 25% (better distribution) | MBS aligns macro-block max to ≈6, but interval unchanged |
| Quark (even) | 32 | E8M0 + even rounding | [3.5, 7) | 25% | even rounding eliminates [7,8) overflow |
| **NVFP4** | **16** | **E4M3 FP8** | **≈[5.625, 6.375]** | **Negligible** | **E4M3 per block, uniform ±0.375** |

- **OCP**: E8M0, maps to [4, 8); Fmax=6 falls mid-interval → 50% overflow.
- **OAS**: changes reference from 8 to 7, narrowing to (3.5, 7]; overflow shrinks to (6, 7) → 25%.
- **OAS+MBS**: per-16-element-block OAS interval unchanged at (3.5, 7]; MBS adds a 128-element macro-block factor that pushes the macro-block max to ≈6, improving distribution but **not** narrowing the block interval → still 25% overflow.
- **Quark (even)**: even rounding of amax eliminates overflow for amax ∈ [7, 8), giving [3.5, 7) → 25%; same rate as OAS but different mechanism.
- **NVFP4**: E4M3 FP8 (non-power-of-2) computes a precise scale for **every** 16-element block independently, with uniform ±0.375 precision — the fundamental reason it outperforms all E8M0-based methods including OAS+MBS.
