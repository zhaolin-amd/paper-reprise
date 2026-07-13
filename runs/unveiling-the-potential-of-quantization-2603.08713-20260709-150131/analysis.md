<!-- en -->
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

<!-- zh -->
**acc_norm PARTIAL 的根因：推理引擎差异**

| | 论文 | 本次复现 |
|---|---|---|
| 推理引擎 | vLLM | HuggingFace direct load |
| 传给 lm-eval | model path (string) | pre-instantiated model object |

lm-eval 接收已实例化 model 时跳过部分初始化（日志警告：`Many other model arguments may be ignored`），影响 log-likelihood 计算。BF16 基线本身就偏低 1.55（74.96 vs 76.51），排除了量化实现的责任——差距完全来自评测基础设施。

**PPL 不受影响**：teacher-forcing 无需跨选项对比 log-likelihood，对推理引擎不敏感 → 5/5 MATCH（最大偏差 ±0.06）。

**修复方向**：`lm_eval --model vllm --model_args pretrained=<path>`

**MXFP4-16 标度映射（已修复）**：plain MXFP4-16 应使用 block size 16 上的 MX/OCP (4,8] 溢出标度（论文 §4.1），而非非饱和的 (3,6] 标度——后者是 OAS 的组件（§4.2）。此前实现误用 (3,6]，使 MXFP4-16 复现出论文的 *OAS* 数值（ppl 13.65）而非自身数值；修复后 ppl = 15.15（论文 15.15，MATCH）。剩余的 acc_norm 差距（−1.83）与其它 config 一样源自评测引擎偏移。

**Group-size 对 OAS/MBS 的影响（Quark block=32 vs 论文 block=16）**：

| 方法 | acc_norm | PPL |
|---|---|---|
| MXFP4-OCP（block=32） | 68.87 | 15.15 |
| MXFP4-Quark（block=32，even scale） | **70.95** | **13.89** |
| MXFP4-16-OAS（block=16） | **71.83** | **13.59** |
| MXFP4-Quark-OAS（block=32） | 71.06 | 13.92 |
| MXFP4-MBS-H（block=16） | **72.46** | **13.05** |
| MXFP4-Quark-MBS-H（block=32） | 72.22 | 13.32 |

Quark 的 even scale 消除了每个 block 内 amax ∈ [7, 8) 的溢出截断，这正是 OCP baseline 精度损失的根源，因此相比纯 OCP 有显著提升（acc +2.08，PPL −1.26）。然而一旦叠加 OAS（OAS 本身已通过 (3.5,7] 的 scale 映射独立消除了溢出），更细粒度的 block=16 成为主导因素：block 越小，每个 block 的 scale 越精准 → block=16 在 acc 和 PPL 两个指标上都略优于 block=32。

**各方法每块映射区间对比（以 16/32 元素块为粒度）**：

| 方法 | 量化粒度 | Scale 格式 | 每块映射区间 | 溢出比例 | 说明 |
|---|---|---|---|---|---|
| OCP | 32 元素 | E8M0 | [4, 8) | 50% | 参考值 8 > Fmax=6，宽区间 |
| OAS | 16 元素 | E8M0 | (3.5, 7] | 25% | 参考值改为 7，缩小溢出区间 |
| OAS+MBS | 16 元素（OAS）+ 128 元素（MBS） | E8M0 + 8 位 factor | (3.5, 7]（同 OAS） | 25%（但分布更优） | OAS 区间不变；MBS 额外让宏块 max ≈6，改善分布而非缩窄区间 |
| Quark（even） | 32 元素 | E8M0（even 取整） | [3.5, 7) | 25% | even 取整消除 amax ∈ [7,8) 溢出 |
| **NVFP4** | **16 元素** | **E4M3 FP8** | **≈[5.625, 6.375]** | **极少** | **每块独立 E4M3，均匀精度 ±0.375** |

OCP/OAS/Quark/OAS+MBS 的块级 scale 均为 E8M0，只能取 2 的幂次，映射区间较宽。OAS 和 Quark 各自用不同路径把溢出比例从 50% 降到 25%：OAS 把参考值从 8 改为 7，Quark 对 amax 做 even 取整。OAS+MBS 在此基础上用 8 位 factor 把宏块最大值精确推向 Fmax，改善了区间内的分布，但 16 元素块的映射区间本身仍是 (3.5, 7]，与纯 OAS 相同。NVFP4 的 E4M3 scale 对**每个** 16 元素块独立计算，精度 ±0.375 均匀覆盖所有块，这是它精度上界高于所有 E8M0 方案（包括 OAS+MBS）的根本原因。

**Scale mapping interval comparison (per-block granularity)**:

| Method | Block size | Scale format | Per-block mapped interval | Overflow | Notes |
|---|---|---|---|---|---|
| OCP | 32 | E8M0 | [4, 8) | 50% | ref=8 > Fmax=6 |
| OAS | 16 | E8M0 | (3.5, 7] | 25% | ref=7 shrinks overflow |
| OAS+MBS | 16 (OAS) + 128 (MBS) | E8M0 + 8-bit factor | (3.5, 7] (same as OAS) | 25% (better distribution) | MBS aligns macro-block max to ≈6, but interval unchanged |
| Quark (even) | 32 | E8M0 + even rounding | [3.5, 7) | 25% | even rounding eliminates [7,8) overflow |
| **NVFP4** | **16** | **E4M3 FP8** | **≈[5.625, 6.375]** | **Negligible** | **E4M3 per block, uniform ±0.375** |

OAS and Quark each reduce overflow from 50% to 25% via different paths. OAS+MBS improves the distribution within (3.5, 7] (macro-block max maps to ≈6) but does not narrow the per-block OAS interval. NVFP4's E4M3 scale provides uniform ±0.375 precision independently for every 16-element block — the fundamental reason it outperforms all E8M0-based methods including OAS+MBS.
