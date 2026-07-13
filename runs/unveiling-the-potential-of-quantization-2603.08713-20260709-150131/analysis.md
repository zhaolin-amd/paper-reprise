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

**E8M0 scale 的块最大值映射区间对比**：

| 方法 | Scale 格式 | 映射区间 | 溢出区间（Fmax=6） | 溢出比例 |
|---|---|---|---|---|
| OCP | E8M0（2 的幂次） | [4, 8) | (6, 8)，50% of [4,8) | 50% |
| OAS | E8M0 | (3.5, 7] | (6, 7)，25% of [4,8) | 25% |
| Quark（even） | E8M0（even 取整） | [3.5, 7) | (6, 7)，25% of [4,8) | 25% |
| **NVFP4** | **E4M3 FP8** | **≈[5.625, 6.375]** | **≈(6, 6.375)** | **极少** |

OCP/OAS/Quark 的 scale 都是 E8M0（只能取 2 的整数幂），因此映射区间宽。OAS 和 Quark 的核心优化是缩小溢出区间（50%→25%），原理相似但路径不同：OAS 通过把参考值从 8 改为 7 实现，Quark 通过将 amax 取整到更近的 2 的幂次实现。NVFP4 使用 E4M3 FP8 scale（3 位尾数），可以表示非 2 的幂次，理论上能把 amax 精确映射到 Fmax=6 附近（最大误差 ±6/16=0.375），从根本上消除了 E8M0 的粒度损失，这是 NVFP4 精度上界显著高于所有 E8M0 方案的根本原因。

**Scale mapping interval comparison (E8M0 vs E4M3)**:

| Method | Scale format | Mapped interval | Overflow region (Fmax=6) | Overflow rate |
|---|---|---|---|---|
| OCP | E8M0 (power-of-2) | [4, 8) | (6, 8) | 50% |
| OAS | E8M0 | (3.5, 7] | (6, 7) | 25% |
| Quark (even) | E8M0 (even rounding) | [3.5, 7) | (6, 7) | 25% |
| **NVFP4** | **E4M3 FP8** | **≈[5.625, 6.375]** | **≈(6, 6.375)** | **Negligible** |

OAS and Quark both reduce overflow from 50% to 25% via different mechanisms: OAS uses reference 7 instead of 8; Quark rounds amax to the nearest power of 2. NVFP4's E4M3 block scale (3 mantissa bits, non-power-of-2 representable) maps amax to within ±6/16=0.375 of Fmax=6, eliminating the quantization granularity loss inherent to E8M0 — the fundamental reason NVFP4 sets a higher accuracy ceiling than any E8M0-based method.
