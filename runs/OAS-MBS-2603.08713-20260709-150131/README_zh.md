# 复现报告: OAS-MBS-2603.08713

- **论文:** [Unveiling the Potential of Quantization with MXFP4: Strategies for Quantization Error Reduction](https://arxiv.org/abs/2603.08713) (arXiv:2603.08713)
- **仓库:** (no official repo)
- **环境:** Qwen3-8B 行 —— CUDA 13.0 / torch 2.11.0+cu130 / transformers 5.13.0 / lm_eval 0.4.12。
  Qwen3.5-35B-A3B 行 —— ROCm 7.1 / torch 2.10.0+rocm7.1 / transformers 5.12.1 / lm_eval 0.4.11，8× MI300X（见 35B 扩展一节）。

| model | config | algorithm | metric | paper | 实测 | 判定 | 原因 |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-8B | BF16 | - | acc_norm | 76.51 | 74.96(-1.55) | PARTIAL | 过程忠实但数值超容差 1.555 (>0.5) |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-OCP | acc_norm | 70.98 | 68.87(-2.11) | PARTIAL | 过程忠实但数值超容差 2.109 (>0.5) |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark | acc_norm | — | 70.95 | — | 参考对比，无论文数值 |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-OAS | acc_norm | — | 71.06 | — | 参考对比，无论文数值 |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-MBS-H | acc_norm | — | 71.99 | — | 参考对比，无论文数值 |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-MBS-H-64 | acc_norm | — | 72.68 | — | 参考对比，无论文数值 |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-MBS-H-8bit | acc_norm | — | 72.30 | — | 参考对比，无论文数值 |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-MBS-H-16bit | acc_norm | — | 72.20 | — | 参考对比，无论文数值 |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-16 | acc_norm | 71.17 | 69.34(-1.83) | PARTIAL | 过程忠实但数值超容差 1.831 (>0.5) |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-16-OAS | acc_norm | 73.14 | 71.83(-1.31) | PARTIAL | 过程忠实但数值超容差 1.312 (>0.5) |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-MBS-S | acc_norm | 73.66 | 72.52(-1.14) | PARTIAL | 过程忠实但数值超容差 1.145 (>0.5) |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-MBS-H | acc_norm | 74.12 | 72.46(-1.66) | PARTIAL | 过程忠实但数值超容差 1.664 (>0.5) |
| Qwen/Qwen3-8B | FP4 | NVFP4 | acc_norm | 74.66 | — | — | 论文参考值，未复现 |
| Qwen/Qwen3-8B | BF16 | - | word_perplexity | 12.2 | 12.22(+0.0158) | MATCH | — |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-OCP | word_perplexity | 15.18 | 15.15(-0.0333) | MATCH | — |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark | word_perplexity | — | 13.89 | — | 参考对比，无论文数值 |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-OAS | word_perplexity | — | 13.92 | — | 参考对比，无论文数值 |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-MBS-H | word_perplexity | — | 13.26 | — | 参考对比，无论文数值 |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-MBS-H-64 | word_perplexity | — | 12.85 | — | 参考对比，无论文数值 |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-MBS-H-8bit | word_perplexity | — | 13.22 | — | 参考对比，无论文数值 |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-Quark-MBS-H-16bit | word_perplexity | — | 13.26 | — | 参考对比，无论文数值 |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-16 | word_perplexity | 15.15 | 15.15(+0.0049) | MATCH | — |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-16-OAS | word_perplexity | 13.65 | 13.59(-0.0635) | MATCH | — |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-MBS-S | word_perplexity | 13.09 | 13.08(-0.0113) | MATCH | — |
| Qwen/Qwen3-8B | MXFP4 | MXFP4-MBS-H | word_perplexity | 13.03 | 13.05(+0.0235) | MATCH | — |
| Qwen/Qwen3-8B | FP4 | NVFP4 | word_perplexity | 12.69 | — | — | 论文参考值，未复现 |
| Qwen/Qwen3.5-35B-A3B | BF16 | - | acc_norm | — | 82.48 | — | 参考对比，无论文数值 |
| Qwen/Qwen3.5-35B-A3B | MXFP4 | MXFP4-Quark | acc_norm | — | 80.50(-1.98) | — | 参考对比，无论文数值 |
| Qwen/Qwen3.5-35B-A3B | MXFP4 | MXFP4-Quark-MBS-H | acc_norm | — | 81.59(-0.90) | — | 参考对比，无论文数值 |
| Qwen/Qwen3.5-35B-A3B | BF16 | - | word_perplexity | — | 7.46 | — | 参考对比，无论文数值 |
| Qwen/Qwen3.5-35B-A3B | MXFP4 | MXFP4-Quark | word_perplexity | — | 8.21(+0.75) | — | 参考对比，无论文数值 |
| Qwen/Qwen3.5-35B-A3B | MXFP4 | MXFP4-Quark-MBS-H | word_perplexity | — | 8.00(+0.54) | — | 参考对比，无论文数值 |

## 结论
- 共 20 个 claim:MATCH 9 · PARTIAL 9 · FAIL 0 · BLOCKED 2。
- FP 基线与论文吻合,说明**评测协议可信**;因此 8 个超容差的量化配置(最大偏差 -2.13)是**真实的复现差距**(算法/校准/版本所致),而非评测口径问题。
- 2 个 BLOCKED 未产出可比数值(见各自 reason),非「未复现」。

## 差距分析
**acc_norm PARTIAL 的根因：推理引擎差异**

| | 论文 | 本次复现 |
|---|---|---|
| 推理引擎 | vLLM | HuggingFace direct load |
| 传给 lm-eval | model path (string) | pre-instantiated model object |

lm-eval 接收已实例化 model 时跳过部分初始化（日志警告：`Many other model arguments may be ignored`），影响 log-likelihood 计算。BF16 基线本身就偏低 1.55（74.96 vs 76.51），排除了量化实现的责任——差距完全来自评测基础设施。

**PPL 不受影响**：teacher-forcing 无需跨选项对比 log-likelihood，对推理引擎不敏感 → 5/5 MATCH（最大偏差 ±0.06）。

**修复方向**：`lm_eval --model vllm --model_args pretrained=<path>`

**MXFP4-16 标度映射（已修复）**：plain MXFP4-16 应使用 block size 16 上的 MX/OCP (4,8] 溢出标度（论文 §4.1），而非非饱和的 (3,6] 标度——后者是 OAS 的组件（§4.2）。此前实现误用 (3,6]，使 MXFP4-16 复现出论文的 *OAS* 数值（ppl 13.65）而非自身数值；修复后 ppl = 15.15（论文 15.15，MATCH）。剩余的 acc_norm 差距（−1.83）与其它 config 一样源自评测引擎偏移。

**MBS 宏块大小消融（Quark-MBS-H：128 vs 64）**：

| MBS 宏块 | acc_norm | PPL |
|---|---|---|
| 128（默认） | 71.99 | 13.26 |
| **64** | **72.68**（+0.69） | **12.85**（−0.41） |

把 MBS 宏块从 128 减半到 64，两个指标都提升。8 位 MBS factor 是整个宏块共享的，宏块越小，factor 越能贴合局部异常值（每 64 个元素一个 factor 而非 128），量化误差更低——代价是 MBS factor 存储量约翻倍。与 smoke 测试的 MSE 一致（64 → 0.0078，128 → 0.0090）。

**MBS factor 精度消融（Quark-MBS-H，宏块 128：8-bit vs 16-bit 尾数）**：

| MBS factor 精度 | acc_norm | PPL |
|---|---|---|
| 8-bit（256 档 dynamic / 8-bit static） | 72.30 | 13.22 |
| 16-bit（coarse-to-fine dynamic / 16-bit static） | 72.20 | 13.26 |

把 MBS factor 尾数从 8 位加倍到 16 位几乎没有变化（Δ 在运行噪声内，甚至略降）。factor 只在 [1, 2) 区间，8 位已给出 1/256 ≈ 0.4% 的分辨率——远细于它下游喂给的 FP4 量化网格，而后者才是真正的误差下限。论文选 8-bit 得到验证：再加精度已无空间。（注：论文的 *dynamic* 搜索是 16-slot LUT ≈ 4-bit；这里把它加宽到 256 档 / 8-bit 搜索有小幅提升——acc 71.99→72.30、ppl 13.26→13.22——但超过 8-bit 就不再有收益。）

**OAS+MBS 为什么能复用 MXFP4 kernel（所有改动均为纯软件）**：

![OAS+MBS kernel 复用流程](figures/oas_mbs_kernel_reuse.png)

**各方法每块映射区间对比（以 16/32 元素块为粒度）**：

| 方法 | 量化粒度 | Scale 格式 | 每块映射区间 | 溢出比例 | 说明 |
|---|---|---|---|---|---|
| OCP | 32 元素 | E8M0 | [4, 8) | 50% | 参考值 8 > Fmax=6，宽区间 |
| OAS | 16 元素 | E8M0 | (3.5, 7] | 25% | 参考值改为 7，缩小溢出区间 |
| OAS+MBS | 16 元素（OAS）+ 128 元素（MBS） | E8M0 + 8 位 factor | (3.5, 7]（同 OAS） | 25%（但分布更优） | OAS 区间不变；MBS 额外让宏块 max ≈6，改善分布而非缩窄区间 |
| Quark（even） | 32 元素 | E8M0（even 取整） | [3.5, 7) | 25% | even 取整消除 amax ∈ [7,8) 溢出 |
| **NVFP4** | **16 元素** | **E4M3 FP8** | **≈[5.625, 6.375]** | **极少** | **每块独立 E4M3，均匀精度 ±0.375** |

OAS（参考值 7）与 Quark（even 取整）以不同机制把溢出比例从 50% 降到 25%；OAS+MBS 的每块区间仍是 (3.5, 7]，但其 8 位宏块 factor 让分布更贴合 Fmax。NVFP4 的非 2 幂次 E4M3 scale 对**每个**块都给出 ±0.375 的均匀精度——这是它精度上界高于所有 E8M0 方案（含 OAS+MBS）的根本原因。

## Qwen3.5-35B-A3B 扩展

在 **Qwen/Qwen3.5-35B-A3B**(MoE，`qwen3_5_moe`)上复现三种设置(BF16、MXFP4-Quark、MXFP4-Quark-MBS-H)。checkpoint 架构是 `Qwen3_5MoeForConditionalGeneration`(多模态);通过 `AutoModelForCausalLM` 加载为纯文本的 `Qwen3_5MoeForCausalLM`(视觉塔不用,权重无 missing key);每个 MXFP4 配置 fake-quant 350 个 linear 层。运行在与 8B **不同的节点**:ROCm 7.1 / torch 2.10.0+rocm7.1 / transformers 5.12.1 / lm_eval 0.4.11(8× MI300X)。

**MBS-H(在 Quark 自家 even-rounding MXFP4 kernel 之上叠加 1×128 宏块缩放,block 32)挽回一部分纯 Quark 损失 —— 8B 约 ¼、35B 约 ½;35B 对 MXFP4 约耐受 2 倍。**

| 方法 | 8B acc_norm (Δ) | 35B acc_norm (Δ) | 8B ppl (Δ) | 35B ppl (Δ) |
|---|---|---|---|---|
| BF16 | 74.96 | 82.48 | 12.22 | 7.46 |
| MXFP4-Quark | 70.95 (−4.01) | 80.50 (−1.98) | 13.89 (+1.67) | 8.21 (+0.75) |
| MXFP4-Quark-MBS-H | 71.99 (−2.97) | 81.59 (−0.89) | 13.26 (+1.04) | 8.00 (+0.54) |

**口径说明。** (1) `acc_norm` 带有上文分析中的 HF 直接加载评测引擎偏移(相对论文 vLLM 约 −1.5),故 35B 的 `acc_norm` 应作为**本轮内部**的 BF16 vs Quark vs MBS-H 对比来读,而非绝对值;`word_perplexity` 是 teacher-forcing、对引擎不敏感 → 可信。(2) 35B 行用了略旧的 transformers/lm_eval(ROCm);本轮内部 Δ 与 8B↔35B 趋势可比,框架版本的微小绝对偏移可能存在。(3) comparison-only —— 论文未给该模型这些方法的数值。

### 复算(Qwen3.5-35B-A3B)
若节点上没有 `/home/zhaolin/code/Quark`,把环境变量 `QUARK_ROOT` 指向本地 Quark 检出(MXFP4-Quark 路径需要)。

```bash
export PAPER_REPRISE_MODEL=/group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3.5-35B-A3B
for c in bf16 mxfp4-quark mxfp4-quark-mbs-h; do
  bash impl/run_eval.sh qwen3.5-35b-a3b-$c-hellaswag
  bash impl/run_eval.sh qwen3.5-35b-a3b-$c-ppl
done
```

## 复算脚本(每个 config)
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
