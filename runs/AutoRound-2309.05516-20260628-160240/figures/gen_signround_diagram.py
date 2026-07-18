"""Regenerate the SignRound / AutoRound algorithm-overview diagram for the report.

SignRound (arXiv 2309.05516) learns the weight *rounding* (perturbation V) and the
weight-clip scale (alpha, beta) per transformer block via signed gradient descent
(SignSGD, ~200 steps), minimizing block-wise output reconstruction error — instead of
plain round-to-nearest (RTN). The learned V/alpha/beta are baked into the quantized
weights, so there is zero inference overhead (PTQ cost).

Output: figures/signround_kernel_reuse.png ... (see OUT). Referenced by README.md/_zh.
Run:  python figures/gen_signround_diagram.py   (needs matplotlib)
Design: matplotlib mathtext (fontset='cm'); FancyBboxPatch blocks; a curved loop-back
arrow for the T-step optimization loop.
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "signround_overview.png")

fig, ax = plt.subplots(figsize=(9, 8))
ax.set_xlim(0, 9)
ax.set_ylim(0, 8)
ax.axis("off")
fig.patch.set_facecolor("white")


def rbox(cx, cy, w, h, fc, ec, lw=1.8):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.12", facecolor=fc,
                                edgecolor=ec, linewidth=lw, zorder=2))


def t(x, y, s, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("zorder", 3)
    ax.text(x, y, s, **kw)


def arr(x0, y0, x1, y1, color="#888", lw=2.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                mutation_scale=16), zorder=4)


# ── trainable params (blue) ──────────────────────────────────────────────────
rbox(4.5, 7.05, 8.2, 1.0, "#EBF5FB", "#2980B9")
t(4.5, 7.42, "Per-block trainable params  —  optimized by SignSGD, T ≈ 200 steps",
  fontsize=11, weight="bold", color="#1a5276")
t(4.5, 6.92,
  r"$V \in [-0.5, 0.5]$  rounding perturbation (init 0)"
  r"$\qquad \alpha, \beta \in [0, 1]$  weight-clip scale (init 1)",
  fontsize=10.5, color="#1a3a55")

# ── FP reference branch (grey) ───────────────────────────────────────────────
rbox(2.15, 5.25, 3.5, 0.95, "#F2F3F4", "#7f8c8d")
t(2.15, 5.5, "FP reference", fontsize=10.5, weight="bold", color="#5d6d7e")
t(2.15, 5.12, r"$y_f = \mathbf{W}\,\mathbf{X}$", fontsize=11, color="#34495e")

# ── quant-dequant branch (amber) ─────────────────────────────────────────────
rbox(6.45, 5.05, 4.7, 1.9, "#FEF9E7", "#E0A800")
t(6.45, 5.72, "Quant → Dequant  (learnable)", fontsize=10.5, weight="bold",
  color="#7d6608")
t(6.45, 5.28,
  r"$s = \dfrac{\max(\mathbf{W})\,\alpha - \min(\mathbf{W})\,\beta}{2^{\rm bit}-1}$",
  fontsize=10.5, color="#5d4037")
t(6.45, 4.72,
  r"$\widetilde{\mathbf{W}} = s\cdot\mathrm{clip}\!\left(\lfloor \mathbf{W}/s + zp + V \rceil\right)$"
  r"$,\;\; y_q = \widetilde{\mathbf{W}}\,\mathbf{X}$",
  fontsize=10, color="#5d4037")

# params -> both branches ;  calib input X feeds both
arr(3.2, 6.55, 2.4, 5.75)
arr(5.8, 6.55, 6.45, 6.02)
t(4.5, 6.28, r"calibration input $\mathbf{X}$", fontsize=9, style="italic",
  color="#666")

# ── loss (green) ─────────────────────────────────────────────────────────────
rbox(4.5, 3.35, 7.0, 0.9, "#EAFAF1", "#27AE60", lw=2.2)
t(4.5, 3.58, "Block-wise output reconstruction loss", fontsize=10.5,
  weight="bold", color="#1e8449")
t(4.5, 3.15, r"$\mathcal{L} = \|\,\mathbf{W}\mathbf{X} - \widetilde{\mathbf{W}}\mathbf{X}\,\|_F^2$",
  fontsize=11, color="#1a3a2a")

arr(2.15, 4.77, 3.4, 3.82)   # FP -> loss
arr(6.45, 4.10, 5.6, 3.82)   # quant -> loss

# ── SignSGD update (blue) ────────────────────────────────────────────────────
rbox(4.5, 1.95, 7.0, 0.9, "#EBF5FB", "#2980B9")
t(4.5, 2.18, "SignSGD update  (keep best V, α, β)", fontsize=10.5,
  weight="bold", color="#1a5276")
t(4.5, 1.75, r"$\theta_{t+1} = \theta_t - lr\cdot\mathrm{sign}(g_t)$"
  r"$,\quad \theta \in \{V, \alpha, \beta\}$",
  fontsize=10.5, color="#1a3a55")

arr(4.5, 2.9, 4.5, 2.42)     # loss -> signsgd

# loop-back arrow: SignSGD (right) -> params (right)  [T steps]
ax.add_patch(FancyArrowPatch((8.05, 1.95), (8.7, 6.55),
             connectionstyle="arc3,rad=-0.55", arrowstyle="->",
             color="#2980B9", lw=2.0, mutation_scale=16, zorder=1))
t(9.15, 4.3, "× T", fontsize=10, weight="bold", color="#2980B9", rotation=90)

# ── key insight (bottom) ─────────────────────────────────────────────────────
rbox(4.5, 0.62, 8.5, 0.92, "#f0f4ff", "#3a6cf4", lw=1.5)
t(4.5, 0.83,
  "Core insight:  only V, α, β are learned (bounded & tiny) — the rest is stock quant.",
  fontsize=9.8, weight="bold", color="#1a2a7a")
t(4.5, 0.45,
  r"Learned rounding is baked into $\widetilde{\mathbf{W}}$ → PTQ cost, "
  "ZERO inference overhead; beats round-to-nearest.",
  fontsize=9.3, color="#333")

plt.tight_layout(pad=0.3)
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}")
