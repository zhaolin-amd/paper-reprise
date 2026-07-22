"""Regenerate the SignRound / AutoRound algorithm-overview diagram for the report.

SignRound (arXiv 2309.05516) learns the weight *rounding* (perturbation V) and the
weight-clip scale (alpha, beta) per transformer block via signed gradient descent
(SignSGD, ~200 steps), minimizing block-wise output reconstruction error — instead of
plain round-to-nearest (RTN). The learned V/alpha/beta are baked into the quantized
weights, so there is zero inference overhead (PTQ cost).

Output: figures/signround_overview.png (referenced by README.md / README_zh.md).
Run:  python figures/gen_signround_diagram.py   (needs matplotlib)
Design: matplotlib mathtext (fontset='cm'); FancyBboxPatch blocks with clear spacing;
arrows start/end OUTSIDE the box edges (no poking in); a curved loop-back arrow for
the T-step optimization loop routed in a reserved right-hand lane.
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "signround_overview.png")

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_xlim(0, 8.8)
ax.set_ylim(0, 9)
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


def arr(x0, y0, x1, y1, color="#8a8a8a", lw=2.0):
    # endpoints are placed just OUTSIDE the box edges by the caller
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=15,
                                shrinkA=0, shrinkB=0), zorder=4)


# ── geometry (a right-hand lane x=8.0 is reserved for the loop-back arrow) ─────
# Params  : x 0.45-7.35  y 7.50-8.50
# FP      : x 0.60-3.30  y 5.53-6.48      Quant: x 3.70-7.40  y 4.85-6.65
# Loss    : x 0.80-7.00  y 3.03-3.98
# SignSGD : x 0.80-7.00  y 1.53-2.48
# Insight : x 0.10-7.70  y 0.15-1.05

# ── trainable params (blue) ──────────────────────────────────────────────────
rbox(3.9, 8.0, 6.9, 1.0, "#EBF5FB", "#2980B9")
t(3.9, 8.36, "Per-block trainable params  —  optimized by SignSGD, T ≈ 200 steps",
  fontsize=11, weight="bold", color="#1a5276")
t(3.9, 7.86,
  r"$V \in [-0.5, 0.5]$  rounding perturbation (init 0)"
  r"$\qquad \alpha, \beta \in [0, 1]$  weight-clip scale (init 1)",
  fontsize=10.5, color="#1a3a55")

# ── FP reference branch (grey) ───────────────────────────────────────────────
rbox(1.95, 6.0, 2.7, 0.95, "#F2F3F4", "#7f8c8d")
t(1.95, 6.25, "FP reference", fontsize=10.5, weight="bold", color="#5d6d7e")
t(1.95, 5.87, r"$y_f = \mathbf{W}\,\mathbf{X}$", fontsize=11, color="#34495e")

# ── quant-dequant branch (amber) ─────────────────────────────────────────────
rbox(5.55, 5.75, 3.7, 1.8, "#FEF9E7", "#E0A800")
t(5.55, 6.38, "Quant → Dequant  (learnable)", fontsize=10.5, weight="bold",
  color="#7d6608")
t(5.55, 5.86,
  r"$s = \dfrac{\max(\mathbf{W})\,\alpha - \min(\mathbf{W})\,\beta}{2^{\rm bit}-1}$",
  fontsize=10.5, color="#5d4037")
t(5.55, 5.18,
  r"$\widetilde{\mathbf{W}} = s\cdot\mathrm{clip}(\lfloor \mathbf{W}/s + zp + V \rceil)$"
  "\n"
  r"$y_q = \widetilde{\mathbf{W}}\,\mathbf{X}$",
  fontsize=10, color="#5d4037", multialignment="center")

# params -> both branches (tails below params @7.42; heads above box tops)
arr(2.5, 7.42, 2.0, 6.57)      # -> FP (top 6.48)
arr(5.3, 7.42, 5.55, 6.74)     # -> Quant (top 6.65)
t(3.95, 6.95, r"calibration input $\mathbf{X}$", fontsize=9, style="italic",
  color="#666")

# ── loss (green) ─────────────────────────────────────────────────────────────
rbox(3.9, 3.5, 6.2, 0.95, "#EAFAF1", "#27AE60", lw=2.2)
t(3.9, 3.73, "Block-wise output reconstruction loss", fontsize=10.5,
  weight="bold", color="#1e8449")
t(3.9, 3.28, r"$\mathcal{L} = \|\,\mathbf{W}\mathbf{X} - \widetilde{\mathbf{W}}\mathbf{X}\,\|_F^2$",
  fontsize=11, color="#1a3a2a")

arr(2.1, 5.44, 3.0, 4.07)      # FP (bottom 5.53) -> loss (top 3.98)
arr(5.7, 4.77, 4.9, 4.07)      # quant (bottom 4.85) -> loss

# ── SignSGD update (blue) ────────────────────────────────────────────────────
rbox(3.9, 2.0, 6.2, 0.95, "#EBF5FB", "#2980B9")
t(3.9, 2.23, "SignSGD update  (keep best V, α, β)", fontsize=10.5,
  weight="bold", color="#1a5276")
t(3.9, 1.78, r"$\theta_{t+1} = \theta_t - lr\cdot\mathrm{sign}(g_t)$"
  r"$,\quad \theta \in \{V, \alpha, \beta\}$",
  fontsize=10.5, color="#1a3a55")

arr(3.9, 2.94, 3.9, 2.56)      # loss (bottom 3.03) -> signsgd (top 2.48)

# loop-back: orthogonal route in the reserved right lane (x=8.0), clear of all boxes
LOOP = "#2980B9"
ax.plot([7.05, 8.0], [2.0, 2.0], color=LOOP, lw=2.0, zorder=1)   # right out of SignSGD
ax.plot([8.0, 8.0], [2.0, 7.9], color=LOOP, lw=2.0, zorder=1)    # up the lane
arr(8.0, 7.9, 7.4, 7.9, color=LOOP)                              # left into params edge
t(8.22, 4.9, "× T ≈ 200", fontsize=9.5, weight="bold", color=LOOP, rotation=90)

# ── key insight (bottom) ─────────────────────────────────────────────────────
rbox(3.9, 0.6, 7.0, 0.9, "#f0f4ff", "#3a6cf4", lw=1.5)
t(3.9, 0.81,
  "Core insight:  only V, α, β are learned (bounded & tiny) — the rest is stock quant.",
  fontsize=9.8, weight="bold", color="#1a2a7a")
t(3.9, 0.43,
  r"Learned rounding is baked into $\widetilde{\mathbf{W}}$ → PTQ cost, "
  "ZERO inference overhead; beats round-to-nearest.",
  fontsize=9.3, color="#333")

plt.tight_layout(pad=0.3)
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}")
