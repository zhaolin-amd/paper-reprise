"""Regenerate the GSQ algorithm-overview diagram for the report.

GSQ (arXiv 2604.18556) is post-training scalar quantization that makes the *discrete*
grid-level assignment differentiable via a **Gumbel-Softmax relaxation**: per coordinate
it learns assignment logits (and per-group scales), takes a soft weighted sum over the K
grid levels, and minimizes block-wise output reconstruction error. Annealing the
temperature tau -> 0 collapses the soft assignment onto a single hard grid level, yielding
a deploy-ready discrete layer (GGUF K-Quant compatible).

Output: figures/gsq_overview.png (referenced by README.md / README_zh.md).
Run:  python figures/gen_gsq_diagram.py   (needs matplotlib)
Design: matplotlib mathtext (fontset='cm'); FancyBboxPatch blocks with clear spacing;
arrows start/end OUTSIDE box edges; loop-back routed in a reserved right-hand lane.
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "gsq_overview.png")

fig, ax = plt.subplots(figsize=(9.5, 10))
ax.set_xlim(0, 8.8)
ax.set_ylim(0, 10)
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
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=15, shrinkA=0, shrinkB=0), zorder=4)


CX = 3.8  # main column center

# ── learned params (blue) ────────────────────────────────────────────────────
rbox(CX, 9.35, 7.0, 1.0, "#EBF5FB", "#2980B9")   # y 8.85-9.85
t(CX, 9.62, "Learned per-block params  (warm-start: GPTQ + noise)",
  fontsize=11, weight="bold", color="#1a5276")
t(CX, 9.14,
  r"per-group scale $s$   ·   per-coordinate level logits $\ell$"
  r"   (for $b\!>\!2$: local-shift, 5 logits/coord)",
  fontsize=10, color="#1a3a55")

# ── Gumbel-Softmax soft assignment (amber, the key novelty) ──────────────────
rbox(CX, 7.25, 7.0, 1.95, "#FEF9E7", "#E0A800")   # y 6.275-8.225
t(CX, 7.92, "Gumbel-Softmax soft assignment over K grid levels",
  fontsize=10.5, weight="bold", color="#7d6608")
t(CX, 7.32,
  r"$p_i = \dfrac{\exp((\kappa\,\ell_i + g_i)/\tau)}{\sum_j \exp((\kappa\,\ell_j + g_j)/\tau)}$"
  r"$,\quad g_i \sim \mathrm{Gumbel}(0,1)$",
  fontsize=10.5, color="#5d4037")
t(CX, 6.62,
  r"$\widetilde{w} = s\cdot\sum_i p_i\, d_i$"
  r"$\qquad$ (soft & differentiable; $\tau$ = temperature)",
  fontsize=10, color="#5d4037")

# ── reconstruction loss (green) ──────────────────────────────────────────────
rbox(CX, 5.15, 6.4, 1.0, "#EAFAF1", "#27AE60", lw=2.2)   # y 4.65-5.65
t(CX, 5.42, "Block-wise output reconstruction  (calibration input X)",
  fontsize=10.5, weight="bold", color="#1e8449")
t(CX, 4.95,
  r"$\mathcal{L} = \| f(\mathbf{X};\mathbf{w}) - f(\mathbf{X};\widetilde{\mathbf{w}}) \|_F^2$",
  fontsize=11, color="#1a3a2a")

# ── optimizer update (blue) ──────────────────────────────────────────────────
rbox(CX, 3.35, 6.4, 1.0, "#EBF5FB", "#2980B9")   # y 2.85-3.85
t(CX, 3.62, "Lion update on {ℓ, s}   ·   anneal temperature τ ↓",
  fontsize=10.5, weight="bold", color="#1a5276")
t(CX, 3.15,
  "resample Gumbel noise each step; freeze block when done, move to next",
  fontsize=9.3, color="#1a3a55")

# ── final hard quantization (teal, the anneal exit) ──────────────────────────
rbox(CX, 1.8, 6.4, 0.95, "#E8F6F3", "#16A085", lw=2.0)   # y 1.325-2.275
t(CX, 2.05, "τ → 0 :  soft assignment collapses to argmax over levels",
  fontsize=10.5, weight="bold", color="#0e6655")
t(CX, 1.58,
  r"$\hat{\mathbf{w}} = s\cdot\mathbf{q}_{\rm hard}$"
  r"$\quad$ (discrete, deploy-ready — GGUF K-Quant)",
  fontsize=10, color="#0b5345")

# ── vertical arrows (endpoints just outside box edges) ───────────────────────
arr(CX, 8.83, CX, 8.30)     # params -> gumbel
arr(CX, 6.25, CX, 5.69)     # gumbel -> loss
arr(CX, 4.63, CX, 3.89)     # loss -> update
arr(CX, 2.83, CX, 2.31)     # update -> final (the tau->0 exit)
t(CX + 0.55, 2.60, r"$\tau\!\to\!0$", fontsize=9.5, color="#0e6655",
  style="italic")

# loop-back: optimizer -> params, orthogonal route in the right lane
LOOP = "#2980B9"
ax.plot([7.05, 7.9], [3.35, 3.35], color=LOOP, lw=2.0, zorder=1)   # right out of update
ax.plot([7.9, 7.9], [3.35, 9.35], color=LOOP, lw=2.0, zorder=1)    # up the lane
arr(7.9, 9.35, 7.35, 9.35, color=LOOP)                            # left into params
t(8.12, 6.3, "× steps  (τ↓)", fontsize=9.5, weight="bold", color=LOOP,
  rotation=90)

# ── key insight (bottom) ─────────────────────────────────────────────────────
rbox(CX, 0.45, 7.7, 0.82, "#f0f4ff", "#3a6cf4", lw=1.5)   # y 0.04-0.86
t(CX, 0.62,
  "Core insight:  Gumbel-Softmax makes the discrete level choice differentiable;",
  fontsize=9.6, weight="bold", color="#1a2a7a")
t(CX, 0.28,
  "annealing τ→0 turns the learned soft assignment hard. Small K (3–8) keeps it tractable.",
  fontsize=9.2, color="#333")

plt.tight_layout(pad=0.3)
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}")
