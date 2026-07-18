"""Regenerate the OAS+MBS algorithm-overview diagram embedded in the run report.

Draws a 3-stage flow (quantisation preprocessing -> unchanged MXFP4 GEMM ->
epilogue correction) showing WHY OAS+MBS reuses the stock MXFP4 kernel unchanged.
Output: figures/oas_mbs_kernel_reuse.png (referenced by README.md / README_zh.md).

Run:  python figures/gen_oas_mbs_diagram.py   (needs matplotlib)
Design notes: matplotlib mathtext (fontset='cm') for correct formula glyphs;
FancyBboxPatch rounded blocks colour-coded by execution unit; annotate() arrows.
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "oas_mbs_kernel_reuse.png")

fig, ax = plt.subplots(figsize=(8.5, 6.8))
ax.set_xlim(0, 8.5)
ax.set_ylim(0, 6.8)
ax.axis("off")
fig.patch.set_facecolor("white")


def rbox(cx, cy, w, h, fc, ec, lw=1.8):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.15", facecolor=fc,
                                edgecolor=ec, linewidth=lw, zorder=2))


def t(x, y, s, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("zorder", 3)
    ax.text(x, y, s, **kw)


def arr(y0, y1, x=4.25):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="->", color="#999", lw=2,
                                mutation_scale=16), zorder=4)


# ── Stage 0: quantisation preprocessing (pure software) ──────────────────────
rbox(4.25, 4.85, 8.0, 2.0, "#EBF5FB", "#2980B9")
t(4.25, 5.65, "Stage 0 · Quantisation preprocessing  (pure software)",
  fontsize=11, weight="bold", color="#1a5276")
t(4.25, 5.1,
  r"① MBS $\;$: $\; c = 1 + m_{\rm MBS}^8$ (8-bit mantissa of $\;6/\alpha_{\max}^{128}$)"
  "\n"
  r"$\quad\quad$pre-scale:  $X' = X \times c$"
  "\n"
  r"② OAS : $\; \mathrm{SF} = 2^k\;$ (modified selection, reference 7 instead of 8)",
  fontsize=10.5, color="#1a3a55", multialignment="center")
t(4.25, 3.95,
  r"Output:  E2M1 data $+$ E8M0 scale $\;\leftarrow\;$ standard MXFP4 format",
  fontsize=10, weight="bold", color="#196f3d")

arr(3.83, 3.33)

# ── Stage 1: MXFP4 GEMM on the Tensor Core (unchanged) ───────────────────────
rbox(4.25, 2.85, 8.0, 0.85, "#EAFAF1", "#27AE60", lw=2.5)
t(4.25, 3.05, "Stage 1 · MXFP4 GEMM  (Tensor Core)",
  fontsize=11, weight="bold", color="#1e8449")
t(4.25, 2.67, r"FP4 $\times$ FP4 $\rightarrow$ FP32  accumulator",
  fontsize=10.5, color="#1a3a2a")
ax.text(4.25, 2.35, "★  KERNEL UNCHANGED  ★",
        fontsize=10, weight="bold", color="white", ha="center", va="center",
        zorder=5, bbox=dict(boxstyle="round,pad=0.25", facecolor="#27AE60",
                            edgecolor="none"))

arr(2.21, 1.72)

# ── Stage 2: epilogue correction on the Vector Core ──────────────────────────
rbox(4.25, 1.28, 8.0, 0.78, "#FEF9E7", "#E0A800")
t(4.25, 1.5, "Stage 2 · Epilogue correction  (Vector Core)",
  fontsize=11, weight="bold", color="#7d6608")
t(4.25, 1.1,
  r"$C_{ij} \leftarrow C_{ij} \times \sigma_{A,i} \times \sigma_{B,j}$"
  r"$\quad$ where $\sigma=1/c\;$  (FMUL, < 1.6% overhead)",
  fontsize=10.5, color="#5d4037")

# ── key insight ──────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.2, 0.05), 8.1, 0.85, boxstyle="round,pad=0.1",
                            facecolor="#f0f4ff", edgecolor="#3a6cf4", lw=1.5,
                            zorder=1))
t(4.25, 0.62,
  "Core insight:  OAS & MBS change only what happens BEFORE and AFTER the kernel.",
  fontsize=10, weight="bold", color="#1a2a7a")
t(4.25, 0.28,
  r"The kernel always receives standard E2M1 $+$ E8M0  —  no hardware change needed.",
  fontsize=9.5, color="#333")

plt.tight_layout(pad=0.3)
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}")
