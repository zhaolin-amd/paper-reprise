# Algorithm-Overview Diagram Guide

Every reproduction report **must** include an **algorithm-overview diagram** (both the
from-scratch and official-repo paths — it is required, not optional). It lets a reader
grasp *how* the method works — and *why* the implementation is structured the way it is —
without reading the code. The results table shows *what* the numbers are; the diagram shows
the mechanism. It is the fastest way for a reviewer to understand a reproduction.

Two reference examples already in the repo (each has a committed generator script and a
self-checked PNG):

- `runs/OAS-MBS-2603.08713-*/figures/gen_oas_mbs_diagram.py` — MXFP4 OAS+MBS, a 3-stage
  vertical flow (preprocessing → unchanged Tensor-Core GEMM → epilogue).
- `runs/AutoRound-2309.05516-*/figures/gen_signround_diagram.py` — SignRound/AutoRound, a
  per-block SignSGD optimization loop with an orthogonal loop-back arrow.

## Where it lives

- Commit the **generator script** under the run's `figures/` dir (e.g.
  `figures/gen_<name>_diagram.py`) — never a throwaway in `/tmp`. The figure must be
  reproducible and editable later.
- Save the rendered PNG to `figures/<name>.png`.
- Embed it in **both** `README.md` and `README_zh.md`, under an `## Algorithm overview` /
  `## 算法概览` section: `![...](figures/<name>.png)`.
  - These per-run READMEs are often hand-maintained (they diverge from the auto-renderer);
    edit them directly. Do **not** re-render with `paper-reprise report` afterwards or you
    will clobber the hand-written sections.

## Tooling

- **matplotlib** is the reliable default. HTML+SVG looks nicer but headless-chromium
  screenshotting is unavailable on the shared nodes (missing `libatk-1.0.so.0`), and a
  GitHub `![](...)` embeds only images — not HTML files. Stick with a matplotlib PNG.
- matplotlib usually isn't in the run venv; use a python that has it
  (e.g. a conda env) or `pip install matplotlib`.
- Building blocks: `FancyBboxPatch(boxstyle="round,pad=...")` for rounded blocks,
  `annotate("", arrowprops=...)` (or `FancyArrowPatch`) for arrows.

## Formulas

- Set `matplotlib.rcParams["mathtext.fontset"] = "cm"` and write formulas in mathtext
  (`$...$`) so symbols and subscripts render correctly
  (e.g. `$m_{\rm MBS}^8$`, `$\alpha_{\max}$`, `$\|WX-\widetilde{W}X\|_F^2$`).
- Do **not** use Unicode subscript/superscript glyphs (`m₈`, `αₚ`, `W̃`) — they render
  badly. Use mathtext instead.

## Layout & spacing (keep it aesthetic)

There is no fixed pixel gap — **judge the spacing from the figure size** (bigger canvas →
bigger gaps) and make it look clean. The hard rules:

- Leave a clear gap between every pair of boxes. Nothing cramped, touching, or
  overlapping — horizontally or vertically.
- **Arrow endpoints sit outside the box edges.** Start the tail just below/beside the
  source box and end the head just before the target box, with a small gap at both ends.
  An arrow's head or tail must never land inside a box. Use `shrinkA=0, shrinkB=0` and
  place the endpoints yourself so matplotlib doesn't re-clip them.
- **Never route an arrow through (piercing) one box to point at another.** For loop-back
  or long arrows, reserve a clear lane — e.g. an orthogonal route down a side margin — so
  the arrow never crosses another box.

## Self-check (required)

After rendering, **Read the PNG back and inspect it**. Verify it is:

- beautiful and uncluttered, with comfortable spacing;
- free of garbled glyphs, box overlaps, and arrows poking into or through boxes;
- accurate — formulas correct, and it captures the algorithm's *core idea* (not every
  detail).

Fix any issue, regenerate, and re-check before committing.
