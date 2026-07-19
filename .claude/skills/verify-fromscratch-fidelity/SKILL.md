---
name: verify-fromscratch-fidelity
description: >
  Audit a paper-reprise run's numerical fidelity beyond the end2end metric: Part A
  (paper-internal closed-form properties) applies to from-scratch implementations
  (impl/, no official repo to diff against); Part B (kernel-composition correctness)
  applies to ANY run — from-scratch or official-repo — whose eval path composes with
  an imported/modified kernel (e.g. Quark's MXFP4 quantize/dequantize, or an official
  repo's own kernel swapped/patched during setup). Triggers on "does this impl
  actually match the paper", "verify this from-scratch implementation", "why is this
  claim PARTIAL/FAIL", "is the MXFP4 kernel integration correct", "before I trust
  this MATCH", "the setup loop patched/replaced a kernel — is that patch correct".
  Do NOT use for grading a claim's numeric tolerance (that's `paper-reprise grade`,
  pure code).
---

# verify-fromscratch-fidelity

An implementation that passes smoke and produces a number in tolerance is not proof
it implements the paper's *method* — it may have converged on the target number by a
different (or wrong) mechanism, or the mechanism may be right but miscoupled to
whatever kernel executes the actual quantize/dequantize. Run Part A whenever you
can't diff the algorithm's logic against an official repo (i.e. the from-scratch
path). Run Part B whenever the eval path calls an imported or modified kernel
(Quark MXFP4, a vendor GEMM, or any library/patch that owns the actual numeric
primitive) rather than the paper's own unmodified code computing it — this includes
**official-repo runs** where the setup loop patched or swapped a kernel to make the
repo's eval command run (check `setup_patches/patch_*.txt` for anything touching
kernel/CUDA/Triton code, a custom op, or a vendored quantization library); the
official-repo path's usual faithfulness signal (the setup-patch trail existing and
smoke passing) tells you a patch was made, not that the patched kernel computes the
right numbers — that's exactly what Part B checks.

Never validate any of the checks below against the paper's expected/target value
(`spec.yaml`, not `spec.public.yaml`) — that would defeat the point. Every check here
is against an INDEPENDENT source of truth: a closed-form formula, a hand-computed toy
example, a spec document, or the algorithm's own stated invariants.

## Inputs

- `run_dir` — the paper-reprise run directory (`runs/<paper>-<id>-<ts>/`).
- `run_dir/impl/` — the from-scratch implementation to audit (Part A + Part B).
- `run_dir/repo/` — the cloned official repo (Part B only, when a kernel inside it
  was patched/swapped — see `setup_patches/patch_*.txt` for what changed).
- `run_dir/paper/` — the paper's LaTeX source (ground truth for Part A).
- `run_dir/spec.public.yaml` (from-scratch) or `spec.yaml` (official-repo) — artifacts,
  methods, hyperparameters (from-scratch: expected values redacted).

Part A needs `run_dir/impl/` (from-scratch only) — skip Part A on an official-repo
run and diff against `repo/` directly instead. Part B applies to either path; check
`setup_patches/patch_*.txt` (official-repo) or `setup_patches/scaffold_*.txt`
(from-scratch) for anything that touches a kernel before deciding it's not needed.

## Part A — algorithm fidelity without an official repo

Work through as many as apply; not every paper offers every angle. Each check should
land as a runnable `impl/test_*.py` (pytest, importing the impl's own functions) plus
one line appended to that turn's `setup_patches/scaffold_*.txt` note. A check you
can't implement should still be logged as "no independent check available for X" —
silently skipping is indistinguishable from forgetting.

1. **Degenerate / limiting-case checks.** Push a hyperparameter to a value where the
   paper's own math says the method must collapse to a known baseline (bit-width to
   lossless, a correction term to zero, group size to the full tensor) and assert the
   impl reproduces that trivial case exactly, not approximately.
2. **Closed-form / textbook identities.** If the method claims a statistical property
   (unbiased estimator, MSE formula for a given source distribution, orthogonality,
   norm preservation), assert it directly on synthetic data (Gaussian/uniform inputs)
   where the formula is analytically known — independent of any real model or the
   paper's reported number.
3. **Internal monotonicity / trend claims.** Papers often assert qualitative trends
   ("accuracy degrades monotonically as group size grows", "loss decreases every
   calibration step"). Verify the trend holds across a small sweep, even when you
   can't match the paper's exact numbers.
4. **Hand-computable toy example.** Construct a tiny input (a 2x2 or 4x4 weight
   block) small enough to compute the expected output by hand or spreadsheet from the
   paper's equations; assert the impl matches it exactly.
5. **Paper's own worked example.** Check the paper body/appendix for a numeric
   walkthrough or step-by-step pseudocode trace; reproduce it literally as a test.
6. **Shared-component check against a reference repo.** When `spec.public.yaml` lists
   prerequisite-method references (a paper that builds on a prior method with its own
   repo), diff only the SHARED sub-step (e.g. the prior method's scale computation)
   against that repo — never the current paper's novel contribution, and never to
   back into the target number. The paper's restated definition wins where they
   differ; log any such override.
7. **Third-party reimplementation cross-check (secondary, non-authoritative).**
   Community reimplementations (AutoGPTQ/AutoAWQ/llm-compressor, a HF blog post,
   GitHub issues discussing the paper) can disambiguate an underspecified equation.
   Use only to choose between candidate readings, never as a source of the target
   number, and log which reading you picked and why.

## Part B — correctness when composing with an imported or modified kernel

Applies whenever the eval path (`impl/`, or a patched `repo/` on the official-repo
path) calls an external kernel (Quark MXFP4/MXFP8, a vendor GEMM, any library that
owns the actual quantize/dequantize numerics) instead of the paper's own unmodified
code computing it. The risk here is different from Part A: the algorithm's *logic*
(scale/rounding decisions) can be correct while the *coupling* to the kernel silently
breaks it (wrong scale encoding, wrong block axis, silent broadcast on a shape
mismatch) — and on the official-repo path this can hide behind an otherwise-faithful
setup-patch trail, since a patch that swaps in a working-but-different kernel still
looks like a normal "fixed a dependency" patch.

1. **Read the kernel's actual contract first.** Before wiring it in, check the
   kernel's docstring/tests/source for: scale encoding (power-of-two exponent vs
   float), fixed vs configurable block size, which axis blocking is applied along,
   rounding mode (nearest-even vs stochastic), and byte-packing layout. Do not infer
   this from a successful run — a shape-compatible but semantically wrong call can
   still execute without error.
2. **Boundary diff against a pure reference.** Write a small numpy/pure-Python
   reference that implements the SAME quantize/dequantize semantics the kernel
   claims. Feed it the exact scale/parameters your algorithm computed, on the exact
   same input tensor the kernel sees, and diff the two outputs. This isolates
   "algorithm decided the wrong scale" from "kernel executed the scale differently
   than assumed."
3. **Round-trip check.** Quantize via the kernel using your algorithm's computed
   parameters, dequantize via the same kernel, and assert the result matches what
   your algorithm's own math predicts — not just that the call succeeds.
4. **Differential fuzz test against the format spec.** For a standardized format
   (e.g. OCP MXFP4), generate random and edge-case tensors (all-zero, uniform sign,
   large dynamic range, subnormal-adjacent values) and diff the kernel's output
   against an independent reference implementation of the format spec. This validates
   the kernel itself, decoupled from your algorithm's decisions.
5. **Explicit shape/axis assertions at the call site.** Add asserts on tensor shape
   and the blocking axis immediately before the kernel call. A silent
   reshape/broadcast mismatch produces a plausible-looking but wrong number and is
   the single hardest coupling bug to catch after the fact.
6. **Bisect the end2end gap.** When a claim comes back PARTIAL/FAIL, use check 2's
   boundary diff to determine whether the discrepancy is upstream (algorithm's
   scale/rounding decision) or downstream (kernel's numeric execution of that
   decision) before touching either side. Log which side it was.
7. **Pin the kernel version.** Record the exact commit/release of the imported kernel
   library used (in `env_snapshot.json` or a patch note) — a silent upstream change to
   rounding behavior should show up as a version diff, not a mystery regression.

## Output

Append findings to the run's patch note — `setup_patches/scaffold_*.txt` on the
from-scratch path, `setup_patches/patch_*.txt` on the official-repo path (one line
per check, pass or fail, referencing the `impl/test_*.py` or `repo/`-local test that
covers it) — so they survive alongside the existing patch trail. On the official-repo
path, land the actual test file next to what it tests (inside `repo/`, following that
repo's own test conventions) rather than inventing an `impl/` that doesn't exist
there. If you want a durable audit summary, write it as `run_dir/analysis.md` —
paper-reprise's report stage appends that verbatim under `## Analysis` in both
`README.md` and `README_zh.md` without ever overwriting it.

## Gotchas

- A MATCH verdict from `grade` only means the final number is in tolerance and the
  run was faithful to the launched config — it says nothing about *why* the number
  landed there. Run Part A/B before trusting a MATCH on a paper you care about, not
  only when investigating a FAIL.
- Every check must have an independent ground truth (formula, toy example, spec doc).
  A check that reduces to "run it and see if the number looks right" is not one of
  these checks — it's just re-running the eval.
- Part B's boundary-diff reference (check B2) only needs to be accurate enough to
  isolate the bug — it does not need to be fast or complete; it exists purely as a
  slow, obviously-correct comparison point.
