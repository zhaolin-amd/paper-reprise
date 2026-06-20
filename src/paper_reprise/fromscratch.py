"""From-scratch provider: reproduce papers with NO official repo by implementing
the paper's quantization method from its description (design §6).

This is the sibling of the official-repo path (setuploop + runexec). Instead of
cloning + running an existing repo, headless Claude IMPLEMENTS the paper's method
as a self-contained `impl/` under the run dir, exposing ONE runnable entrypoint;
the same env-build + smoke-test + retry/timeout guardrails make it runnable, then
the run executor runs that entrypoint per claim and persists raw output. Grade is
untouched and never sees this module — isolation per §2.2.

Every real-world action (the scaffold headless call, env build, smoke/eval
subprocess, the clock) is behind an injectable seam so the whole thing is
offline-testable; the autouse fixture backstops anything that forgets.
"""
from __future__ import annotations

from paper_reprise.models import Claim, Spec
from paper_reprise.rundir import RunDir

# The single runnable entrypoint the scaffold MUST produce (design §6: "executable
# eval commands"). Kept conventional so the smoke + run commands are deterministic.
_ENTRYPOINT = "impl/run_eval.sh"


def fromscratch_smoke_command() -> str:
    """Tiny-scale invocation of the scaffolded entrypoint for the smoke test."""
    return f"bash {_ENTRYPOINT} --smoke"


def fromscratch_eval_command(claim: Claim) -> str:
    """Per-claim invocation of the scaffolded entrypoint (prints the metric)."""
    return f"bash {_ENTRYPOINT} {claim.id}"


_SCAFFOLD_TEMPLATE = """No official repo exists for this paper. IMPLEMENT the \
paper's quantization method ({methods}) from scratch, as a SELF-CONTAINED \
implementation under `impl/` in this run directory.

Read the paper LaTeX source in `paper/` and the extracted reproduction spec in \
`spec.yaml` (artifacts = quantized products, claims = one metric each). Implement \
exactly the method and eval protocol the spec describes.

Expose EXACTLY ONE runnable entrypoint `{entrypoint}` that:
  - takes a single argument: a claim id (e.g. `c1`), or `--smoke` for a tiny-scale \
self-test (a few samples, batch 1) used only to prove the code runs;
  - quantizes per the claim's artifact config and runs its eval protocol;
  - prints the resulting metric value to stdout in a parseable form \
(e.g. `perplexity: 5.80`).

HONESTY RULES (mandatory):
  - Do NOT fabricate, invent, or hard-code any result number. The entrypoint must \
COMPUTE the metric. A run that cannot compute must exit non-zero, never print a \
made-up value.
  - Do NOT read the paper's expected values or tolerances to shortcut the result.

For EACH file you create under `impl/`, append ONE line describing what it \
implements to `{patch_note}` (create the file; one line per file). When `impl/` \
and `{entrypoint}` exist and `--smoke` runs, you are done."""


def build_scaffold_prompt(rd: RunDir, spec: Spec) -> str:
    """Build the headless-claude instruction to implement the paper's method as a
    self-contained `impl/` with one runnable entrypoint. Pure string builder."""
    methods = ", ".join(sorted({a.method for a in spec.artifacts})) or "the paper's method"
    return _SCAFFOLD_TEMPLATE.format(
        methods=methods, entrypoint=_ENTRYPOINT,
        patch_note="setup_patches/scaffold.txt",
    )
