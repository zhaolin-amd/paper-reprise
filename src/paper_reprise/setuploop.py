"""Setup stage: the agentic env-debug loop (the only agentic stage, design §4.1).

Goal (single, machine-decidable): fix a conda/uv env until the repo's own eval
command passes a smoke test ONCE, under a retry cap AND a total timeout. On
exhausting the guardrails, return ok=False and hand off the full setup log — we
never silently give up. Setup only makes the env runnable; it never runs real
experiments or computes real numbers.

Every real-world action (env creation, smoke run, pip freeze, the headless "fix"
call) is behind an injectable seam so the whole loop is offline-testable.
"""
from __future__ import annotations

from paper_reprise.models import Spec
from paper_reprise.rundir import RunDir

# Tiny-scale flags so the smoke run is cheap (design §4.1: ~8 samples, 1 batch).
_TINY_FLAGS = "--limit 8 --batch-size 1"

# Repo files we treat as a ready-made smoke entry, in priority order.
_EXAMPLE_CANDIDATES = (
    "examples/smoke.sh",
    "examples/example.sh",
    "examples/run.sh",
    "scripts/smoke.sh",
)


def shrink_command(command: str) -> str:
    """Append tiny-scale flags to a full eval command for the smoke run."""
    return f"{command} {_TINY_FLAGS}"


def select_smoke_command(rd: RunDir, spec: Spec) -> str:
    """Pick the smoke command: repo's own example/test if present, else the first
    claim's eval command shrunk to tiny scale."""
    for rel in _EXAMPLE_CANDIDATES:
        if (rd.repo_dir / rel).is_file():
            return f"bash {rel}"
    if spec.claims:
        return shrink_command(spec.claims[0].eval_protocol.command)
    return ""
