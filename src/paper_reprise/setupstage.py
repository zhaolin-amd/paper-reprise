"""Setup stage: the agentic env-debug loop.

`run_setup` delegates to an injected setup executor — the bounded headless-Claude
loop that builds a conda/uv env and fixes deps until the smoke test passes once
(official path: setuploop; no-repo path: fromscratch). With no executor it falls
back to a stub env snapshot, used only for offline/contract tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from paper_reprise.models import Spec
from paper_reprise.rundir import RunDir


@dataclass
class SetupResult:
    ok: bool
    env_snapshot: dict = field(default_factory=dict)
    patches: list[str] = field(default_factory=list)
    error: str = ""


def run_setup(rd: RunDir, spec: Spec,
              executor: Optional[Callable] = None) -> SetupResult:
    if executor is not None:
        return executor(rd, spec)
    # Plan 1 stub: pretend the env was built. Real impl lands in Plan 2.
    snapshot = {"torch": "stub", "transformers": "stub", "cuda": "stub"}
    (rd.root / "env_snapshot.json").write_text(json.dumps(snapshot, indent=2))
    return SetupResult(ok=True, env_snapshot=snapshot, patches=[])


def make_setup_executor(*, manager: str = "uv", max_retries: int = 6,
                        timeout_s: float = 3600.0) -> Callable[[RunDir, Spec], SetupResult]:
    """Build the executor(rd, spec) the pipeline injects into run_setup. Imported
    lazily to avoid a circular import (setuploop imports SetupResult from here)."""
    from paper_reprise.setuploop import run_setup_loop

    def executor(rd: RunDir, spec: Spec) -> SetupResult:
        return run_setup_loop(rd, spec, manager=manager,
                              max_retries=max_retries, timeout_s=timeout_s)

    return executor
