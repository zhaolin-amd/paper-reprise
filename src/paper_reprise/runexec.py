"""Run stage: the real GPU executor (design §4.2).

Runs each claim's eval command inside the setup-built env, in the cloned repo,
persists raw stdout, and reports run metadata + the resolved actual_config. It
only RUNS and records — it never parses metrics or computes verdicts (grade does
that later, from the persisted output). Every real-world action (the eval
subprocess, GPU detection, the clock) is behind an injectable seam so the
orchestration is offline-testable.
"""
from __future__ import annotations

import re
from typing import Optional

from paper_reprise.models import Artifact, Claim

_SEED_PATTERNS = (
    r"--seed[=\s]+(\d+)",
    r"\bseed=(\d+)",
)


def build_eval_command(claim: Claim) -> str:
    """The command to run for this claim: the reproduction command specextract
    captured in the eval protocol (preferring the repo's official command is a
    specextract concern, already encoded here)."""
    return claim.eval_protocol.command


def extract_seed(command: str) -> Optional[int]:
    """Parse a seed from the command if present, else None (never guess one)."""
    for pat in _SEED_PATTERNS:
        m = re.search(pat, command)
        if m:
            return int(m.group(1))
    return None


def resolve_actual_config(claim: Claim, artifact: Artifact) -> dict:
    """The config the eval is launched with, keyed to match grade's faithfulness
    comparison (seqlen/stride/few_shot from the protocol; wbits/group_size from
    the artifact). Absent optional values are omitted, not invented."""
    ep = claim.eval_protocol
    cfg: dict = {}
    if ep.seqlen is not None:
        cfg["seqlen"] = ep.seqlen
    if ep.stride is not None:
        cfg["stride"] = ep.stride
    if ep.few_shot is not None:
        cfg["few_shot"] = ep.few_shot
    for k in ("wbits", "group_size"):
        if k in artifact.quant_config:
            cfg[k] = artifact.quant_config[k]
    return cfg
