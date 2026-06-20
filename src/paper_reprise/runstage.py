"""Run stage: deterministic execution of quant + eval per claim.

The actual quantization/eval is delegated to an `executor` callable (one impl per
provider: official-repo now, from-scratch later). Plan 1 wires the contract and
the on-disk write / blocked-handling; Plan 2 supplies the real GPU executor.

executor(claim, artifact, claim_dir) -> dict with keys:
  stdout_path, actual_config, gpu, seed, minutes
"""
from __future__ import annotations

from typing import Callable

from paper_reprise.models import RunResult, Spec
from paper_reprise.rundir import RunDir


def run_claims(rd: RunDir, spec: Spec,
               executor: Callable) -> tuple[list[RunResult], dict]:
    artifacts = {a.id: a for a in spec.artifacts}
    results: list[RunResult] = []
    actual_configs: dict = {}
    for claim in spec.claims:
        claim_dir = rd.claim_dir(claim.id)
        artifact = artifacts[claim.artifact]
        try:
            out = executor(claim, artifact, claim_dir)
            results.append(RunResult(
                claim_id=claim.id, command=claim.eval_protocol.command,
                seed=out.get("seed"), gpu=out.get("gpu"), minutes=out.get("minutes"),
                stdout_path=out["stdout_path"], status="ran"))
            actual_configs[claim.id] = out.get("actual_config", {})
        except Exception as e:
            results.append(RunResult(
                claim_id=claim.id, command=claim.eval_protocol.command,
                stdout_path=str(claim_dir / "stdout.log"),
                status="blocked", block_reason=str(e)))
            actual_configs[claim.id] = {}
    return results, actual_configs
