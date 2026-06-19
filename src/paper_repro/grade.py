"""Pure-code judge. Isolated from execution: reads only spec + run's on-disk output.

Two independent checks:
  1. value:    |measured - expected| <= tolerance
  2. faithful: actual run config matches the claim's eval_protocol / artifact

Verdict matrix (design §5.1):
  MATCH   = value AND faithful
  PARTIAL = (value AND not faithful) OR (faithful AND not value)  [reason required]
  FAIL    = not value AND not faithful
  BLOCKED = run blocked / unparseable / calib UNKNOWN (not "failed to reproduce")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from paper_repro.models import Artifact, Claim, ClaimGrade, RunResult
from paper_repro.parsers import parse_metric

# config keys whose divergence breaks faithfulness
_FAITHFUL_KEYS = ("seqlen", "stride", "wbits", "group_size", "few_shot")


def _faithfulness(claim: Claim, actual_config: dict) -> tuple[bool, list[str]]:
    expected_cfg = {}
    ep = claim.eval_protocol
    if ep.seqlen is not None:
        expected_cfg["seqlen"] = ep.seqlen
    if ep.stride is not None:
        expected_cfg["stride"] = ep.stride
    if ep.few_shot is not None:
        expected_cfg["few_shot"] = ep.few_shot

    diffs = []
    for k in _FAITHFUL_KEYS:
        if k in expected_cfg and k in actual_config:
            if expected_cfg[k] != actual_config[k]:
                diffs.append(f"{k} 不一致 (spec={expected_cfg[k]} actual={actual_config[k]})")
    return (len(diffs) == 0, diffs)


def grade_claim(claim: Claim, artifact: Artifact, run: RunResult,
                actual_config: dict) -> ClaimGrade:
    # --- BLOCKED short-circuits ---
    if run.status == "blocked":
        return ClaimGrade(claim_id=claim.id, verdict="BLOCKED", measured=None,
                          expected=claim.expected,
                          reason=f"run 未跑成: {run.block_reason or 'unknown'}",
                          checks={"value": False, "faithful": False})

    if artifact.calib_status == "UNKNOWN":
        return ClaimGrade(claim_id=claim.id, verdict="BLOCKED", measured=None,
                          expected=claim.expected,
                          reason="calib 配置缺失 (calib_status=UNKNOWN),结果不可比",
                          checks={"value": False, "faithful": False})

    text = ""
    p = Path(run.stdout_path)
    if p.exists():
        text = p.read_text()
    measured = parse_metric(claim.eval_protocol.metric, text)
    if measured is None:
        return ClaimGrade(claim_id=claim.id, verdict="BLOCKED", measured=None,
                          expected=claim.expected,
                          reason=f"无法从输出解析 {claim.eval_protocol.metric}",
                          checks={"value": False, "faithful": False})

    # --- two checks ---
    value_ok = abs(measured - claim.expected) <= claim.tolerance
    faithful_ok, diffs = _faithfulness(claim, actual_config)

    if value_ok and faithful_ok:
        verdict, reason = "MATCH", "—"
    elif value_ok and not faithful_ok:
        verdict, reason = "PARTIAL", "数值达标但过程有偏差: " + "; ".join(diffs)
    elif faithful_ok and not value_ok:
        delta = abs(measured - claim.expected)
        verdict, reason = "PARTIAL", f"过程忠实但数值超容差 {delta:.4g} (>{claim.tolerance})"
    else:
        verdict, reason = "FAIL", "数值超容差且过程有偏差: " + "; ".join(diffs)

    return ClaimGrade(claim_id=claim.id, verdict=verdict, measured=measured,
                      expected=claim.expected, reason=reason,
                      checks={"value": value_ok, "faithful": faithful_ok})
