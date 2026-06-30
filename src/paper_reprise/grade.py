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

from paper_reprise.models import Artifact, Claim, ClaimGrade, RunResult
from paper_reprise.parsers import parse_metric

# config keys whose divergence breaks faithfulness
_FAITHFUL_KEYS = ("seqlen", "stride", "wbits", "group_size", "few_shot")


def _faithfulness(claim: Claim, artifact: Artifact,
                  actual_config: dict) -> tuple[bool, list[tuple]]:
    """Returns (ok, diffs) where each diff is (key, spec_value, actual_value) — language
    neutral, so the reason can be rendered in either language by the caller."""
    expected_cfg = {}
    ep = claim.eval_protocol
    if ep.seqlen is not None:
        expected_cfg["seqlen"] = ep.seqlen
    if ep.stride is not None:
        expected_cfg["stride"] = ep.stride
    if ep.few_shot is not None:
        expected_cfg["few_shot"] = ep.few_shot
    for k in ("wbits", "group_size"):
        if k in artifact.quant_config:
            expected_cfg[k] = artifact.quant_config[k]

    diffs = []
    for k in _FAITHFUL_KEYS:
        if k in expected_cfg and k in actual_config:
            if expected_cfg[k] != actual_config[k]:
                diffs.append((k, expected_cfg[k], actual_config[k]))
    # LIMITATION (deferred): actual_config is currently the spec-RESOLVED config the
    # executor launched with (runexec.resolve_actual_config), not values introspected
    # from the black-box eval. So on the official path this check compares spec against
    # spec-derived values and passes by construction; its real teeth this phase are
    # calib_status==UNKNOWN -> BLOCKED and the setup_patches trail in the report. A
    # future pass should override specific keys with values PARSED from the eval log to
    # catch a script that silently diverged, and treat a missing key as "unverified"
    # rather than a vacuous pass.
    return (len(diffs) == 0, diffs)


def _fmt_diffs(diffs: list[tuple], lang: str) -> str:
    word = "mismatch" if lang == "en" else "不一致"
    return "; ".join(f"{k} {word} (spec={s} actual={a})" for k, s, a in diffs)


def grade_claim(claim: Claim, artifact: Artifact, run: RunResult,
                actual_config: dict) -> ClaimGrade:
    def blocked(reason_en, reason_zh):
        return ClaimGrade(claim_id=claim.id, verdict="BLOCKED", measured=None,
                          expected=claim.expected, reason=reason_en, reason_zh=reason_zh,
                          checks={"value": False, "faithful": False})

    # --- BLOCKED short-circuits ---
    if run.status == "blocked":
        br = run.block_reason or "unknown"
        return blocked(f"run did not complete: {br}", f"run 未跑成: {br}")

    if artifact.calib_status == "UNKNOWN":
        return blocked("calibration config missing (calib_status=UNKNOWN); not comparable",
                       "calib 配置缺失 (calib_status=UNKNOWN),结果不可比")

    text = ""
    p = Path(run.stdout_path)
    if p.exists():
        text = p.read_text()
    measured = parse_metric(claim.eval_protocol.metric, text)
    if measured is None:
        m = claim.eval_protocol.metric
        return blocked(f"could not parse {m} from output", f"无法从输出解析 {m}")

    # --- two checks ---
    value_ok = abs(measured - claim.expected) <= claim.tolerance
    faithful_ok, diffs = _faithfulness(claim, artifact, actual_config)

    if value_ok and faithful_ok:
        verdict, reason_en, reason_zh = "MATCH", "—", "—"
    elif value_ok and not faithful_ok:
        verdict = "PARTIAL"
        reason_en = "value within tolerance but process diverged: " + _fmt_diffs(diffs, "en")
        reason_zh = "数值达标但过程有偏差: " + _fmt_diffs(diffs, "zh")
    elif faithful_ok and not value_ok:
        verdict = "PARTIAL"
        delta = abs(measured - claim.expected)
        reason_en = f"process faithful but value off tolerance {delta:.4g} (>{claim.tolerance})"
        reason_zh = f"过程忠实但数值超容差 {delta:.4g} (>{claim.tolerance})"
    else:
        verdict = "FAIL"
        reason_en = "value off tolerance and process diverged: " + _fmt_diffs(diffs, "en")
        reason_zh = "数值超容差且过程有偏差: " + _fmt_diffs(diffs, "zh")

    return ClaimGrade(claim_id=claim.id, verdict=verdict, measured=measured,
                      expected=claim.expected, reason=reason_en, reason_zh=reason_zh,
                      checks={"value": value_ok, "faithful": faithful_ok})
