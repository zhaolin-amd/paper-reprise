"""Plan stage: feasibility / anomaly sentinel.

Default: silent pass (compute is not a constraint). Escalates to a user decision
only on (1) infeasible hardware, or (2) estimate wildly diverging from the paper.
"""
from __future__ import annotations

import re

from paper_reprise.models import ClaimPlan, PlanReport, Spec

# Extract the GPU family token from a free-form hardware string, ignoring count
# prefixes/suffixes: "1x H200" / "H200-141G x8" -> H200, "8x MI300X" -> MI300X.
# Covers AMD Instinct (MI…) and NVIDIA (H/A/B/V/L…, RTX…).
_GPU_FAMILY_RE = re.compile(
    r"\b(?:MI\d{2,3}[A-Z]?|[HABV]\d{2,3}|L\d{1,2}[A-Z]?|RTX\s?\d+)\b", re.IGNORECASE)


def _hardware_feasible(required: str | None, available: list[str]) -> bool:
    if not required:
        return True
    # The required GPU family token must appear in some available entry. Match the
    # family by pattern (not the first whitespace token) so a count like "1x H200"
    # resolves to H200, not "1x".
    m = _GPU_FAMILY_RE.search(required)
    family = (m.group(0) if m else required.split("-")[0].split()[0]).upper().replace(" ", "")
    return any(family in a.upper().replace(" ", "") for a in available)


def build_plan(spec: Spec, available_hardware: list[str]) -> PlanReport:
    claims: list[ClaimPlan] = []
    reasons: list[str] = []
    for c in spec.claims:
        feasible = _hardware_feasible(c.hardware, available_hardware)
        anomaly = None
        if not feasible:
            anomaly = f"硬件不可行: 需要 {c.hardware},可用 {available_hardware}"
            reasons.append(f"{c.id}: {anomaly}")
        claims.append(ClaimPlan(claim_id=c.id, required_hardware=c.hardware,
                                feasible=feasible, anomaly=anomaly))

    needs = len(reasons) > 0
    return PlanReport(claims=claims, needs_user_decision=needs,
                      decision_reason="; ".join(reasons) if needs else None)
