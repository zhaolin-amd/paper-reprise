"""Plan stage: feasibility / anomaly sentinel.

Default: silent pass (compute is not a constraint). Escalates to a user decision
only on (1) infeasible hardware, or (2) estimate wildly diverging from the paper.
"""
from __future__ import annotations

from paper_repro.models import ClaimPlan, PlanReport, Spec


def _hardware_feasible(required: str | None, available: list[str]) -> bool:
    if not required:
        return True
    # crude check: the required GPU family token must appear in some available entry
    family = required.split("-")[0].split()[0].upper()   # "H200-141G x8" -> "H200"
    return any(family in a.upper() for a in available)


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
