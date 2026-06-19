from paper_repro.models import Spec, Artifact, Claim, EvalProtocol
from paper_repro.planstage import build_plan


def _spec_with_hw(hardware):
    return Spec(
        paper="p", repo=None,
        artifacts=[Artifact(id="a1", base_model="m", method="AWQ", quant_config={})],
        claims=[Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="official", command="c",
                                                 metric="perplexity", dataset="wikitext2"),
                      expected=1.0, tolerance=0.05, source="T", hardware=hardware)],
    )


def test_silent_pass_when_all_feasible():
    plan = build_plan(_spec_with_hw(None), available_hardware=["A100-80G"])
    assert plan.needs_user_decision is False
    assert plan.claims[0].feasible is True


def test_flags_infeasible_hardware():
    plan = build_plan(_spec_with_hw("H200-141G x8"), available_hardware=["A100-80G"])
    assert plan.needs_user_decision is True
    assert plan.claims[0].feasible is False
    assert "H200" in (plan.decision_reason or "")


def test_no_hardware_requirement_is_feasible():
    plan = build_plan(_spec_with_hw(None), available_hardware=[])
    assert plan.claims[0].feasible is True
