from paper_reprise.models import Spec, Artifact, Claim, EvalProtocol
from paper_reprise.planstage import build_plan


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


def test_count_prefix_resolves_to_gpu_family():
    # "1x H200" must match the family H200, not the count token "1x"
    plan = build_plan(_spec_with_hw("1x H200"), available_hardware=["NVIDIA H200"])
    assert plan.claims[0].feasible is True


def test_amd_instinct_feasible_match():
    plan = build_plan(_spec_with_hw("8x MI300X"), available_hardware=["AMD Instinct MI300X"])
    assert plan.claims[0].feasible is True


def test_amd_family_mismatch_flagged():
    plan = build_plan(_spec_with_hw("MI355X"), available_hardware=["AMD Instinct MI300X"])
    assert plan.claims[0].feasible is False


def test_nvidia_required_amd_available_flagged():
    plan = build_plan(_spec_with_hw("1x H100"), available_hardware=["AMD Instinct MI300X"])
    assert plan.claims[0].feasible is False
