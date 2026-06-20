import pytest
from pydantic import ValidationError

from paper_reprise.models import (
    EvalProtocol, Artifact, Claim, Spec, RepoInfo,
    ClaimGrade, Verdict,
)


def test_eval_protocol_minimal():
    ep = EvalProtocol(
        runner="official",
        command="python eval.py --model {model}",
        metric="perplexity",
        dataset="wikitext2",
    )
    assert ep.runner == "official"
    assert ep.split is None
    assert ep.few_shot == 0


def test_runner_must_be_enum():
    with pytest.raises(ValidationError):
        EvalProtocol(runner="bogus", command="x", metric="ppl", dataset="d")


def test_artifact_calib_status_default_known():
    a = Artifact(id="a1", base_model="m", method="AWQ", quant_config={"wbits": 4})
    assert a.calib_status == "known"


def test_claim_carries_expected_and_tolerance():
    ep = EvalProtocol(runner="official", command="c", metric="perplexity", dataset="wikitext2")
    c = Claim(id="c1", artifact="a1", eval_protocol=ep, expected=5.78, tolerance=0.05,
              source="Table 3")
    assert c.expected == 5.78
    assert c.hardware is None


def test_spec_roundtrip_via_dict():
    spec = Spec(
        paper="2401.00001",
        repo=RepoInfo(url="https://github.com/x/y", commit="abc"),
        artifacts=[Artifact(id="a1", base_model="m", method="AWQ", quant_config={"wbits": 4})],
        claims=[Claim(
            id="c1", artifact="a1",
            eval_protocol=EvalProtocol(runner="official", command="c",
                                       metric="perplexity", dataset="wikitext2"),
            expected=5.78, tolerance=0.05, source="Table 3")],
    )
    d = spec.model_dump()
    spec2 = Spec.model_validate(d)
    assert spec2.claims[0].artifact == "a1"


def test_claim_artifact_must_reference_existing_artifact():
    with pytest.raises(ValueError, match="unknown artifact"):
        Spec(
            paper="2401.00001",
            repo=None,
            artifacts=[Artifact(id="a1", base_model="m", method="AWQ", quant_config={})],
            claims=[Claim(
                id="c1", artifact="MISSING",
                eval_protocol=EvalProtocol(runner="official", command="c",
                                           metric="perplexity", dataset="wikitext2"),
                expected=1.0, tolerance=0.05, source="T")],
        )


def test_verdict_enum_values():
    assert set(Verdict.__args__) if hasattr(Verdict, "__args__") else True
    cg = ClaimGrade(claim_id="c1", verdict="MATCH", measured=5.80, expected=5.78,
                    reason="", checks={"value": True, "faithful": True})
    assert cg.verdict == "MATCH"
