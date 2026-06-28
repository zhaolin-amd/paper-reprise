import pytest
from pydantic import ValidationError

from paper_reprise.models import (
    EvalProtocol, Artifact, Claim, Spec, RepoInfo, ReferenceRepo,
    ClaimGrade, Verdict,
)


def _min_spec(**kw):
    return Spec(
        paper="2401.00001",
        artifacts=[Artifact(id="a1", base_model="m", method="X", quant_config={"wbits": 4})],
        claims=[Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="custom", command="c",
                                                 metric="perplexity", dataset="w"),
                      expected=5.0, tolerance=0.05, source="T")],
        **kw,
    )


def test_spec_references_default_empty():
    assert _min_spec().references == []


def test_spec_references_roundtrip_and_survives_public_redaction():
    from paper_reprise.rundir import public_spec_dict
    spec = _min_spec(references=[ReferenceRepo(method="QJL",
                                               repo_url="https://github.com/x/qjl",
                                               note="see Definition 1")])
    again = Spec.model_validate(spec.model_dump())
    assert again.references[0].method == "QJL"
    assert again.references[0].repo_url == "https://github.com/x/qjl"
    # references carry no target numbers, so they remain in the redacted (public) spec
    pub = public_spec_dict(spec)
    assert pub["references"][0]["repo_url"] == "https://github.com/x/qjl"


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


def test_eval_protocol_coerces_llm_extraction_quirks():
    # specextract (LLM) sometimes emits few_shot: null and extra_args as a dict;
    # coerce instead of aborting the whole pipeline on a type mismatch.
    ep = EvalProtocol(
        runner="official", command="c", metric="avg_acc", dataset="arc_easy",
        few_shot=None,
        extra_args={"temperature": 1.0, "num_repeats": 10},
    )
    assert ep.few_shot == 0
    assert ep.extra_args == '{"num_repeats": 10, "temperature": 1.0}'  # json, sorted keys
    # a plain string passes through unchanged
    ep2 = EvalProtocol(runner="official", command="c", metric="acc", dataset="d",
                       extra_args="--limit 8")
    assert ep2.extra_args == "--limit 8"


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


def test_calib_status_accepts_uppercase_known():
    a = Artifact(id="a1", base_model="m", method="AWQ", quant_config={}, calib_status="KNOWN")
    assert a.calib_status == "known"


def test_calib_status_accepts_lowercase_unknown():
    a = Artifact(id="a1", base_model="m", method="AWQ", quant_config={}, calib_status="unknown")
    assert a.calib_status == "UNKNOWN"


def test_calib_status_rejects_garbage():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Artifact(id="a1", base_model="m", method="AWQ", quant_config={}, calib_status="maybe")


def test_calib_status_coerces_llm_calibration_phrasing():
    # specextract often writes the calibration *situation* instead of the enum; the common
    # variants must not abort the run (this was a real specextract failure on 2309.05516).
    def cs(v):
        return Artifact(id="a", base_model="m", method="X", quant_config={},
                        calib_status=v).calib_status
    assert cs("calibrated") == "known"
    assert cs("uncalibrated") == "known"
    assert cs("data-free") == "known"
    assert cs("RTN") == "known"
    assert cs("unspecified") == "UNKNOWN"
    assert cs("not stated") == "UNKNOWN"


def test_quant_config_bits_aliased_to_wbits():
    a = Artifact(id="a1", base_model="m", method="AWQ", quant_config={"bits": 2, "group_size": 128})
    assert a.quant_config["wbits"] == 2
    assert a.quant_config["bits"] == 2   # original preserved


def test_quant_config_explicit_wbits_not_overwritten():
    a = Artifact(id="a1", base_model="m", method="AWQ", quant_config={"bits": 2, "wbits": 4})
    assert a.quant_config["wbits"] == 4   # explicit wbits wins


def test_quant_config_ternary_bits_aliased():
    # the GSQ-style spec used bits: "ternary" (a string) — must alias without crashing
    a = Artifact(id="a1", base_model="m", method="AWQ",
                 quant_config={"bits": "ternary", "group_size": 128})
    assert a.quant_config["wbits"] == "ternary"
