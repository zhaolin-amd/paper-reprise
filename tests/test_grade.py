from paper_reprise.models import (
    Artifact, Claim, EvalProtocol, RunResult,
)
from paper_reprise.grade import grade_claim


def _claim(seqlen=2048, calib_status="known", expected=5.78, tol=0.05):
    return Claim(
        id="c1", artifact="a1",
        eval_protocol=EvalProtocol(runner="official", command="c",
                                   metric="perplexity", dataset="wikitext2",
                                   seqlen=seqlen),
        expected=expected, tolerance=tol, source="T",
    )


def _artifact(calib_status="known"):
    return Artifact(id="a1", base_model="m", method="AWQ",
                    quant_config={"wbits": 4, "seqlen": 2048}, calib_status=calib_status)


def _run(stdout_path, status="ran"):
    return RunResult(claim_id="c1", command="c", stdout_path=str(stdout_path),
                     status=status)


def test_match_when_value_in_tol_and_faithful(tmp_path):
    out = tmp_path / "c1.log"
    out.write_text("perplexity: 5.80")
    g = grade_claim(_claim(), _artifact(), _run(out), actual_config={"seqlen": 2048})
    assert g.verdict == "MATCH"
    assert g.measured == 5.80


def test_partial_when_value_off_but_faithful(tmp_path):
    out = tmp_path / "c1.log"
    out.write_text("perplexity: 6.50")
    g = grade_claim(_claim(), _artifact(), _run(out), actual_config={"seqlen": 2048})
    assert g.verdict == "PARTIAL"
    assert "超容差" in g.reason or "tolerance" in g.reason.lower()


def test_partial_when_value_ok_but_config_diverged(tmp_path):
    out = tmp_path / "c1.log"
    out.write_text("perplexity: 5.80")
    g = grade_claim(_claim(seqlen=2048), _artifact(), _run(out),
                    actual_config={"seqlen": 4096})
    assert g.verdict == "PARTIAL"
    assert "seqlen" in g.reason


def test_fail_when_value_off_and_config_diverged(tmp_path):
    out = tmp_path / "c1.log"
    out.write_text("perplexity: 9.99")
    g = grade_claim(_claim(seqlen=2048), _artifact(), _run(out),
                    actual_config={"seqlen": 4096})
    assert g.verdict == "FAIL"


def test_blocked_when_run_blocked(tmp_path):
    out = tmp_path / "missing.log"
    g = grade_claim(_claim(), _artifact(), _run(out, status="blocked"),
                    actual_config={"seqlen": 2048})
    assert g.verdict == "BLOCKED"


def test_blocked_when_unparseable(tmp_path):
    out = tmp_path / "c1.log"
    out.write_text("no number here")
    g = grade_claim(_claim(), _artifact(), _run(out), actual_config={"seqlen": 2048})
    assert g.verdict == "BLOCKED"
    assert "解析" in g.reason or "parse" in g.reason.lower()


def test_blocked_when_calib_unknown(tmp_path):
    out = tmp_path / "c1.log"
    out.write_text("perplexity: 5.80")
    g = grade_claim(_claim(), _artifact(calib_status="UNKNOWN"), _run(out),
                    actual_config={"seqlen": 2048})
    assert g.verdict == "BLOCKED"
    assert "calib" in g.reason.lower()


def test_partial_when_wbits_diverged(tmp_path):
    out = tmp_path / "c1.log"
    out.write_text("perplexity: 5.80")
    g = grade_claim(_claim(), _artifact(), _run(out),
                    actual_config={"seqlen": 2048, "wbits": 8})
    assert g.verdict == "PARTIAL"
    assert "wbits" in g.reason


def test_match_at_exact_tolerance_boundary(tmp_path):
    out = tmp_path / "c1.log"
    out.write_text("perplexity: 5.83")
    g = grade_claim(_claim(expected=5.78, tol=0.05), _artifact(), _run(out),
                    actual_config={"seqlen": 2048})
    assert g.verdict == "MATCH"


def test_missing_actual_key_is_vacuously_faithful(tmp_path):
    out = tmp_path / "c1.log"
    out.write_text("perplexity: 5.80")
    g = grade_claim(_claim(), _artifact(), _run(out), actual_config={})
    assert g.verdict == "MATCH"


def test_bits_only_artifact_makes_wbits_faithfulness_comparable(tmp_path):
    # an artifact declared with quant_config={"bits": N} (no wbits) must still get
    # its weight bit-width compared by the faithfulness check (bits->wbits alias).
    out = tmp_path / "c1.log"
    out.write_text("perplexity: 5.80")  # value within tolerance of 5.78
    art = Artifact(id="a1", base_model="m", method="AWQ", quant_config={"bits": 4})
    g = grade_claim(_claim(), art, _run(out), actual_config={"seqlen": 2048, "wbits": 8})
    assert g.verdict == "PARTIAL"          # wbits 4 (spec) vs 8 (actual) caught
    assert "wbits" in g.reason


def test_grade_reason_is_bilingual():
    from paper_reprise.models import Artifact, Claim, EvalProtocol, RunResult
    from paper_reprise.grade import grade_claim
    art = Artifact(id="a", base_model="m", method="X", quant_config={"wbits": 4},
                   calib_status="known")
    claim = Claim(id="c", artifact="a",
                  eval_protocol=EvalProtocol(runner="custom", command="x",
                                             metric="perplexity", dataset="d"),
                  expected=5.0, tolerance=0.05, source="T")
    run = RunResult(claim_id="c", command="x", stdout_path="/nonexistent", status="blocked",
                    block_reason="boom")
    g = grade_claim(claim, art, run, actual_config={})
    assert g.verdict == "BLOCKED"
    assert g.reason == "run did not complete: boom"        # English canonical
    assert g.reason_zh == "run 未跑成: boom"                # Chinese rendering
