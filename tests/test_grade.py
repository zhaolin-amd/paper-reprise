from paper_repro.models import (
    Artifact, Claim, EvalProtocol, Spec, RunResult, RepoInfo,
)
from paper_repro.grade import grade_claim


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
    out = tmp_path / "c1.log"; out.write_text("perplexity: 5.80")
    g = grade_claim(_claim(), _artifact(), _run(out), actual_config={"seqlen": 2048})
    assert g.verdict == "MATCH"
    assert g.measured == 5.80


def test_partial_when_value_off_but_faithful(tmp_path):
    out = tmp_path / "c1.log"; out.write_text("perplexity: 6.50")
    g = grade_claim(_claim(), _artifact(), _run(out), actual_config={"seqlen": 2048})
    assert g.verdict == "PARTIAL"
    assert "超容差" in g.reason or "tolerance" in g.reason.lower()


def test_partial_when_value_ok_but_config_diverged(tmp_path):
    out = tmp_path / "c1.log"; out.write_text("perplexity: 5.80")
    g = grade_claim(_claim(seqlen=2048), _artifact(), _run(out),
                    actual_config={"seqlen": 4096})
    assert g.verdict == "PARTIAL"
    assert "seqlen" in g.reason


def test_fail_when_value_off_and_config_diverged(tmp_path):
    out = tmp_path / "c1.log"; out.write_text("perplexity: 9.99")
    g = grade_claim(_claim(seqlen=2048), _artifact(), _run(out),
                    actual_config={"seqlen": 4096})
    assert g.verdict == "FAIL"


def test_blocked_when_run_blocked(tmp_path):
    out = tmp_path / "missing.log"
    g = grade_claim(_claim(), _artifact(), _run(out, status="blocked"),
                    actual_config={"seqlen": 2048})
    assert g.verdict == "BLOCKED"


def test_blocked_when_unparseable(tmp_path):
    out = tmp_path / "c1.log"; out.write_text("no number here")
    g = grade_claim(_claim(), _artifact(), _run(out), actual_config={"seqlen": 2048})
    assert g.verdict == "BLOCKED"
    assert "解析" in g.reason or "parse" in g.reason.lower()


def test_blocked_when_calib_unknown(tmp_path):
    out = tmp_path / "c1.log"; out.write_text("perplexity: 5.80")
    g = grade_claim(_claim(), _artifact(calib_status="UNKNOWN"), _run(out),
                    actual_config={"seqlen": 2048})
    assert g.verdict == "BLOCKED"
    assert "calib" in g.reason.lower()
