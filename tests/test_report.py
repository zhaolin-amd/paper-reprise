from paper_reprise.models import (
    Spec, Artifact, Claim, EvalProtocol, RepoInfo, ClaimGrade, RunResult, IngestInfo,
)
from paper_reprise.report import render_reports


def _ctx():
    spec = Spec(
        paper="2401.00001",
        repo=RepoInfo(url="https://github.com/x/y", commit="abc123"),
        artifacts=[Artifact(id="a1", base_model="Llama2-7B", method="AWQ",
                            quant_config={"wbits": 4, "group_size": 128})],
        claims=[Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="official", command="python e.py",
                                                 metric="perplexity", dataset="wikitext2"),
                      expected=5.78, tolerance=0.05, source="Table 3")],
    )
    ingest = IngestInfo(arxiv_id="2401.00001", title="Test Paper",
                        source_url="https://arxiv.org/abs/2401.00001",
                        repo=RepoInfo(url="https://github.com/x/y", commit="abc123"))
    grades = [ClaimGrade(claim_id="c1", verdict="MATCH", measured=5.80, expected=5.78,
                         reason="—", checks={"value": True, "faithful": True})]
    runs = [RunResult(claim_id="c1", command="python e.py", seed=0, gpu="A100x1",
                      minutes=18.0, stdout_path="runs/c1/stdout.log")]
    env = {"torch": "2.3.0", "transformers": "4.36.0", "cuda": "12.1"}
    return spec, ingest, grades, runs, env


def test_renders_both_languages():
    spec, ingest, grades, runs, env = _ctx()
    zh, en = render_reports(spec, ingest, grades, runs, env, patches=[])
    assert "复现报告" in zh
    assert "Reproduction Report" in en


def test_uses_measured_not_expected_for_actual_column():
    spec, ingest, grades, runs, env = _ctx()
    zh, _ = render_reports(spec, ingest, grades, runs, env, patches=[])
    assert "5.80" in zh
    assert "MATCH" in zh


def test_summary_counts_verdicts():
    spec, ingest, grades, runs, env = _ctx()
    zh, _ = render_reports(spec, ingest, grades, runs, env, patches=[])
    assert "MATCH 1" in zh


def test_env_snapshot_in_report():
    spec, ingest, grades, runs, env = _ctx()
    zh, _ = render_reports(spec, ingest, grades, runs, env, patches=[])
    assert "4.36.0" in zh
    assert "abc123" in zh
