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


def test_report_includes_per_task_raw_scores(tmp_path):
    from paper_reprise.report import render_reports
    from paper_reprise.models import (Spec, Artifact, Claim, EvalProtocol,
                                       IngestInfo, ClaimGrade, RunResult)
    art = Artifact(id="a1", base_model="m", method="GSQ", quant_config={"wbits": 2})
    claim = Claim(id="c1", artifact="a1",
                  eval_protocol=EvalProtocol(runner="official", command="x",
                                             metric="acc_norm_avg", dataset="arc_easy,piqa"),
                  expected=68.55, tolerance=0.5, source="T")
    spec = Spec(paper="p", repo=None, artifacts=[art], claims=[claim])
    log = tmp_path / "stdout.log"
    log.write_text("|Tasks|Metric|Value|\n|---|---|---|\n|arc_easy|acc_norm|0.73|\n|piqa|acc|0.76|\nacc_norm_avg: 0.745\n")
    run = RunResult(claim_id="c1", command="x", stdout_path=str(log), status="ran")
    grade = ClaimGrade(claim_id="c1", verdict="PARTIAL", measured=74.5, expected=68.55,
                       reason="-", checks={"value": False, "faithful": True})
    zh, en = render_reports(spec, IngestInfo(arxiv_id="p", source_url="u"),
                            [grade], [run], env={}, patches=[])
    for doc in (zh, en):
        assert "|arc_easy|acc_norm|0.73|" in doc      # raw table embedded verbatim
        assert "|piqa|acc|0.76|" in doc
    assert "各任务原始分数" in zh and "Per-task raw scores" in en
