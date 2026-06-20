from click.testing import CliRunner

import paper_reprise.cli as cli_mod
from paper_reprise.cli import cli


def test_cli_help():
    res = CliRunner().invoke(cli, ["--help"])
    assert res.exit_code == 0
    assert "run" in res.output
    assert "report" in res.output


def test_cli_run_help_lists_yes_flag():
    res = CliRunner().invoke(cli, ["run", "--help"])
    assert res.exit_code == 0
    assert "--yes" in res.output


def test_cli_report_rerenders(tmp_path, monkeypatch):
    from paper_reprise.rundir import RunDir
    from paper_reprise.models import (Spec, Artifact, Claim, EvalProtocol, RepoInfo,
                                    IngestInfo)
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    spec = Spec(paper="2401.00001", repo=RepoInfo(url="u", commit="c"),
                artifacts=[Artifact(id="a1", base_model="m", method="AWQ",
                                    quant_config={"wbits": 4})],
                claims=[Claim(id="c1", artifact="a1",
                              eval_protocol=EvalProtocol(runner="official", command="c",
                                                         metric="perplexity",
                                                         dataset="wikitext2"),
                              expected=5.78, tolerance=0.05, source="T")])
    rd.write_spec(spec)
    rd.write_ingest(IngestInfo(arxiv_id="2401.00001", source_url="u",
                               repo=RepoInfo(url="u", commit="c")))
    (rd.claim_dir("c1") / "stdout.log").write_text("perplexity: 5.80")

    res = CliRunner().invoke(cli, ["report", str(rd.root)])
    assert res.exit_code == 0
    assert (rd.root / "report.zh.md").exists()


def test_cli_run_resolves_title_then_aborts_on_specextract(tmp_path, monkeypatch):
    # title → arxiv id resolution happens; then specextract has no real spec so
    # the pipeline aborts at specextract (no GPU work needed). We assert the
    # resolver was consulted and the run dir is created under the resolved id.
    seen = {}

    def fake_resolve(query, **kwargs):
        seen["query"] = query
        return "2401.00001"

    def fake_fetch_sources_factory(**kwargs):
        def _fs(rd, arxiv_id, url):
            seen["fetched"] = arxiv_id
        return _fs

    monkeypatch.setattr(cli_mod, "resolve_arxiv_id", fake_resolve)
    monkeypatch.setattr(cli_mod, "make_fetch_sources", fake_fetch_sources_factory)
    # no real source content → specextract aborts; force it deterministically so the
    # test does not depend on a `claude` binary being absent from the environment.
    import paper_reprise.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "extract_spec", lambda rd: None)

    from click.testing import CliRunner
    res = CliRunner().invoke(
        cli_mod.cli,
        ["run", "AWQ Activation-aware Weight Quantization",
         "--base-dir", str(tmp_path), "--yes"],
    )
    assert res.exit_code == 0
    assert seen["query"] == "AWQ Activation-aware Weight Quantization"
    assert seen["fetched"] == "2401.00001"
    assert "Aborted at: specextract" in res.output


def test_cli_run_versioned_id_not_treated_as_title(tmp_path, monkeypatch):
    # a versioned bare id must NOT go to the title resolver
    called = {"resolve": 0}
    monkeypatch.setattr(cli_mod, "resolve_arxiv_id",
                        lambda q, **k: called.__setitem__("resolve", called["resolve"] + 1) or "X")

    def fake_factory(**kwargs):
        def _fs(rd, arxiv_id, url):
            called["fetched_id"] = arxiv_id
        return _fs
    monkeypatch.setattr(cli_mod, "make_fetch_sources", fake_factory)
    import paper_reprise.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "extract_spec", lambda rd: None)

    from click.testing import CliRunner
    res = CliRunner().invoke(
        cli_mod.cli, ["run", "2401.00001v2", "--base-dir", str(tmp_path), "--yes"]
    )
    assert res.exit_code == 0
    assert called["resolve"] == 0          # resolver NOT consulted
    assert called["fetched_id"] == "2401.00001"   # version stripped by normalize_input in pipeline


def test_cli_run_unresolvable_title_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "resolve_arxiv_id", lambda q, **k: None)
    from click.testing import CliRunner
    res = CliRunner().invoke(
        cli_mod.cli, ["run", "no such paper", "--base-dir", str(tmp_path), "--yes"]
    )
    assert res.exit_code != 0
    assert "could not resolve" in res.output.lower()


def test_cli_run_passes_real_setup_executor(tmp_path, monkeypatch):
    import paper_reprise.cli as cli_mod

    captured = {}

    def fake_pipeline(**kwargs):
        captured["setup_executor"] = kwargs["setup_executor"]
        from paper_reprise.pipeline import PipelineResult
        return PipelineResult(root=tmp_path, aborted_at="specextract")

    sentinel = object()
    monkeypatch.setattr(cli_mod, "make_setup_executor", lambda **k: sentinel)
    monkeypatch.setattr("paper_reprise.pipeline.run_pipeline", fake_pipeline)

    from click.testing import CliRunner
    res = CliRunner().invoke(
        cli_mod.cli, ["run", "2401.00001", "--base-dir", str(tmp_path), "--yes"])
    assert res.exit_code == 0
    assert captured["setup_executor"] is sentinel


def test_cli_run_passes_real_run_executor(tmp_path, monkeypatch):
    import paper_reprise.cli as cli_mod

    captured = {}

    def fake_pipeline(**kwargs):
        captured["run_executor"] = kwargs["run_executor"]
        from paper_reprise.pipeline import PipelineResult
        return PipelineResult(root=tmp_path, aborted_at="specextract")

    sentinel = object()
    monkeypatch.setattr(cli_mod, "make_run_executor", lambda **k: sentinel)
    monkeypatch.setattr("paper_reprise.pipeline.run_pipeline", fake_pipeline)

    from click.testing import CliRunner
    res = CliRunner().invoke(
        cli_mod.cli, ["run", "2401.00001", "--base-dir", str(tmp_path), "--yes"])
    assert res.exit_code == 0
    assert captured["run_executor"] is sentinel


def test_cli_report_reads_actual_config_for_faithfulness(tmp_path):
    # a run dir where the recorded actual_config DIVERGES from the spec → report
    # must grade it PARTIAL (faithfulness fails), not a vacuous MATCH.
    import json as _json

    from paper_reprise.models import (Artifact, Claim, EvalProtocol, IngestInfo,
                                      RepoInfo, Spec)
    from paper_reprise.rundir import RunDir
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    spec = Spec(paper="2401.00001", repo=RepoInfo(url="u", commit="c"),
                artifacts=[Artifact(id="a1", base_model="m", method="AWQ",
                                    quant_config={"wbits": 4})],
                claims=[Claim(id="c1", artifact="a1",
                              eval_protocol=EvalProtocol(runner="official", command="c",
                                                         metric="perplexity",
                                                         dataset="wikitext2", seqlen=2048),
                              expected=5.78, tolerance=0.05, source="T")])
    rd.write_spec(spec)
    rd.write_ingest(IngestInfo(arxiv_id="2401.00001", source_url="u",
                               repo=RepoInfo(url="u", commit="c")))
    cdir = rd.claim_dir("c1")
    (cdir / "stdout.log").write_text("perplexity: 5.80")          # value in tolerance
    (cdir / "actual_config.json").write_text(_json.dumps({"seqlen": 4096}))  # diverged!

    from click.testing import CliRunner
    from paper_reprise.cli import cli
    res = CliRunner().invoke(cli, ["report", str(rd.root)])
    assert res.exit_code == 0
    report = (rd.root / "report.zh.md").read_text()
    assert "PARTIAL" in report          # faithfulness caught the seqlen divergence
    assert "seqlen" in report
