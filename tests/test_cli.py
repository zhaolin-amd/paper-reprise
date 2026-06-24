import pytest
from click.testing import CliRunner

import paper_reprise.cli as cli_mod
import paper_reprise.runexec as runexec_mod
from paper_reprise.cli import cli


@pytest.fixture(autouse=True)
def _no_real_gpu_probe(monkeypatch):
    # CLI run/resume auto-detect hardware, and the real run executor labels each
    # claim with the GPU; keep tests hermetic + fast by not shelling out to
    # nvidia-smi/amd-smi (which can take seconds, or hit the timeout, under load).
    monkeypatch.setattr(cli_mod, "detect_available_hardware", lambda: [])
    monkeypatch.setattr(runexec_mod, "_detect_gpu", lambda: "test-gpu")


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
    # the CLI injects a dispatcher wrapping the official setup executor; with a
    # repo PRESENT it must route to that (sentinel) official executor.
    import paper_reprise.cli as cli_mod

    captured = {}

    def fake_pipeline(**kwargs):
        captured["setup_executor"] = kwargs["setup_executor"]
        from paper_reprise.pipeline import PipelineResult
        return PipelineResult(root=tmp_path, aborted_at="specextract")

    monkeypatch.setattr(cli_mod, "make_setup_executor",
                        lambda **k: (lambda rd, spec: "official-setup"))
    monkeypatch.setattr(cli_mod, "make_fromscratch_setup_executor",
                        lambda **k: (lambda rd, spec: "fromscratch-setup"))
    monkeypatch.setattr("paper_reprise.pipeline.run_pipeline", fake_pipeline)

    from click.testing import CliRunner
    res = CliRunner().invoke(
        cli_mod.cli, ["run", "2401.00001", "--base-dir", str(tmp_path), "--yes"])
    assert res.exit_code == 0

    from paper_reprise.rundir import RunDir
    rd = RunDir.create(tmp_path / "r", arxiv_id="p", timestamp="t")
    (rd.repo_dir / "setup.py").write_text("x")          # repo present
    assert captured["setup_executor"](rd, None) == "official-setup"


def test_cli_run_passes_real_run_executor(tmp_path, monkeypatch):
    # the CLI injects a dispatcher wrapping the official run executor; with a repo
    # PRESENT it must route to that (sentinel) official executor.
    import paper_reprise.cli as cli_mod

    captured = {}

    def fake_pipeline(**kwargs):
        captured["run_executor"] = kwargs["run_executor"]
        from paper_reprise.pipeline import PipelineResult
        return PipelineResult(root=tmp_path, aborted_at="specextract")

    monkeypatch.setattr(cli_mod, "make_run_executor",
                        lambda **k: (lambda c, a, cd: "official-run"))
    monkeypatch.setattr(cli_mod, "make_fromscratch_run_executor",
                        lambda **k: (lambda c, a, cd: "fromscratch-run"))
    monkeypatch.setattr("paper_reprise.pipeline.run_pipeline", fake_pipeline)

    from click.testing import CliRunner
    res = CliRunner().invoke(
        cli_mod.cli, ["run", "2401.00001", "--base-dir", str(tmp_path), "--yes"])
    assert res.exit_code == 0

    from paper_reprise.rundir import RunDir
    rd = RunDir.create(tmp_path / "r2", arxiv_id="p", timestamp="t")
    (rd.repo_dir / "main.py").write_text("x")           # repo present
    cd = rd.claim_dir("c1")
    assert captured["run_executor"](None, None, cd) == "official-run"


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


def test_cli_resume_help_lists_command():
    from click.testing import CliRunner
    import paper_reprise.cli as cli_mod
    res = CliRunner().invoke(cli_mod.cli, ["--help"])
    assert res.exit_code == 0
    assert "resume" in res.output


def test_cli_run_fetches_title_for_run_dir_name(tmp_path, monkeypatch):
    import paper_reprise.cli as cli_mod
    captured = {}

    def fake_pipeline(**kwargs):
        captured["paper_name"] = kwargs["paper_name"]
        from paper_reprise.pipeline import PipelineResult
        return PipelineResult(root=tmp_path, aborted_at="specextract")

    monkeypatch.setattr(cli_mod, "fetch_arxiv_title", lambda aid: "Some Paper Title")
    monkeypatch.setattr("paper_reprise.pipeline.run_pipeline", fake_pipeline)

    from click.testing import CliRunner
    res = CliRunner().invoke(
        cli_mod.cli, ["run", "2401.00001", "--base-dir", str(tmp_path), "--yes"])
    assert res.exit_code == 0
    assert captured["paper_name"] == "Some Paper Title"


def test_cli_run_without_yes_stops_for_spec_review(tmp_path, monkeypatch):
    # default (no --yes): approve_spec calls spec_selection_prompt interactively;
    # when the user enters "q" it returns False, pipeline aborts at spec-approval
    # and prints the retry/resume message.
    import paper_reprise.cli as cli_mod
    captured = {}

    def fake_pipeline(**kwargs):
        # call approve_spec with a minimal real spec so spec_selection_prompt works
        from paper_reprise.models import Spec, Artifact, Claim, EvalProtocol
        spec = Spec(
            paper="2401.00001", repo=None,
            artifacts=[Artifact(id="a1", base_model="org/M", method="GSQ",
                                quant_config={"wbits": 4})],
            claims=[Claim(id="c1", artifact="a1",
                          eval_protocol=EvalProtocol(runner="official", command="e",
                                                     metric="ppl", dataset="wiki"),
                          expected=5.8, tolerance=0.05, source="T1")],
        )
        from unittest.mock import patch
        with patch("paper_reprise.cli.click.prompt", return_value="q"):
            captured["approve_spec_result"] = kwargs["approve_spec"](spec)
        from paper_reprise.pipeline import PipelineResult
        return PipelineResult(root=tmp_path, aborted_at="spec-approval")

    monkeypatch.setattr(cli_mod, "fetch_arxiv_title", lambda aid: None)
    monkeypatch.setattr("paper_reprise.pipeline.run_pipeline", fake_pipeline)

    from click.testing import CliRunner
    res = CliRunner().invoke(cli_mod.cli, ["run", "2401.00001", "--base-dir", str(tmp_path)])
    assert res.exit_code == 0
    assert captured["approve_spec_result"] is False          # "q" aborts
    assert "paper-reprise run" in res.output
    assert "pick again" in res.output.lower()
    assert str(tmp_path) in res.output


def test_cli_run_with_yes_approves_spec(tmp_path, monkeypatch):
    import paper_reprise.cli as cli_mod
    captured = {}

    def fake_pipeline(**kwargs):
        captured["approve_spec_result"] = kwargs["approve_spec"](None)
        from paper_reprise.pipeline import PipelineResult
        return PipelineResult(root=tmp_path, aborted_at="specextract")

    monkeypatch.setattr(cli_mod, "fetch_arxiv_title", lambda aid: None)
    monkeypatch.setattr("paper_reprise.pipeline.run_pipeline", fake_pipeline)

    from click.testing import CliRunner
    res = CliRunner().invoke(
        cli_mod.cli, ["run", "2401.00001", "--base-dir", str(tmp_path), "--yes"])
    assert res.exit_code == 0
    assert captured["approve_spec_result"] is True           # --yes auto-approves


# ── spec_selection_prompt ────────────────────────────────────────────────────

def _make_spec(n_claims=3):
    """Build a Spec with n_claims each referencing a distinct artifact."""
    from paper_reprise.models import Spec, Artifact, Claim, EvalProtocol
    artifacts = [
        Artifact(id=f"art-{i}", base_model=f"org/Model-{i}B",
                 method="GSQ", quant_config={"wbits": 2, "group_size": 128})
        for i in range(n_claims)
    ]
    claims = [
        Claim(id=f"c{i}", artifact=f"art-{i}",
              eval_protocol=EvalProtocol(runner="official", command=f"eval {i}",
                                         metric="avg_acc", dataset="arc"),
              expected=70.0 + i, tolerance=0.5, source=f"Table 1 row {i}",
              hardware="1x A100")
        for i in range(n_claims)
    ]
    return Spec(paper="2401.00001", repo=None, artifacts=artifacts, claims=claims)


def test_selection_all_keeps_everything():
    from paper_reprise.cli import spec_selection_prompt
    from unittest.mock import patch
    spec = _make_spec(3)
    with patch("paper_reprise.cli.click.prompt", return_value="all"):
        kept = spec_selection_prompt(spec, "test-paper")
    assert kept is True
    assert len(spec.claims) == 3
    assert len(spec.artifacts) == 3


def test_selection_subset_keeps_chosen_and_prunes_orphans():
    from paper_reprise.cli import spec_selection_prompt
    from unittest.mock import patch
    spec = _make_spec(3)
    with patch("paper_reprise.cli.click.prompt", return_value="1 3"):
        kept = spec_selection_prompt(spec, "test-paper")
    assert kept is True
    assert [c.id for c in spec.claims] == ["c0", "c2"]
    # artifact for c1 (art-1) must be pruned — no claim references it anymore
    remaining_artifact_ids = {a.id for a in spec.artifacts}
    assert "art-0" in remaining_artifact_ids
    assert "art-2" in remaining_artifact_ids
    assert "art-1" not in remaining_artifact_ids


def test_selection_zero_claims_aborts():
    from paper_reprise.cli import spec_selection_prompt
    from unittest.mock import patch
    spec = _make_spec(2)
    # "5" is out of range — no valid claim selected
    with patch("paper_reprise.cli.click.prompt", return_value="5"):
        kept = spec_selection_prompt(spec, "test-paper")
    assert kept is False
    # spec is unchanged (abort before mutation)
    assert len(spec.claims) == 2
    assert len(spec.artifacts) == 2  # unchanged on abort


def test_selection_q_aborts():
    from paper_reprise.cli import spec_selection_prompt
    from unittest.mock import patch
    spec = _make_spec(2)
    with patch("paper_reprise.cli.click.prompt", return_value="q"):
        kept = spec_selection_prompt(spec, "test-paper")
    assert kept is False
    assert len(spec.claims) == 2  # unchanged
    assert len(spec.artifacts) == 2  # unchanged on abort


def test_selection_single_claim():
    from paper_reprise.cli import spec_selection_prompt
    from unittest.mock import patch
    spec = _make_spec(3)
    with patch("paper_reprise.cli.click.prompt", return_value="2"):
        kept = spec_selection_prompt(spec, "test-paper")
    assert kept is True
    assert len(spec.claims) == 1
    assert spec.claims[0].id == "c1"   # claim index 2 → c1 (1-based)
    assert len(spec.artifacts) == 1
    assert spec.artifacts[0].id == "art-1"


def test_cli_run_injects_dispatching_executors(tmp_path, monkeypatch):
    # the CLI must inject executors that route by repo presence: with an empty
    # repo dir they pick the from-scratch executor. We assert the injected setup
    # executor, run against a repo-less run dir, calls the from-scratch path.
    import paper_reprise.cli as cli_mod

    captured = {}

    def fake_pipeline(**kwargs):
        captured["setup_executor"] = kwargs["setup_executor"]
        captured["run_executor"] = kwargs["run_executor"]
        from paper_reprise.pipeline import PipelineResult
        return PipelineResult(root=tmp_path, aborted_at="specextract")

    # stub the four real factories so no real Claude/subprocess is touched; each
    # returns a labeled sentinel executor.
    monkeypatch.setattr(cli_mod, "make_setup_executor",
                        lambda **k: (lambda rd, spec: "official-setup"))
    monkeypatch.setattr(cli_mod, "make_fromscratch_setup_executor",
                        lambda **k: (lambda rd, spec: "fromscratch-setup"))
    monkeypatch.setattr(cli_mod, "make_run_executor",
                        lambda **k: (lambda c, a, cd: "official-run"))
    monkeypatch.setattr(cli_mod, "make_fromscratch_run_executor",
                        lambda **k: (lambda c, a, cd: "fromscratch-run"))
    monkeypatch.setattr("paper_reprise.pipeline.run_pipeline", fake_pipeline)

    from click.testing import CliRunner
    res = CliRunner().invoke(
        cli_mod.cli, ["run", "2401.00001", "--base-dir", str(tmp_path), "--yes"])
    assert res.exit_code == 0

    # exercise the injected dispatchers against a repo-less run dir
    from paper_reprise.models import Artifact, Claim, EvalProtocol, Spec
    from paper_reprise.rundir import RunDir
    rd = RunDir.create(tmp_path / "r", arxiv_id="p", timestamp="t")  # empty repo_dir
    spec = Spec(paper="p", repo=None,
                artifacts=[Artifact(id="a1", base_model="m", method="AWQ",
                                    quant_config={"wbits": 4})],
                claims=[Claim(id="c1", artifact="a1",
                              eval_protocol=EvalProtocol(runner="custom", command="x",
                                                         metric="perplexity",
                                                         dataset="wikitext2"),
                              expected=5.78, tolerance=0.05, source="T")])
    assert captured["setup_executor"](rd, spec) == "fromscratch-setup"
    cd = rd.claim_dir("c1")
    assert captured["run_executor"](spec.claims[0], spec.artifacts[0], cd) == "fromscratch-run"


def test_cli_run_interactive_selection_filters_spec_and_continues(tmp_path, monkeypatch):
    """Gate 1 now asks the user to select claims; selected subset gets written to
    spec.yaml and the pipeline continues (no spec-approval abort)."""
    import paper_reprise.pipeline as pipeline_mod
    from paper_reprise.models import Spec, Artifact, Claim, EvalProtocol

    captured_spec = {}

    def fake_extract_spec(rd):
        spec = Spec(
            paper="2401.00001", repo=None,
            artifacts=[
                Artifact(id="a1", base_model="org/M1", method="GSQ",
                         quant_config={"wbits": 2}),
                Artifact(id="a2", base_model="org/M2", method="GSQ",
                         quant_config={"wbits": 3}),
            ],
            claims=[
                Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="official", command="eval1",
                                                  metric="ppl", dataset="wiki"),
                      expected=5.8, tolerance=0.05, source="T1"),
                Claim(id="c2", artifact="a2",
                      eval_protocol=EvalProtocol(runner="official", command="eval2",
                                                  metric="ppl", dataset="wiki"),
                      expected=6.1, tolerance=0.05, source="T2"),
            ],
        )
        return spec

    def fake_finish_pipeline(rd, spec, ingest, **kwargs):
        captured_spec["claims"] = [c.id for c in spec.claims]
        captured_spec["artifacts"] = [a.id for a in spec.artifacts]
        return pipeline_mod.PipelineResult(root=rd.root, aborted_at=None)

    monkeypatch.setattr(pipeline_mod, "extract_spec", fake_extract_spec)
    monkeypatch.setattr(pipeline_mod, "_finish_pipeline", fake_finish_pipeline)
    monkeypatch.setattr(cli_mod, "make_fetch_sources",
                        lambda **k: (lambda rd, arxiv_id, url: None))
    monkeypatch.setattr(cli_mod, "fetch_arxiv_title", lambda arxiv_id: None)

    from unittest.mock import patch
    with patch("paper_reprise.cli.click.prompt", return_value="1"):
        res = CliRunner().invoke(
            cli_mod.cli,
            ["run", "2401.00001", "--base-dir", str(tmp_path)],
        )

    assert res.exit_code == 0
    assert "spec-approval" not in res.output   # did NOT abort
    # Only the first claim (and its artifact) made it through
    assert captured_spec["claims"] == ["c1"]
    assert captured_spec["artifacts"] == ["a1"]
    # Second artifact was pruned (no claim references it)
    assert "a2" not in captured_spec["artifacts"]


def test_cli_clean_frees_model_and_env_keeps_records(tmp_path):
    from paper_reprise.rundir import RunDir
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    ck = rd.repo_dir / "runtime" / "checkpoints" / "a"
    ck.mkdir(parents=True)
    (ck / "model.safetensors").write_bytes(b"\0" * (11 * 1024 * 1024))
    (rd.root / "env" / "bin").mkdir(parents=True)
    (rd.root / "env" / "bin" / "python").write_bytes(b"\0" * 4096)
    (rd.root / "report.zh.md").write_text("report")       # record, kept

    res = CliRunner().invoke(cli, ["clean", str(rd.root)])
    assert res.exit_code == 0
    assert not (ck / "model.safetensors").exists()        # model freed
    assert not (rd.root / "env").exists()                 # env freed
    assert (rd.root / "report.zh.md").exists()            # record kept
    assert "Freed" in res.output
