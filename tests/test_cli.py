
from click.testing import CliRunner

from paper_repro.cli import cli


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
    from paper_repro.rundir import RunDir
    from paper_repro.models import (Spec, Artifact, Claim, EvalProtocol, RepoInfo,
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
