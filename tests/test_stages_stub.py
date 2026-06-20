from pathlib import Path

from paper_reprise.models import Spec, Artifact, Claim, EvalProtocol
from paper_reprise.rundir import RunDir
from paper_reprise.setupstage import run_setup, SetupResult
from paper_reprise.runstage import run_claims


def _spec():
    return Spec(
        paper="p", repo=None,
        artifacts=[Artifact(id="a1", base_model="m", method="AWQ",
                            quant_config={"wbits": 4, "seqlen": 2048})],
        claims=[Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="official", command="echo ppl",
                                                 metric="perplexity", dataset="wikitext2",
                                                 seqlen=2048),
                      expected=5.78, tolerance=0.05, source="T")],
    )


def test_setup_stub_returns_env_snapshot(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    res = run_setup(rd, _spec(), executor=None)
    assert isinstance(res, SetupResult)
    assert res.ok is True
    assert "torch" in res.env_snapshot


def test_run_claims_writes_stdout_and_returns_results(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    spec = _spec()

    def fake_executor(claim, artifact, claim_dir):
        log = Path(claim_dir) / "stdout.log"
        log.write_text("perplexity: 5.80")
        return {"stdout_path": str(log), "actual_config": {"seqlen": 2048},
                "gpu": "A100x1", "seed": 0, "minutes": 1.0}

    results, configs = run_claims(rd, spec, executor=fake_executor)
    assert results[0].stdout_path.endswith("stdout.log")
    assert configs["c1"]["seqlen"] == 2048
    assert Path(results[0].stdout_path).read_text() == "perplexity: 5.80"


def test_run_claims_marks_blocked_on_executor_error(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    spec = _spec()

    def boom_executor(claim, artifact, claim_dir):
        raise RuntimeError("kernel compile failed")

    results, _ = run_claims(rd, spec, executor=boom_executor)
    assert results[0].status == "blocked"
    assert "kernel compile failed" in results[0].block_reason
