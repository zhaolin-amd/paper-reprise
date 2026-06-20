import paper_reprise.setuploop as setuploop
from paper_reprise.models import Artifact, Claim, EvalProtocol, Spec
from paper_reprise.rundir import RunDir
from paper_reprise.setupstage import SetupResult, make_setup_executor, run_setup


def _spec():
    return Spec(
        paper="p", repo=None,
        artifacts=[Artifact(id="a1", base_model="m", method="AWQ", quant_config={"wbits": 4})],
        claims=[Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="official", command="echo ppl",
                                                 metric="perplexity", dataset="wikitext2"),
                      expected=5.78, tolerance=0.05, source="T")],
    )


def test_stub_path_still_works_when_executor_none(tmp_path):
    # the Plan 1 stub seam must remain intact
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    res = run_setup(rd, _spec(), executor=None)
    assert res.ok is True
    assert "torch" in res.env_snapshot


def test_make_setup_executor_runs_loop_with_injected_io(tmp_path, monkeypatch):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    # stub every real I/O seam so no subprocess/claude is touched
    monkeypatch.setattr(setuploop, "_create_env", lambda env_dir, manager: (0, "ok"))
    monkeypatch.setattr(setuploop, "_run_smoke", lambda c, cwd, e: (0, "ok"))
    monkeypatch.setattr(setuploop, "_freeze_env",
                        lambda e: {"torch": "2.3.0", "transformers": "4.40.0",
                                   "cuda": "12.1", "pip_freeze": ""})
    monkeypatch.setattr(setuploop, "_run_fixer", lambda p, cwd, n: None)

    executor = make_setup_executor(manager="uv", max_retries=2, timeout_s=10.0)
    res = run_setup(rd, _spec(), executor=executor)
    assert isinstance(res, SetupResult)
    assert res.ok is True
    assert res.env_snapshot["torch"] == "2.3.0"
