import json
from pathlib import Path

from paper_reprise.fromscratch import (
    build_scaffold_prompt,
    fromscratch_eval_command,
    fromscratch_smoke_command,
    make_fromscratch_run_executor,
    run_fromscratch_setup,
)
from paper_reprise.models import Artifact, Claim, EvalProtocol, Spec
from paper_reprise.rundir import RunDir
from paper_reprise.runstage import run_claims
from paper_reprise.setupstage import SetupResult


def _spec(command="python eval_ppl.py --model m --dataset wikitext2"):
    return Spec(
        paper="2401.00001", repo=None,
        artifacts=[Artifact(id="a1", base_model="m", method="AWQ",
                            quant_config={"wbits": 4, "group_size": 128})],
        claims=[Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="custom", command=command,
                                                 metric="perplexity", dataset="wikitext2"),
                      expected=5.78, tolerance=0.05, source="Table 3")],
    )


def test_smoke_command_is_tiny_scale_entrypoint():
    assert fromscratch_smoke_command() == "bash impl/run_eval.sh --smoke"


def test_eval_command_invokes_entrypoint_with_claim_id():
    assert fromscratch_eval_command(_spec().claims[0]) == "bash impl/run_eval.sh c1"


def test_scaffold_prompt_instructs_impl_entrypoint_and_forbids_fabrication(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    prompt = build_scaffold_prompt(rd, _spec())
    # points at the paper source + the spec
    assert "paper/" in prompt
    assert "spec.yaml" in prompt
    # the single runnable entrypoint contract
    assert "impl/run_eval.sh" in prompt
    # the method to implement is surfaced from the spec
    assert "AWQ" in prompt
    # honesty rule: must NOT fabricate numbers
    low = prompt.lower()
    assert "fabricat" in low or "do not invent" in low or "must not invent" in low
    # patch-note discipline: one-line note per file
    assert "one line" in low or "one-line" in low


def _setup_fakes():
    return dict(
        create_env=lambda env_dir, manager: (0, "env created"),
        run_scaffold=lambda prompt, cwd, expect_file, timeout: True,
        run_smoke=lambda command, cwd, env_dir: (0, "perplexity: 5.80"),
        freeze_env=lambda env_dir: {"torch": "2.3.0", "transformers": "4.40.0",
                                    "cuda": "12.1", "pip_freeze": "torch==2.3.0"},
        now=iter([0.0, 1.0, 2.0, 3.0, 4.0]).__next__,
    )


def test_setup_success_on_first_scaffold(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    res = run_fromscratch_setup(rd, _spec(), max_retries=3, timeout_s=100.0,
                                **_setup_fakes())
    assert isinstance(res, SetupResult)
    assert res.ok is True
    assert res.env_snapshot["torch"] == "2.3.0"
    snap = json.loads((rd.root / "env_snapshot.json").read_text())
    assert snap["transformers"] == "4.40.0"
    assert any(rd.setup_log_dir.iterdir())          # log handed off


def test_setup_retries_then_succeeds_and_records_patches(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    smoke_codes = iter([1, 0])      # first smoke fails, second passes

    def fake_smoke(command, cwd, env_dir):
        c = next(smoke_codes)
        return (c, "boom" if c else "perplexity: 5.80")

    n = {"i": 0}

    def fake_scaffold(prompt, cwd, expect_file, timeout):
        # the agent writes a patch note describing the file it implemented
        (rd.setup_patches_dir / f"scaffold_{n['i']}.txt").write_text(f"impl awq step {n['i']}")
        n["i"] += 1
        return True

    f = _setup_fakes()
    f.update(run_smoke=fake_smoke, run_scaffold=fake_scaffold,
             now=iter([0.0] * 10).__next__)
    res = run_fromscratch_setup(rd, _spec(), max_retries=5, timeout_s=100.0, **f)
    assert res.ok is True
    assert n["i"] == 2                               # scaffolded, smoke failed, scaffolded again
    assert res.patches == ["impl awq step 0", "impl awq step 1"]


def test_setup_hits_retry_cap_returns_failure_with_log(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    f = _setup_fakes()
    f.update(run_smoke=lambda c, cwd, e: (1, "still broken"),
             now=iter([0.0] * 20).__next__)
    res = run_fromscratch_setup(rd, _spec(), max_retries=2, timeout_s=1e9, **f)
    assert res.ok is False
    assert "2" in res.error and "setup_log/" in res.error
    assert not (rd.root / "env_snapshot.json").exists()
    assert any(rd.setup_log_dir.iterdir())


def test_setup_times_out_returns_failure(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    f = _setup_fakes()
    f.update(run_smoke=lambda c, cwd, e: (1, "boom"),
             now=iter([0.0, 5.0, 999.0]).__next__)
    res = run_fromscratch_setup(rd, _spec(), max_retries=99, timeout_s=100.0, **f)
    assert res.ok is False
    assert "timed out" in res.error


def test_setup_env_creation_failure_is_surfaced(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    f = _setup_fakes()
    f.update(create_env=lambda env_dir, manager: (1, "uv not found"))
    res = run_fromscratch_setup(rd, _spec(), max_retries=3, timeout_s=100.0, **f)
    assert res.ok is False
    assert "env creation failed" in res.error
    assert (rd.setup_log_dir / "create_env.log").read_text() == "uv not found"


def test_setup_scaffold_never_produces_entrypoint_fails(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    f = _setup_fakes()
    # scaffold turn never produces the entrypoint; smoke would never be reached
    f.update(run_scaffold=lambda p, cwd, ef, to: False,
             now=iter([0.0] * 20).__next__)
    res = run_fromscratch_setup(rd, _spec(), max_retries=2, timeout_s=1e9, **f)
    assert res.ok is False
    assert "setup_log/" in res.error


def _artifact():
    return Artifact(id="a1", base_model="m", method="AWQ",
                    quant_config={"wbits": 4, "group_size": 128})


def test_run_executor_runs_entrypoint_persists_and_returns_metadata(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    claim_dir = rd.claim_dir("c1")
    calls = {}

    def fake_run_eval(command, cwd, env_dir, log_path):
        calls["command"] = command
        calls["cwd"] = cwd
        calls["env_dir"] = env_dir
        Path(log_path).write_text("perplexity: 5.80")
        return 0, "perplexity: 5.80"

    claim = _spec().claims[0]
    executor = make_fromscratch_run_executor(
        run_eval=fake_run_eval, detect_gpu=lambda: "A100",
        now=iter([100.0, 220.0]).__next__)
    out = executor(claim, _artifact(), claim_dir)

    assert calls["command"] == "bash impl/run_eval.sh c1"
    assert calls["cwd"] == rd.root                  # impl lives under the run root
    assert calls["env_dir"] == rd.root / "env"
    assert out["stdout_path"] == str(claim_dir / "stdout.log")
    assert out["gpu"] == "A100"
    assert out["minutes"] == 2.0
    assert out["actual_config"]["wbits"] == 4
    assert json.loads((claim_dir / "actual_config.json").read_text())["group_size"] == 128


def test_run_executor_nonzero_exit_becomes_blocked_via_run_claims(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")

    def fail_eval(command, cwd, env_dir, log_path):
        Path(log_path).write_text("Traceback: not implemented")
        return 1, "Traceback: not implemented"

    executor = make_fromscratch_run_executor(
        run_eval=fail_eval, detect_gpu=lambda: "A100", now=iter([0.0, 60.0]).__next__)
    results, configs = run_claims(rd, _spec(), executor=executor)
    assert results[0].status == "blocked"
    assert "exited 1" in results[0].block_reason
    assert (rd.claim_dir("c1") / "stdout.log").read_text().startswith("Traceback")
    assert (rd.claim_dir("c1") / "actual_config.json").exists()
