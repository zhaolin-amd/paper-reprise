import json
from pathlib import Path

import paper_reprise.fromscratch as fromscratch
from paper_reprise.fromscratch import (
    build_scaffold_prompt,
    fromscratch_eval_command,
    fromscratch_smoke_command,
    make_fromscratch_run_executor,
    make_fromscratch_setup_executor,
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
    prompt = build_scaffold_prompt(rd, _spec(), "setup_patches/scaffold_0.txt")
    # points at the paper source + the REDACTED spec (never the full spec.yaml as
    # the thing to read — expected/tolerance are withheld there)
    assert "paper/" in prompt
    assert "spec.public.yaml" in prompt
    assert "redact" in prompt.lower()
    # the single runnable entrypoint contract
    assert "impl/run_eval.sh" in prompt
    # the impl must honor the --tasks/--gpus overrides (env vars), so they work on
    # the from-scratch path too
    assert "${PAPER_REPRISE_TASKS:-" in prompt
    assert "${PAPER_REPRISE_GPUS:-" in prompt
    # the method to implement is surfaced from the spec
    assert "AWQ" in prompt
    # honesty rule: must NOT fabricate numbers
    low = prompt.lower()
    assert "fabricat" in low or "do not invent" in low or "must not invent" in low
    # patch-note discipline: one-line note per file
    assert "one line" in low or "one-line" in low
    # the per-turn note path is embedded so the loop can collect it
    assert "setup_patches/scaffold_0.txt" in prompt


def test_scaffold_prompt_embeds_smoke_failure_only_when_provided(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    # initial scaffold: no failure context, so no command/traceback
    initial = build_scaffold_prompt(rd, _spec(), "setup_patches/scaffold_0.txt")
    assert "bash impl/run_eval.sh --smoke" not in initial
    assert "BANG_TRACEBACK" not in initial
    # retry scaffold: failing smoke command + captured output embedded for debugging
    retry = build_scaffold_prompt(
        rd, _spec(), "setup_patches/scaffold_1.txt",
        failure=("bash impl/run_eval.sh --smoke", "BANG_TRACEBACK: boom"))
    assert "bash impl/run_eval.sh --smoke" in retry
    assert "BANG_TRACEBACK: boom" in retry


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
        # Exercise the REAL per-turn-file contract: the agent writes its note to the
        # exact relative path the prompt names (not a name the fake invents). Parsing
        # it out of the prompt is what proves each turn gets a distinct note file.
        import re
        m = re.search(r"setup_patches/scaffold_\d+\.txt", prompt)
        assert m, "prompt must name a per-turn scaffold note path"
        (rd.root / m.group(0)).write_text(f"impl awq step {n['i']}")
        n["i"] += 1
        return True

    f = _setup_fakes()
    f.update(run_smoke=fake_smoke, run_scaffold=fake_scaffold,
             now=iter([0.0] * 10).__next__)
    res = run_fromscratch_setup(rd, _spec(), max_retries=5, timeout_s=100.0, **f)
    assert res.ok is True
    assert n["i"] == 2                               # scaffolded, smoke failed, scaffolded again
    # Both turns' notes must be captured — the per-turn filename is what lets
    # collect_new_patches_scaffold see the second turn (a fixed name would be deduped).
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


def test_setup_exception_in_seam_becomes_ok_false_not_raise(tmp_path):
    # A crash anywhere in the loop body (here: create_env) must be caught and turned
    # into ok=False so the pipeline keeps going — mirrors run_setup_loop's guard.
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")

    def boom_create_env(env_dir, manager):
        raise RuntimeError("disk on fire")

    f = _setup_fakes()
    f.update(create_env=boom_create_env)
    res = run_fromscratch_setup(rd, _spec(), max_retries=3, timeout_s=100.0, **f)
    assert isinstance(res, SetupResult)
    assert res.ok is False
    assert "crashed" in res.error and "disk on fire" in res.error


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

    # command is made model-aware: PAPER_REPRISE_MODEL export + the entrypoint
    assert calls["command"].endswith("bash impl/run_eval.sh c1")
    assert calls["command"].startswith("export PAPER_REPRISE_MODEL=")
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
    # the recorded command is the resolved entrypoint that actually ran, NOT the
    # unresolved spec command (which the from-scratch path never executes)
    assert results[0].command.startswith("export PAPER_REPRISE_MODEL=")
    assert results[0].command.endswith("bash impl/run_eval.sh c1")
    assert "eval_ppl.py" not in results[0].command


def test_make_fromscratch_setup_executor_runs_with_injected_io(tmp_path, monkeypatch):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    monkeypatch.setattr(fromscratch, "_create_env", lambda env_dir, manager: (0, "ok"))
    monkeypatch.setattr(fromscratch, "_run_scaffold", lambda p, cwd, ef, to: True)
    monkeypatch.setattr(fromscratch, "_run_smoke", lambda c, cwd, e: (0, "perplexity: 5.8"))
    monkeypatch.setattr(fromscratch, "_freeze_env",
                        lambda e: {"torch": "2.3.0", "transformers": "4.40.0",
                                   "cuda": "12.1", "pip_freeze": ""})
    executor = make_fromscratch_setup_executor(max_retries=2, timeout_s=10.0)
    res = executor(rd, _spec())
    assert isinstance(res, SetupResult)
    assert res.ok is True
    assert res.env_snapshot["torch"] == "2.3.0"


def test_run_scaffold_seam_success_keyed_on_entrypoint_file(tmp_path, monkeypatch):
    # _run_scaffold returns True iff the entrypoint exists AND this turn modified
    # impl/. We stub run_headless to simulate the agent writing the file.
    expect = tmp_path / "impl" / "run_eval.sh"

    def fake_run_headless(prompt, allowed_tools, cwd, expect_file, timeout=None):
        Path(expect_file).parent.mkdir(parents=True, exist_ok=True)
        Path(expect_file).write_text("echo perplexity: 5.8")
        from paper_reprise.headless import HeadlessResult
        return HeadlessResult(ok=True, output_path=expect_file)

    monkeypatch.setattr(fromscratch, "run_headless", fake_run_headless)
    assert fromscratch._run_scaffold("prompt", tmp_path, expect, 5.0) is True


def test_run_scaffold_noop_turn_with_stale_entrypoint_returns_false(tmp_path, monkeypatch):
    # A later turn that crashed/did nothing leaves the entrypoint from an earlier
    # turn on disk. run_headless's file-existence contract would call that 'ok', but
    # _run_scaffold must report not-produced because impl/ did not change this turn.
    expect = tmp_path / "impl" / "run_eval.sh"
    expect.parent.mkdir(parents=True, exist_ok=True)
    expect.write_text("echo perplexity: 5.8")          # left by a prior turn

    def noop_run_headless(prompt, allowed_tools, cwd, expect_file, timeout=None):
        from paper_reprise.headless import HeadlessResult
        return HeadlessResult(ok=True, output_path=expect_file)   # touched nothing

    monkeypatch.setattr(fromscratch, "run_headless", noop_run_headless)
    assert fromscratch._run_scaffold("prompt", tmp_path, expect, 5.0) is False


def test_setup_writes_redacted_public_spec_without_expected(tmp_path):
    # The from-scratch setup must drop a spec.public.yaml the agent can read, with
    # the paper's expected/tolerance/source stripped; the full spec.yaml is the
    # grade artifact and is not what the agent is pointed at.
    import yaml
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    run_fromscratch_setup(rd, _spec(), max_retries=3, timeout_s=100.0, **_setup_fakes())
    pub = rd.root / "spec.public.yaml"
    assert pub.exists()
    data = yaml.safe_load(pub.read_text())
    claim = data["claims"][0]
    assert "expected" not in claim and "tolerance" not in claim and "source" not in claim
    # method + eval protocol (what must be implemented) are preserved
    assert data["artifacts"][0]["method"] == "AWQ"
    assert claim["eval_protocol"]["metric"] == "perplexity"


def test_setup_resolves_model_into_smoke_command(tmp_path):
    # The from-scratch smoke must run under the SAME model resolution as the eval
    # (PAPER_REPRISE_MODEL exported), mirroring the official path — else an impl that
    # finds the model via $PAPER_REPRISE_MODEL fails smoke but would pass eval.
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    seen = {}

    def capture_smoke(command, cwd, env_dir):
        seen["command"] = command
        return (0, "perplexity: 5.80")

    f = _setup_fakes()
    f.update(run_smoke=capture_smoke)
    run_fromscratch_setup(rd, _spec(), max_retries=2, timeout_s=100.0, **f)
    assert seen["command"].startswith("export PAPER_REPRISE_MODEL=")
    assert seen["command"].endswith("bash impl/run_eval.sh --smoke")


def test_smoke_metric_gate_accepts_metric_rejects_prose():
    from paper_reprise.fromscratch import _smoke_reported_metric
    # a standalone metric line passes, even amid log noise
    assert _smoke_reported_metric("perplexity: 5.80")
    assert _smoke_reported_metric("loading model...\naccuracy: 0.87\ndone")
    # diagnostic prose with `word: number` must NOT count as a metric
    assert not _smoke_reported_metric("exit code: 0")
    assert not _smoke_reported_metric("batch size: 8")
    assert not _smoke_reported_metric("see foo.py:123 for details")
    assert not _smoke_reported_metric("all good, no errors")


def test_setup_rejects_smoke_that_exits_zero_without_metric(tmp_path):
    # A silent exit-0 smoke (no metric printed) must NOT pass setup — it proves
    # nothing computed. With max_retries=1 and every smoke silent, setup fails.
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    f = _setup_fakes()
    f.update(run_smoke=lambda c, cwd, e: (0, "done, no errors"),
             now=iter([0.0] * 20).__next__)
    res = run_fromscratch_setup(rd, _spec(), max_retries=1, timeout_s=1e9, **f)
    assert res.ok is False
    # the augmented failure message was logged for the next turn / the operator
    assert "no parseable metric" in (rd.setup_log_dir / "smoke_0.log").read_text()
