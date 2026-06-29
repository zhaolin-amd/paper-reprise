import json
from pathlib import Path

from paper_reprise.models import Artifact, Claim, EvalProtocol, Spec
from paper_reprise.rundir import RunDir
from paper_reprise.setuploop import (
    assemble_snapshot,
    build_fixer_prompt,
    collect_new_patches,
    run_setup_loop,
    select_smoke_command,
    shrink_command,
)
from paper_reprise.setupstage import SetupResult


def _spec(command="python eval_ppl.py --model m --dataset wikitext2"):
    return Spec(
        paper="p", repo=None,
        artifacts=[Artifact(id="a1", base_model="m", method="AWQ",
                            quant_config={"wbits": 4})],
        claims=[Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="official", command=command,
                                                 metric="perplexity", dataset="wikitext2"),
                      expected=5.78, tolerance=0.05, source="T")],
    )


def test_shrink_command_appends_tiny_scale_flags():
    out = shrink_command("python eval_ppl.py --model m")
    assert out == "python eval_ppl.py --model m --limit 8 --batch-size 1"


def test_select_smoke_prefers_repo_example(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    (rd.repo_dir / "examples").mkdir()
    (rd.repo_dir / "examples" / "smoke.sh").write_text("echo hi")
    cmd = select_smoke_command(rd, _spec())
    assert cmd == "bash examples/smoke.sh"


def test_select_smoke_falls_back_to_shrunk_claim_command(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")  # empty repo, no example
    cmd = select_smoke_command(rd, _spec())
    assert cmd == "python eval_ppl.py --model m --dataset wikitext2 --limit 8 --batch-size 1"


def test_assemble_snapshot_normalizes_keys():
    freeze = {
        "torch": "2.3.0", "transformers": "4.40.0", "cuda": "12.1",
        "pip_freeze": "torch==2.3.0\ntransformers==4.40.0",
    }
    snap = assemble_snapshot(freeze)
    assert snap["torch"] == "2.3.0"
    assert snap["transformers"] == "4.40.0"
    assert snap["cuda"] == "12.1"
    assert "torch==2.3.0" in snap["pip_freeze"]


def test_assemble_snapshot_keeps_rocm_for_amd_builds():
    snap = assemble_snapshot({"torch": "2.3.0", "rocm": "6.1", "pip_freeze": ""})
    assert snap["rocm"] == "6.1"
    assert snap["cuda"] == "unknown"   # an AMD build has no CUDA


def test_assemble_snapshot_fills_unknown_for_missing():
    snap = assemble_snapshot({"pip_freeze": ""})
    assert snap["torch"] == "unknown"
    assert snap["transformers"] == "unknown"
    assert snap["cuda"] == "unknown"
    assert snap["rocm"] == "unknown"


def test_collect_new_patches_returns_only_unseen(tmp_path):
    d = tmp_path / "patches"
    d.mkdir()
    (d / "patch_0.txt").write_text("pinned transformers==4.36")
    seen: set[str] = set()
    first = collect_new_patches(d, seen)
    assert first == ["pinned transformers==4.36"]
    assert "patch_0.txt" in seen
    # a second call with the same dir sees nothing new
    assert collect_new_patches(d, seen) == []
    # a newly written note is picked up, sorted by filename
    (d / "patch_1.txt").write_text("added bitsandbytes")
    assert collect_new_patches(d, seen) == ["added bitsandbytes"]


def test_build_fixer_prompt_includes_command_traceback_and_patch_contract():
    prompt = build_fixer_prompt(
        command="python eval_ppl.py --limit 8",
        output="ModuleNotFoundError: No module named 'bitsandbytes'",
        patch_note_path="setup_patches/patch_2.txt",
    )
    assert "python eval_ppl.py --limit 8" in prompt
    assert "ModuleNotFoundError" in prompt
    assert "setup_patches/patch_2.txt" in prompt
    # must instruct: one-line note per change, and forbid running real experiments
    assert "patch" in prompt.lower()
    assert "do not" in prompt.lower() or "don't" in prompt.lower()


def _fakes():
    """Return a dict of default injectable fakes a test can override per-key."""
    return dict(
        create_env=lambda env_dir, manager: (0, "env created"),
        run_smoke=lambda command, cwd, env_dir: (0, "ok"),
        freeze_env=lambda env_dir: {"torch": "2.3.0", "transformers": "4.40.0",
                                    "cuda": "12.1", "pip_freeze": "torch==2.3.0"},
        run_fixer=lambda prompt, cwd, patch_note: None,
        now=iter([0.0, 1.0, 2.0, 3.0, 4.0]).__next__,
    )


def test_loop_success_on_first_smoke_pass(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    res = run_setup_loop(rd, _spec(), manager="uv", max_retries=3, timeout_s=100.0,
                         **_fakes())
    assert isinstance(res, SetupResult)
    assert res.ok is True
    assert res.env_snapshot["torch"] == "2.3.0"
    assert res.patches == []
    # snapshot persisted to disk for the report
    snap = json.loads((rd.root / "env_snapshot.json").read_text())
    assert snap["transformers"] == "4.40.0"
    # the smoke output log was handed off into setup_log/
    assert any(rd.setup_log_dir.iterdir())


def test_loop_fails_twice_then_succeeds_and_records_patches(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    smoke_codes = iter([1, 1, 0])  # fail, fail, pass

    def fake_smoke(command, cwd, env_dir):
        c = next(smoke_codes)
        return (c, "traceback" if c else "ok")

    fix_calls = {"n": 0}

    def fake_fixer(prompt, cwd, patch_note):
        # the real agent writes a patch note describing its change
        Path(patch_note).write_text(f"patch step {fix_calls['n']}")
        fix_calls["n"] += 1

    f = _fakes()
    f.update(run_smoke=fake_smoke, run_fixer=fake_fixer,
             now=iter([0.0] * 10).__next__)
    res = run_setup_loop(rd, _spec(), manager="uv", max_retries=5, timeout_s=100.0, **f)

    assert res.ok is True
    assert fix_calls["n"] == 2                       # two fix turns before success
    assert res.patches == ["patch step 0", "patch step 1"]
    assert res.env_snapshot["torch"] == "2.3.0"


def test_loop_hits_retry_cap_returns_failure_with_log(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")

    def fake_fixer(prompt, cwd, patch_note):
        Path(patch_note).write_text("tried something")

    f = _fakes()
    f.update(run_smoke=lambda c, cwd, e: (1, "boom"),   # never passes
             run_fixer=fake_fixer, now=iter([0.0] * 20).__next__)
    res = run_setup_loop(rd, _spec(), manager="uv", max_retries=2, timeout_s=1e9, **f)

    assert res.ok is False
    assert "2 retries" in res.error
    assert "setup_log/" in res.error
    assert res.patches == ["tried something", "tried something"]   # trail preserved
    assert not (rd.root / "env_snapshot.json").exists()            # no snapshot on failure
    assert any(rd.setup_log_dir.iterdir())                         # log handed off


def test_loop_times_out_returns_failure(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    # clock jumps past the timeout on the second check
    f = _fakes()
    f.update(run_smoke=lambda c, cwd, e: (1, "boom"),
             run_fixer=lambda p, cwd, n: None,
             now=iter([0.0, 5.0, 999.0]).__next__)
    res = run_setup_loop(rd, _spec(), manager="uv", max_retries=99, timeout_s=100.0, **f)

    assert res.ok is False
    assert "timed out" in res.error


def test_loop_returns_failure_when_a_seam_raises(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    f = _fakes()
    def boom_create(env_dir, manager):
        raise RuntimeError("uv binary not found")
    f.update(create_env=boom_create)
    res = run_setup_loop(rd, _spec(), manager="uv", max_retries=2, timeout_s=100.0, **f)
    assert res.ok is False
    assert "setup crashed" in res.error


def test_loop_freeze_failure_keeps_success_with_unknown_snapshot(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    def boom_freeze(env_dir):
        raise RuntimeError("python missing in env")
    f = _fakes()
    f.update(run_smoke=lambda c, cwd, e: (0, "ok"), freeze_env=boom_freeze)
    res = run_setup_loop(rd, _spec(), manager="uv", max_retries=2, timeout_s=100.0, **f)
    assert res.ok is True                       # smoke passed → success preserved
    assert res.env_snapshot["torch"] == "unknown"


def test_run_fixer_passes_timeout_to_run_headless(monkeypatch, tmp_path):
    import paper_reprise.setuploop as sl
    captured = {}
    def fake_run_headless(**kwargs):
        captured.update(kwargs)
        from paper_reprise.headless import HeadlessResult
        return HeadlessResult(ok=True)
    monkeypatch.setattr(sl, "run_headless", fake_run_headless)
    sl._run_fixer("prompt", tmp_path, tmp_path / "patch_0.txt")
    assert captured["timeout"] == sl._FIXER_TIMEOUT_S


def test_loop_env_creation_failure_is_surfaced(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    f = _fakes()
    f.update(create_env=lambda env_dir, manager: (1, "conda not found"))
    res = run_setup_loop(rd, _spec(), manager="conda", max_retries=3, timeout_s=100.0, **f)

    assert res.ok is False
    assert "env creation failed" in res.error
    assert (rd.setup_log_dir / "create_env.log").read_text() == "conda not found"


def test_run_smoke_puts_absolute_env_bin_on_path(tmp_path, monkeypatch):
    # Same absolute-path requirement as _activated_env: a relative env/bin would break
    # once the smoke command cd's elsewhere. Run a trivial command that echoes $PATH.
    from paper_reprise.setuploop import _run_smoke
    monkeypatch.chdir(tmp_path)
    (tmp_path / "env" / "bin").mkdir(parents=True)
    code, out = _run_smoke("echo PATHIS=$PATH", tmp_path, Path("env"))   # relative env
    assert code == 0
    abs_bin = str((tmp_path / "env" / "bin").resolve())
    assert abs_bin in out
