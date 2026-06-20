# Plan 2c: Run-Stage GPU Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `run` stage's placeholder executor with the real one — for each claim, run its eval command (the reproduction command specextract captured) inside the env that the setup stage built, in the cloned repo, persist the raw stdout, capture the run metadata (gpu/seed/minutes) and the resolved `actual_config`, and hand those back through the existing `run_claims` seam. This is the last piece: after it, `paper-reprise run <id>` goes end to end (ingest → spec → setup → run → grade → report).

**Architecture:** A new `runexec.py` module holds the real executor and its injectable I/O seams, exactly mirroring `fetch.py` / `setuploop.py`. The executor matches the existing `executor(claim, artifact, claim_dir) -> dict` contract that `run_claims` already calls (no seam change); it derives `rd.root`/`env/`/`repo/` from `claim_dir`. The eval subprocess, GPU detection, and the clock are behind injectable seams so the whole orchestration is offline-testable; the pure logic (command building, seed extraction, actual_config resolution) is tested directly. `actual_config` is also persisted to `claim_dir/actual_config.json` so the `report` re-render can grade faithfully (closing the Plan-1 TODO in `cli.py`/`grade.py`).

**Tech Stack:** Python 3.11, stdlib `subprocess`/`json`/`os`/`re`/`pathlib`/`time`, pydantic (existing models), pytest (offline via injected fakes), uv, ruff (line-length 100). `from __future__ import annotations`.

Design doc: `docs/superpowers/specs/2026-06-19-paper-reprise-agent-design.md` (§4.2 Run; §5.1 grade isolation)

---

## Context for the implementer

Plan 1 = skeleton; 2a = real ingest fetch; 2b = real setup loop. Plan 2c = the `run` stage executor. After this, nothing in the core pipeline is a stub.

The seam you MUST preserve (do not change):

- `run_claims(rd, spec, executor) -> (list[RunResult], dict)` (`runstage.py`) already does the orchestration: it calls `executor(claim, artifact, claim_dir)` per claim, expects a dict with keys `stdout_path, actual_config, gpu, seed, minutes`, builds the `RunResult`, and on ANY exception marks that claim `status="blocked"` with `block_reason=str(e)`. **Do not modify `runstage.py`.** Plan 2c only builds the executor that gets injected.
- The pipeline calls `run_claims(rd, spec, executor=run_executor)` (pipeline.py:67) and the CLI currently passes a `run_executor` that raises `"real GPU executor not implemented (Plan 2c)"` (cli.py:58-59). Replace that with `make_run_executor()`.

Relevant existing API (do not change):

- `RunDir`: `rd.claim_dir(claim_id)` returns `rd.root / "runs" / claim_id` (already mkdir-ed). So inside the executor, given `claim_dir`: `root = claim_dir.parent.parent`, `env_dir = root / "env"` (built by setup, 2b), `repo_dir = root / "repo"` (cloned by ingest, 2a).
- `Claim` / `EvalProtocol` (`models.py`): `claim.eval_protocol.command` (str), `.seqlen`, `.stride`, `.few_shot` (int, default 0); `Artifact.quant_config` (dict, may hold `wbits`/`group_size`).
- `grade._faithfulness` compares `actual_config` against the spec over keys `("seqlen", "stride", "wbits", "group_size", "few_shot")`, sourcing the expected values from `eval_protocol` (seqlen/stride/few_shot) and `artifact.quant_config` (wbits/group_size). So `resolve_actual_config` MUST return those same keys for the comparison to be meaningful.
- The env-activation pattern from 2b's `_run_smoke` (prepend `env_dir/bin` to PATH, set `VIRTUAL_ENV`) is reused here so the eval runs in the setup-built interpreter, not the ambient one.

**Run/grade separation (design §5.1):** the executor only RUNS and persists raw output + metadata. It must NOT parse metrics, compute verdicts, or read `expected`/`tolerance`. Grade (pure code) reads the persisted stdout later.

**Honest limitation on `actual_config` (read this — it shapes the design):** the executor runs `eval_protocol.command` (a black box). It can faithfully report the config it *launched with* (resolved from the spec), but it cannot introspect what the script *actually* used internally. So `resolve_actual_config` echoes the spec-resolved values. Consequence: on the official path, grade's faithfulness check (check 2) compares spec-against-spec-derived values and will pass — its real teeth in this phase come from `calib_status==UNKNOWN → BLOCKED` (handled in grade, independent of actual_config) and from the `setup_patches` trail surfaced in the report. A future pass can override specific `actual_config` keys with values *parsed from the eval log* to catch a script that silently diverged; that parsing is explicitly DEFERRED (noted in Self-Review). We persist `actual_config.json` now so that future pass and the `report` re-render have a real artifact to read.

This plan does NOT touch the from-scratch provider (design §6), nor add a quantization-as-separate-step orchestration (the captured `eval_protocol.command` is the repo's own reproduction command, which handles quant+eval; separate quant orchestration is out of scope).

---

## File Structure

```
src/paper_reprise/
  runexec.py     # NEW — real run executor + injectable seams (eval subprocess, gpu detect, clock).
  runstage.py    # unchanged (seam already correct)
  cli.py         # MODIFY — pass make_run_executor() instead of the raising stub; report reads actual_config.json
tests/
  test_runexec.py   # NEW — offline tests via injected fakes
```

**Responsibility split inside `runexec.py`:**

- Pure logic (offline-testable directly):
  - `build_eval_command(claim) -> str` — the command to run (the captured `eval_protocol.command`).
  - `extract_seed(command) -> int | None` — parse `--seed N` / `seed=N` if present.
  - `resolve_actual_config(claim, artifact) -> dict` — the spec-resolved config dict (seqlen/stride/few_shot/wbits/group_size), matching grade's comparison keys.
- Low-level injectable I/O (the only functions touching the outside world):
  - `_run_eval(command, cwd, env_dir, log_path) -> tuple[int, str]` — run the eval in the built env, persist combined output to log_path, return (exit_code, output).
  - `_detect_gpu() -> str` — best-effort GPU label (nvidia-smi / CUDA_VISIBLE_DEVICES / "unknown").
- Orchestration (testable by injecting fakes + clock):
  - `make_run_executor(*, run_eval=None, detect_gpu=None, now=None, timeout_s=7200.0) -> Callable` returns `executor(claim, artifact, claim_dir) -> dict`.

---

## Task 1: pure logic — command, seed, actual_config

**Files:**
- Create: `src/paper_reprise/runexec.py`
- Test: `tests/test_runexec.py`

- [ ] **Step 1: Write the failing test**

`tests/test_runexec.py`:
```python
from paper_reprise.models import Artifact, Claim, EvalProtocol
from paper_reprise.runexec import build_eval_command, extract_seed, resolve_actual_config


def _claim(command="python eval.py --model m", seqlen=2048, stride=2048, few_shot=0):
    return Claim(
        id="c1", artifact="a1",
        eval_protocol=EvalProtocol(runner="official", command=command,
                                   metric="perplexity", dataset="wikitext2",
                                   seqlen=seqlen, stride=stride, few_shot=few_shot),
        expected=5.78, tolerance=0.05, source="T",
    )


def _artifact(wbits=4, group_size=128):
    qc = {"wbits": wbits}
    if group_size is not None:
        qc["group_size"] = group_size
    return Artifact(id="a1", base_model="m", method="AWQ", quant_config=qc)


def test_build_eval_command_returns_protocol_command():
    assert build_eval_command(_claim("python main.py --reproduce")) == "python main.py --reproduce"


def test_extract_seed_dash_flag():
    assert extract_seed("python eval.py --seed 42 --model m") == 42


def test_extract_seed_equals_form():
    assert extract_seed("python eval.py seed=7") == 7


def test_extract_seed_absent_returns_none():
    assert extract_seed("python eval.py --model m") is None


def test_resolve_actual_config_matches_grade_keys():
    cfg = resolve_actual_config(_claim(seqlen=2048, stride=1024, few_shot=5), _artifact(wbits=4, group_size=128))
    assert cfg == {"seqlen": 2048, "stride": 1024, "few_shot": 5, "wbits": 4, "group_size": 128}


def test_resolve_actual_config_omits_absent_optional_keys():
    # no stride on the protocol, no group_size on the artifact → those keys omitted
    c = _claim(seqlen=2048, stride=None, few_shot=0)
    cfg = resolve_actual_config(c, _artifact(wbits=8, group_size=None))
    assert cfg == {"seqlen": 2048, "few_shot": 0, "wbits": 8}
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_runexec.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.runexec'`

- [ ] **Step 3: Write the implementation**

`src/paper_reprise/runexec.py`:
```python
"""Run stage: the real GPU executor (design §4.2).

Runs each claim's eval command inside the setup-built env, in the cloned repo,
persists raw stdout, and reports run metadata + the resolved actual_config. It
only RUNS and records — it never parses metrics or computes verdicts (grade does
that later, from the persisted output). Every real-world action (the eval
subprocess, GPU detection, the clock) is behind an injectable seam so the
orchestration is offline-testable.
"""
from __future__ import annotations

import re
from typing import Optional

from paper_reprise.models import Artifact, Claim

_SEED_PATTERNS = (
    r"--seed[=\s]+(\d+)",
    r"\bseed=(\d+)",
)


def build_eval_command(claim: Claim) -> str:
    """The command to run for this claim: the reproduction command specextract
    captured in the eval protocol (preferring the repo's official command is a
    specextract concern, already encoded here)."""
    return claim.eval_protocol.command


def extract_seed(command: str) -> Optional[int]:
    """Parse a seed from the command if present, else None (never guess one)."""
    for pat in _SEED_PATTERNS:
        m = re.search(pat, command)
        if m:
            return int(m.group(1))
    return None


def resolve_actual_config(claim: Claim, artifact: Artifact) -> dict:
    """The config the eval is launched with, keyed to match grade's faithfulness
    comparison (seqlen/stride/few_shot from the protocol; wbits/group_size from
    the artifact). Absent optional values are omitted, not invented."""
    ep = claim.eval_protocol
    cfg: dict = {}
    if ep.seqlen is not None:
        cfg["seqlen"] = ep.seqlen
    if ep.stride is not None:
        cfg["stride"] = ep.stride
    if ep.few_shot is not None:
        cfg["few_shot"] = ep.few_shot
    for k in ("wbits", "group_size"):
        if k in artifact.quant_config:
            cfg[k] = artifact.quant_config[k]
    return cfg
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_runexec.py -v`
Expected: PASS, 6 tests green

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/runexec.py tests/test_runexec.py
git commit -m "feat(run): eval command + seed extraction + actual_config resolution (pure logic)"
```

---

## Task 2: real I/O seams — _run_eval (env-activated) + _detect_gpu

**Files:**
- Modify: `src/paper_reprise/runexec.py`
- Test: `tests/test_runexec.py`

`_run_eval` runs the command in the setup-built env (PATH/VIRTUAL_ENV activation, same as 2b's `_run_smoke`), persisting combined stdout+stderr to `log_path` and returning `(exit_code, output)`. `_detect_gpu` is a best-effort label. We test `_run_eval` with a real harmless `echo` (instant, no GPU/network) to prove the wiring (activation env + log persistence) actually works; `_detect_gpu` is tested only for its safe fallback.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_runexec.py`:
```python
from pathlib import Path

from paper_reprise.runexec import _detect_gpu, _run_eval


def test_run_eval_persists_output_and_returns_exit_code(tmp_path):
    log = tmp_path / "stdout.log"
    env_dir = tmp_path / "env"
    (env_dir / "bin").mkdir(parents=True)
    code, out = _run_eval("echo hello-run", cwd=tmp_path, env_dir=env_dir, log_path=log)
    assert code == 0
    assert "hello-run" in out
    assert "hello-run" in log.read_text()   # persisted to disk for grade


def test_run_eval_nonzero_exit_is_captured(tmp_path):
    log = tmp_path / "stdout.log"
    env_dir = tmp_path / "env"
    (env_dir / "bin").mkdir(parents=True)
    code, out = _run_eval("exit 3", cwd=tmp_path, env_dir=env_dir, log_path=log)
    assert code == 3


def test_detect_gpu_returns_string_or_unknown(monkeypatch):
    # with no nvidia-smi and no CUDA_VISIBLE_DEVICES, falls back to "unknown"
    import paper_reprise.runexec as runexec
    monkeypatch.setattr(runexec.shutil, "which", lambda name: None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _detect_gpu() == "unknown"
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_runexec.py::test_run_eval_persists_output_and_returns_exit_code -v`
Expected: FAIL, `ImportError: cannot import name '_run_eval'`

- [ ] **Step 3: Write the implementation**

Add to the top imports of `src/paper_reprise/runexec.py`: `import os`, `import shutil`, `import subprocess`, `from pathlib import Path`. Add a module constant `_EVAL_TIMEOUT_S = 7200` and these functions:

```python
def _activated_env(env_dir: Path) -> dict:
    """Return an environment dict with env_dir's venv/conda prefix activated."""
    env = dict(os.environ)
    env["PATH"] = f"{env_dir / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = str(env_dir)
    return env


def _run_eval(command: str, cwd: Path, env_dir: Path, log_path: Path) -> tuple[int, str]:
    """Run the eval command in the built env, persist combined output to log_path,
    return (exit_code, output). A per-call timeout guards a hung eval."""
    try:
        proc = subprocess.run(command, shell=True, cwd=str(cwd), env=_activated_env(env_dir),
                              capture_output=True, text=True, timeout=_EVAL_TIMEOUT_S)
        out = proc.stdout + proc.stderr
        code = proc.returncode
    except subprocess.TimeoutExpired as e:
        out = f"eval timed out after {_EVAL_TIMEOUT_S}s\n{e}"
        code = 124
    log_path.write_text(out)
    return code, out


def _detect_gpu() -> str:
    """Best-effort GPU label: nvidia-smi name, else CUDA_VISIBLE_DEVICES, else 'unknown'."""
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30)
            name = proc.stdout.strip().splitlines()[0].strip() if proc.stdout.strip() else ""
            if name:
                return name
        except (subprocess.SubprocessError, OSError):
            pass
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return f"CUDA_VISIBLE_DEVICES={visible}"
    return "unknown"
```

Note: `Optional` (Task 1) and the new `os`/`shutil`/`subprocess`/`Path` are all used now; keep the import block ruff-clean.

- [ ] **Step 4: Run the tests, confirm they pass**

Run: `uv run pytest tests/test_runexec.py -v`
Expected: PASS, all runexec tests green (9 total). The `echo`/`exit` tests run a real (instant, harmless) shell; no GPU/network.

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/runexec.py tests/test_runexec.py
git commit -m "feat(run): env-activated eval subprocess + best-effort gpu detection"
```

---

## Task 3: make_run_executor — orchestration (success path, injected I/O)

**Files:**
- Modify: `src/paper_reprise/runexec.py`
- Test: `tests/test_runexec.py`

`make_run_executor` returns `executor(claim, artifact, claim_dir)`. The executor derives `root`/`env_dir`/`repo_dir` from `claim_dir`, builds the command, times the eval, persists `actual_config.json`, and returns the dict `run_claims` expects. I/O seams + clock are injected (real defaults from Task 2).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_runexec.py`:
```python
import json

from paper_reprise.rundir import RunDir
from paper_reprise.runexec import make_run_executor


def test_executor_runs_persists_and_returns_metadata(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    claim_dir = rd.claim_dir("c1")
    calls = {}

    def fake_run_eval(command, cwd, env_dir, log_path):
        calls["command"] = command
        calls["cwd"] = cwd
        calls["env_dir"] = env_dir
        Path(log_path).write_text("perplexity: 5.80")
        return 0, "perplexity: 5.80"

    executor = make_run_executor(
        run_eval=fake_run_eval, detect_gpu=lambda: "A100",
        now=iter([100.0, 220.0]).__next__,
    )
    out = executor(_claim("python eval.py --seed 42"), _artifact(), claim_dir)

    # derived paths
    assert calls["cwd"] == rd.repo_dir
    assert calls["env_dir"] == rd.root / "env"
    # returned metadata
    assert out["stdout_path"] == str(claim_dir / "stdout.log")
    assert out["gpu"] == "A100"
    assert out["seed"] == 42
    assert out["minutes"] == 2.0                      # (220-100)/60
    assert out["actual_config"]["wbits"] == 4
    # actual_config persisted for the report re-render
    saved = json.loads((claim_dir / "actual_config.json").read_text())
    assert saved["seqlen"] == 2048
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_runexec.py::test_executor_runs_persists_and_returns_metadata -v`
Expected: FAIL, `ImportError: cannot import name 'make_run_executor'`

- [ ] **Step 3: Write the implementation**

Add to the top imports of `runexec.py`: `import json`, `import time`, `from typing import Callable` (merge with the existing `Optional` import → `from typing import Callable, Optional`). Add:

```python
def _rundir_paths(claim_dir: Path) -> tuple[Path, Path, Path]:
    """Given rd.claim_dir(id) == rd.root/'runs'/id, derive (root, env_dir, repo_dir)."""
    root = claim_dir.parent.parent
    return root, root / "env", root / "repo"


def make_run_executor(
    *,
    run_eval: Callable[[str, Path, Path, Path], tuple[int, str]] | None = None,
    detect_gpu: Callable[[], str] | None = None,
    now: Callable[[], float] | None = None,
) -> Callable[[Claim, Artifact, Path], dict]:
    """Build the executor(claim, artifact, claim_dir) -> dict that run_claims injects.

    The executor runs the claim's eval command in the setup-built env, persists
    raw stdout + the resolved actual_config, and returns run metadata. A non-zero
    eval raises (run_claims turns that into a BLOCKED result — the eval did not
    successfully run, which is not the same as 'failed to reproduce')."""
    run_eval = run_eval or _run_eval
    detect_gpu = detect_gpu or _detect_gpu
    now = now or time.monotonic

    def executor(claim: Claim, artifact: Artifact, claim_dir: Path) -> dict:
        _root, env_dir, repo_dir = _rundir_paths(claim_dir)
        command = build_eval_command(claim)
        log_path = claim_dir / "stdout.log"
        gpu = detect_gpu()
        start = now()
        code, _out = run_eval(command, repo_dir, env_dir, log_path)
        minutes = (now() - start) / 60.0

        actual_config = resolve_actual_config(claim, artifact)
        (claim_dir / "actual_config.json").write_text(json.dumps(actual_config, indent=2))

        if code != 0:
            raise RuntimeError(f"eval exited {code}; see {log_path}")

        return {
            "stdout_path": str(log_path),
            "actual_config": actual_config,
            "gpu": gpu,
            "seed": extract_seed(command),
            "minutes": minutes,
        }

    return executor
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_runexec.py::test_executor_runs_persists_and_returns_metadata -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/runexec.py tests/test_runexec.py
git commit -m "feat(run): make_run_executor orchestration (run, persist actual_config, metadata)"
```

---

## Task 4: executor blocked path — non-zero eval surfaces via run_claims

**Files:**
- Test: `tests/test_runexec.py` (implementation complete from Task 3)

A non-zero eval must raise, and `run_claims` (unchanged) must turn that into a `status="blocked"` RunResult with the stdout still persisted. This task proves the executor integrates with the real `run_claims` correctly — no new production code.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_runexec.py`:
```python
from paper_reprise.models import Spec
from paper_reprise.runstage import run_claims


def _spec_one(command="python eval.py"):
    return Spec(paper="p", repo=None, artifacts=[_artifact()],
                claims=[_claim(command)])


def test_nonzero_eval_becomes_blocked_via_run_claims(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")

    def fail_eval(command, cwd, env_dir, log_path):
        Path(log_path).write_text("Traceback: CUDA out of memory")
        return 1, "Traceback: CUDA out of memory"

    executor = make_run_executor(run_eval=fail_eval, detect_gpu=lambda: "A100",
                                 now=iter([0.0, 60.0]).__next__)
    results, configs = run_claims(rd, _spec_one(), executor=executor)

    assert results[0].status == "blocked"
    assert "eval exited 1" in results[0].block_reason
    # stdout was still persisted (grade/report can show the traceback)
    assert (rd.claim_dir("c1") / "stdout.log").read_text().startswith("Traceback")
    # actual_config.json was written before the raise (for the report re-render)
    assert (rd.claim_dir("c1") / "actual_config.json").exists()


def test_successful_eval_is_ran_via_run_claims(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")

    def ok_eval(command, cwd, env_dir, log_path):
        Path(log_path).write_text("perplexity: 5.80")
        return 0, "perplexity: 5.80"

    executor = make_run_executor(run_eval=ok_eval, detect_gpu=lambda: "A100",
                                 now=iter([0.0, 30.0]).__next__)
    results, configs = run_claims(rd, _spec_one(), executor=executor)

    assert results[0].status == "ran"
    assert results[0].gpu == "A100"
    assert configs["c1"]["wbits"] == 4
```

- [ ] **Step 2: Run the tests, confirm they pass**

Run: `uv run pytest tests/test_runexec.py -v`
Expected: PASS, all runexec tests green. No new production code — Task 3's executor + the unchanged `run_claims` already implement this. If a test fails, the executor or path-derivation is wrong; fix `runexec.py`, do not change the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_runexec.py
git commit -m "test(run): non-zero eval becomes BLOCKED via run_claims; success is ran"
```

---

## Task 5: wire make_run_executor into the CLI + grade the re-render faithfully

**Files:**
- Modify: `src/paper_reprise/cli.py`
- Test: `tests/test_cli.py`

Replace the CLI's raising `run_executor` stub with `make_run_executor()`. Also fix the `report` command: it currently grades with `actual_config={}` (the Plan-1 TODO), forcing vacuous faithfulness on re-render. Now that the executor persists `claim_dir/actual_config.json`, read it back.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:
```python
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
```

- [ ] **Step 2: Run the tests, confirm they fail**

Run: `uv run pytest tests/test_cli.py::test_cli_run_passes_real_run_executor tests/test_cli.py::test_cli_report_reads_actual_config_for_faithfulness -v`
Expected: FAIL — `make_run_executor` not imported/used in cli.py; and the report command passes `actual_config={}` so it grades MATCH, not PARTIAL.

- [ ] **Step 3: Write the implementation**

(a) Add the import near the other `from paper_reprise...` imports in `cli.py`:
```python
from paper_reprise.runexec import make_run_executor
```

(b) In the `run` command, delete the stub:
```python
    def run_executor(claim, artifact, claim_dir):
        raise RuntimeError("real GPU executor not implemented (Plan 2c)")
```
and change the `run_pipeline(...)` call's `run_executor=run_executor` to `run_executor=make_run_executor()`. The call becomes:
```python
    result = run_pipeline(
        input_arg=input_arg, base_dir=Path(base_dir), timestamp=_timestamp(),
        available_hardware=[], approve_spec=approve_spec, approve_plan=approve_plan,
        fetch_sources=make_fetch_sources(), setup_executor=make_setup_executor(),
        run_executor=make_run_executor(),
    )
```

(c) Fix the `report` command. It currently builds each grade with `actual_config={}`. Replace the per-claim loop body so it reads `actual_config.json` when present. The current loop is:
```python
    artifacts = {a.id: a for a in spec.artifacts}
    from paper_reprise.models import RunResult
    grades, runs = [], []
    for c in spec.claims:
        log = rd.claim_dir(c.id) / "stdout.log"
        rr = RunResult(claim_id=c.id, command=c.eval_protocol.command,
                       stdout_path=str(log),
                       status="ran" if log.exists() else "blocked",
                       block_reason=None if log.exists() else "no stdout.log")
        runs.append(rr)
        # PLAN-2 TODO: actual_config={} forces the faithfulness check to pass vacuously
        # on re-render, so `report` can never detect a config divergence. Persist each
        # run's actual_config to the run dir (e.g. runs/<claim_id>/actual_config.json)
        # and read it back here once the real executor records it.
        grades.append(grade_claim(c, artifacts[c.artifact], rr, actual_config={}))
```
Replace it with (drop the TODO comment, read the json):
```python
    import json as _json

    artifacts = {a.id: a for a in spec.artifacts}
    from paper_reprise.models import RunResult
    grades, runs = [], []
    for c in spec.claims:
        cdir = rd.claim_dir(c.id)
        log = cdir / "stdout.log"
        rr = RunResult(claim_id=c.id, command=c.eval_protocol.command,
                       stdout_path=str(log),
                       status="ran" if log.exists() else "blocked",
                       block_reason=None if log.exists() else "no stdout.log")
        runs.append(rr)
        cfg_path = cdir / "actual_config.json"
        actual_config = _json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        grades.append(grade_claim(c, artifacts[c.artifact], rr, actual_config=actual_config))
```

- [ ] **Step 4: Run the tests, confirm they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS, all CLI tests green (existing + 2 new). The pre-existing `test_cli_report_rerenders` still passes (no `actual_config.json` in that test's run dir → falls back to `{}`, same as before).

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/cli.py tests/test_cli.py
git commit -m "feat(run): wire make_run_executor into CLI; report reads actual_config.json (closes Plan-1 TODO)"
```

---

## Task 6: full suite + ruff + offline guarantee + grade TODO cleanup

**Files:**
- Modify: `src/paper_reprise/grade.py` (remove the now-resolved PLAN-2 TODO comment, if any remains)
- Test: reuse existing

- [ ] **Step 1: Remove the stale grade TODO comment**

Check `src/paper_reprise/grade.py` for the `PLAN-2 TODO` comment about `actual_config` / missing-key faithfulness (around the `_faithfulness` function). Read it first:
Run: `grep -n "PLAN-2 TODO" src/paper_reprise/grade.py`
If it refers to actual_config persistence (now done in Plan 2c), update or remove it so it's not stale. Keep any part still genuinely deferred (e.g. "missing key = vacuously faithful" is still true and is the documented limitation — keep a one-line note pointing to the deferred output-parsing pass, but drop the "Persist ... once the real executor records it" part since that's now done). Use judgment; if the comment is purely about persistence that's now done, remove it.

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS, all tests green (Plan 1 + 2a + 2b + the new runexec/cli tests).

- [ ] **Step 3: Confirm tests are fast (no real GPU/subprocess leak)**

Run: `uv run pytest tests/test_runexec.py -q`
Expected: PASS in well under a second. The only real subprocess is the harmless `echo`/`exit` in Task 2; everything else injects fakes.

- [ ] **Step 4: ruff check**

Run: `uv run ruff check src/ tests/`
Expected: "All checks passed!" Confirm `os`, `shutil`, `subprocess`, `json`, `time`, `re`, `Callable`, `Optional`, `Path` are all used in `runexec.py` (no F401); `make_run_executor` imported+used in `cli.py`.

- [ ] **Step 5: Offline-guarantee grep**

Run: `grep -rn "subprocess.run\|nvidia-smi" src/paper_reprise/runexec.py`
Expected: subprocess calls appear ONLY inside `_run_eval` and `_detect_gpu` (the injectable seams) — never in the orchestration or pure logic.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore(run): drop resolved actual_config TODO; Plan 2c full suite green + ruff clean"
```

---

## Self-Review

**1. Spec coverage** (design §4.2 Run):

- "No agent after setup passes; execute per spec item by item" → `run_claims` (unchanged) loops claims; the executor runs one eval per claim, no agent. ✓
- "run eval per eval_protocol.command, persist raw stdout to runs/<claim_id>/" → `_run_eval` writes `claim_dir/stdout.log`; the executor returns that path (Task 2, 3). ✓
- "record actual command, seed, start/end time, GPU used" → command via build_eval_command; seed via extract_seed; minutes via injected clock; gpu via _detect_gpu (Tasks 1-3). ✓
- "Prefer the official reproduction command … quant_config degrades to a grading basis" → the executor runs `eval_protocol.command` (which specextract set, preferring the official command); quant_config is NOT used to assemble the command, only echoed into actual_config for grade's faithfulness basis (Task 1, 3). ✓
- "run does not parse results or grade, only persists raw output" → the executor never imports grade/report, never reads expected/tolerance, never parses metrics; it only runs + records (Tasks 1-3; §5.1 isolation respected). ✓
- "efficiency claims flagged with hardware" → gpu is captured per run into RunResult.gpu (the report already renders env/gpu); the claim's `hardware` field is a spec/plan concern handled upstream, untouched here. ✓

**Deferred (not omissions):**
- **actual_config from parsed eval output.** The executor reports the spec-resolved config (what it launched with), not values introspected from the black-box script's actual behavior. So on the official path grade's faithfulness check passes by construction; its real teeth this phase are `calib_status==UNKNOWN → BLOCKED` and the `setup_patches` trail in the report. A future pass can override specific actual_config keys with values parsed from the eval log to catch silent divergence — explicitly deferred.
- Separate quantization-step orchestration (the captured command handles quant+eval; assembling quant params ourselves is out of scope).
- From-scratch provider (design §6), GPU sandboxing (flagged in Plan 2b as its own future plan).

**2. Placeholder scan:** No TBD/TODO left in new code. The carried-over `run_executor` stub is REMOVED in Task 5. The stale grade `PLAN-2 TODO` about persistence is cleaned in Task 6 (the persistence half is now real; the "missing key = vacuously faithful" limitation is kept as a one-line documented note tied to the deferred parsing pass).

**3. Type consistency:**
- Executor contract `executor(claim, artifact, claim_dir) -> dict` with keys `stdout_path, actual_config, gpu, seed, minutes` — matches what `run_claims` reads (runstage.py:28-32). ✓
- `_run_eval(command, cwd, env_dir, log_path) -> (int, str)` — definition (Task 2), the executor's call site (Task 3), and all test fakes use the same 4-arg shape. ✓
- `resolve_actual_config(claim, artifact) -> dict` returns exactly the keys grade._faithfulness compares (seqlen/stride/few_shot/wbits/group_size). ✓
- `make_run_executor(*, run_eval=None, detect_gpu=None, now=None)` — None-default + `x = x or _default` call-time resolution, same pattern as 2b's run_setup_loop, so module-global monkeypatch and explicit injection both work. ✓
- `_rundir_paths(claim_dir)` derivation matches `RunDir.claim_dir` layout (`rd.root/"runs"/id`), so `root = claim_dir.parent.parent`. ✓
- CLI `report` reads `actual_config.json` and passes it to `grade_claim(c, artifact, rr, actual_config=...)` — matches grade's signature. ✓
