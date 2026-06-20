# Plan 2b: Setup Agentic Debug Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `setup` stage stub with the real agentic debug loop — the ONE genuinely agentic stage in the system. Build a conda/uv environment and let headless claude fix dependencies/APIs until the repo's own eval command passes a single smoke test (a machine-decidable exit condition), under a retry cap AND a total timeout, then freeze an env snapshot and surface the agent's patch trail. On exhausting the guardrails it returns `SetupResult(ok=False, error=...)` and hands off the full setup log — it never silently gives up.

**Architecture:** A new `setuploop.py` module holds the real loop; `setupstage.py` keeps its existing `run_setup(rd, spec, executor=None) -> SetupResult` seam and gains a `make_setup_executor(...)` factory that returns the `executor(rd, spec)` callable the pipeline already injects. Every real-world action — env creation (conda/uv subprocess), smoke-command execution (subprocess), pip-freeze/version capture (subprocess), and the headless "fix the env" claude call — is behind an injectable `_`-prefixed seam, exactly mirroring `fetch.py`. The pure/orchestration logic (smoke-test selection, loop control, retry+timeout bookkeeping, success detection, snapshot assembly, patch collection) is tested offline with fakes simulating "fails twice then succeeds", "never succeeds → hits cap", and "times out". Time is injected as a `now()` clock so the timeout is deterministic in tests with no real wall-clock.

**Tech Stack:** Python 3.11, stdlib `subprocess`/`json`/`pathlib`/`time`, pydantic (existing models), pytest (offline via injected fakes), uv, ruff (line-length 100). `from __future__ import annotations` in every module.

Design doc: `docs/superpowers/specs/2026-06-19-paper-reprise-agent-design.md` (§4.1 Setup — the only agentic stage; §2.2 isolation; §5 grade/report boundary)

---

## Context for the implementer

Plan 1 shipped a deterministic skeleton; Plan 2a made ingest really fetch (`paper/`, `repo/`). Plan 2b implements `setup`. Plan 2c (the GPU executor for the `run` stage) and the from-scratch provider are **out of scope** — do not touch `runstage.py`, `grade.py`, or `report.py`.

The seam you MUST preserve (do not change signatures):

- `setupstage.run_setup(rd, spec, executor=None) -> SetupResult` — the pipeline calls `run_setup(rd, spec, executor=setup_executor)` (pipeline.py:62). When `executor is None` it keeps the Plan 1 stub behaviour (a test relies on this: `tests/test_stages_stub.py::test_setup_stub_returns_env_snapshot`). When an executor is passed, `run_setup` delegates to it. Plan 2b's job is to build that executor.
- `SetupResult(ok: bool, env_snapshot: dict, patches: list[str], error: str)` — keep it exactly. `report.py` reads `setup.env_snapshot` (expects keys `torch`/`transformers`/`cuda`) and `setup.patches` (a list of strings). The pipeline aborts at `setup` when `not setup.ok` (pipeline.py:63-64).

Relevant existing API (do not change):

- `RunDir`: `rd.root`, `rd.repo_dir` (cloned repo from 2a), `rd.setup_log_dir`, `rd.setup_patches_dir` (all `mkdir`-ed by `RunDir.create`). Env snapshot is written to `rd.root / "env_snapshot.json"`.
- `Spec` / `Claim` / `EvalProtocol` (`models.py`): a claim has `eval_protocol.command` (str), `eval_protocol.runner`, `eval_protocol.dataset`, etc. The smoke fallback reads the first claim's `command`.
- `run_headless(prompt, allowed_tools, cwd, expect_file) -> HeadlessResult` (`headless.py`) is already injectable via `_call_claude` and is guarded by the autouse `_block_real_claude` fixture (`tests/conftest.py`). The setup loop's "fix" call uses headless claude, but the loop does NOT depend on an expected-file appearing the way specextract does — instead it asks the agent to write a patch note and then re-runs the smoke test itself to judge success. We therefore call a thin injectable `_run_fixer(...)` seam rather than `run_headless` directly, so the loop test can simulate the agent without any file-existence coupling. The real `_run_fixer` is a tiny wrapper over `run_headless`.

**Setup/run separation (design §4.1 + §2.2):** setup only makes the env *runnable* (smoke test = tiny scale, e.g. 8 samples / 1 batch). It must NOT run real experiments or compute real numbers, and it must NOT read or write grade/report artifacts. Keep this module ignorant of verdicts, tolerances, and expected values.

This plan does NOT touch the GPU executor (Plan 2c) or ingest fetch (Plan 2a, done).

---

## File Structure

```
src/paper_reprise/
  setuploop.py     # NEW — the real agentic debug loop + injectable I/O seams.
  setupstage.py    # MODIFY — keep run_setup/SetupResult seam; add make_setup_executor() factory.
  cli.py           # MODIFY — pass make_setup_executor() instead of setup_executor=None.
tests/
  test_setuploop.py   # NEW — offline tests of the loop via injected fakes (env/smoke/freeze/fixer/clock).
  test_setupstage.py  # NEW — make_setup_executor wiring + the existing stub-path assertion.
```

**Responsibility split inside `setuploop.py`:**

- Low-level injectable I/O (the only functions that touch the outside world):
  - `_create_env(env_dir: Path, manager: str) -> tuple[int, str]` — conda/uv subprocess; returns `(exit_code, log)`.
  - `_run_smoke(command: str, cwd: Path, env_dir: Path) -> tuple[int, str]` — runs the smoke command; returns `(exit_code, combined_output)`.
  - `_freeze_env(env_dir: Path) -> dict` — pip freeze + torch/transformers/CUDA versions.
  - `_run_fixer(prompt: str, cwd: Path, patch_note: Path) -> None` — one headless claude "fix the env" turn (thin wrapper over `run_headless`).
- Pure logic (offline-testable directly, no I/O):
  - `select_smoke_command(rd, spec) -> str` — repo example/test if present, else first claim command shrunk to tiny scale.
  - `shrink_command(command: str) -> str` — append tiny-scale flags (n_samples=8, batch=1) deterministically.
  - `assemble_snapshot(freeze: dict) -> dict` — normalize a freeze dict into the report's expected keys.
  - `collect_new_patches(patches_dir: Path, seen: set[str]) -> list[str]` — diff the patches dir to find notes the agent just wrote.
- Orchestration (testable by injecting fakes for the four I/O seams + a `now` clock):
  - `run_setup_loop(rd, spec, *, manager, max_retries, timeout_s, now, create_env, run_smoke, freeze_env, run_fixer) -> SetupResult`
- Factory (in `setupstage.py`):
  - `make_setup_executor(*, manager="uv", max_retries=6, timeout_s=3600.0) -> Callable[[RunDir, Spec], SetupResult]`

**How the hard parts are solved (read before coding):**

1. **Smoke-test success is detected deterministically by exit code.** `_run_smoke` returns `(exit_code, output)`. `exit_code == 0` ⇒ pass. No log parsing, no LLM judgement — machine-decidable, per §4.1. The loop, not the agent, decides success by re-running the smoke command itself.
2. **The loop knows what the agent changed via the patch trail.** Each fixer turn is told to append a one-line note to a uniquely-named file in `rd.setup_patches_dir` (`patch_<n>.txt`). After every fixer turn the loop calls `collect_new_patches` (a set-diff of filenames) to capture exactly the notes added this turn into `SetupResult.patches`. No reliance on the agent returning structured data.
3. **The timeout is enforced with an injected clock, not real wall-clock.** `run_setup_loop` takes `now: Callable[[], float]` (defaults to `time.monotonic`). Before each smoke attempt it checks `now() - start >= timeout_s`. Tests inject a fake clock whose successive calls return increasing values, so "times out after N steps" is exact and instant. The retry cap (`max_retries`) bounds the count of fix→retry iterations independently. Either guardrail exceeded ⇒ `ok=False` with the log handed off — never a silent give-up.

---

## Task 1: smoke-command selection + shrink (pure logic)

**Files:**
- Create: `src/paper_reprise/setuploop.py`
- Test: `tests/test_setuploop.py`

`select_smoke_command` prefers the repo's own runnable example/test; if none, it falls back to the first claim's eval command shrunk to tiny scale. `shrink_command` deterministically appends tiny-scale flags so the smoke run is cheap (design §4.1: "8 samples, 1 batch").

- [ ] **Step 1: Write the failing test**

`tests/test_setuploop.py`:
```python
from pathlib import Path

from paper_reprise.models import Artifact, Claim, EvalProtocol, Spec
from paper_reprise.rundir import RunDir
from paper_reprise.setuploop import select_smoke_command, shrink_command


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
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_setuploop.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.setuploop'`

- [ ] **Step 3: Write the implementation**

`src/paper_reprise/setuploop.py`:
```python
"""Setup stage: the agentic env-debug loop (the only agentic stage, design §4.1).

Goal (single, machine-decidable): fix a conda/uv env until the repo's own eval
command passes a smoke test ONCE, under a retry cap AND a total timeout. On
exhausting the guardrails, return ok=False and hand off the full setup log — we
never silently give up. Setup only makes the env runnable; it never runs real
experiments or computes real numbers.

Every real-world action (env creation, smoke run, pip freeze, the headless "fix"
call) is behind an injectable seam so the whole loop is offline-testable.
"""
from __future__ import annotations

from pathlib import Path

from paper_reprise.models import Spec
from paper_reprise.rundir import RunDir

# Tiny-scale flags so the smoke run is cheap (design §4.1: ~8 samples, 1 batch).
_TINY_FLAGS = "--limit 8 --batch-size 1"

# Repo files we treat as a ready-made smoke entry, in priority order.
_EXAMPLE_CANDIDATES = (
    "examples/smoke.sh",
    "examples/example.sh",
    "examples/run.sh",
    "scripts/smoke.sh",
)


def shrink_command(command: str) -> str:
    """Append tiny-scale flags to a full eval command for the smoke run."""
    return f"{command} {_TINY_FLAGS}"


def select_smoke_command(rd: RunDir, spec: Spec) -> str:
    """Pick the smoke command: repo's own example/test if present, else the first
    claim's eval command shrunk to tiny scale."""
    for rel in _EXAMPLE_CANDIDATES:
        if (rd.repo_dir / rel).is_file():
            return f"bash {rel}"
    if spec.claims:
        return shrink_command(spec.claims[0].eval_protocol.command)
    return ""
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_setuploop.py -v`
Expected: PASS, 3 tests green

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/setuploop.py tests/test_setuploop.py
git commit -m "feat(setup): smoke-command selection + tiny-scale shrink (pure logic)"
```

---

## Task 2: env snapshot assembly + patch-trail diff (pure logic)

**Files:**
- Modify: `src/paper_reprise/setuploop.py`
- Test: `tests/test_setuploop.py`

`assemble_snapshot` normalizes a raw freeze dict into the keys the report expects (`torch`/`transformers`/`cuda`) while preserving the full `pip_freeze`. `collect_new_patches` set-diffs the patches dir so the loop captures exactly the notes the agent wrote this turn.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_setuploop.py`:
```python
from paper_reprise.setuploop import assemble_snapshot, collect_new_patches


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


def test_assemble_snapshot_fills_unknown_for_missing():
    snap = assemble_snapshot({"pip_freeze": ""})
    assert snap["torch"] == "unknown"
    assert snap["transformers"] == "unknown"
    assert snap["cuda"] == "unknown"


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
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_setuploop.py::test_collect_new_patches_returns_only_unseen -v`
Expected: FAIL, `ImportError: cannot import name 'assemble_snapshot'`

- [ ] **Step 3: Write the implementation**

Add to `src/paper_reprise/setuploop.py`:
```python
def assemble_snapshot(freeze: dict) -> dict:
    """Normalize a raw freeze dict into the report's expected keys, keeping the
    full pip freeze. Missing versions become 'unknown' (never silently dropped)."""
    return {
        "torch": freeze.get("torch") or "unknown",
        "transformers": freeze.get("transformers") or "unknown",
        "cuda": freeze.get("cuda") or "unknown",
        "pip_freeze": freeze.get("pip_freeze", ""),
    }


def collect_new_patches(patches_dir: Path, seen: set[str]) -> list[str]:
    """Return the contents of patch-note files in patches_dir not yet in `seen`,
    sorted by filename, and record them in `seen`. This is how the loop learns
    what the agent changed each turn."""
    new: list[str] = []
    for p in sorted(patches_dir.glob("patch_*.txt")):
        if p.name in seen:
            continue
        seen.add(p.name)
        new.append(p.read_text().strip())
    return new
```

- [ ] **Step 4: Run the tests, confirm they pass**

Run: `uv run pytest tests/test_setuploop.py -v`
Expected: PASS, all setuploop tests green (6 total now)

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/setuploop.py tests/test_setuploop.py
git commit -m "feat(setup): env snapshot normalization + patch-trail set-diff (pure logic)"
```

---

## Task 3: the fixer prompt builder (pure logic)

**Files:**
- Modify: `src/paper_reprise/setuploop.py`
- Test: `tests/test_setuploop.py`

`build_fixer_prompt` produces the instruction handed to headless claude on a failed smoke run: the failing command, the captured traceback, and the contract to (a) fix the env/repo minimally and (b) record each change as a one-line note in a uniquely-named patch file. Keeping it a pure string builder makes it assertable offline.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_setuploop.py`:
```python
from paper_reprise.setuploop import build_fixer_prompt


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
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_setuploop.py::test_build_fixer_prompt_includes_command_traceback_and_patch_contract -v`
Expected: FAIL, `ImportError: cannot import name 'build_fixer_prompt'`

- [ ] **Step 3: Write the implementation**

Add to `src/paper_reprise/setuploop.py`:
```python
_FIXER_TEMPLATE = """The repo's smoke command failed. Fix the conda/uv environment \
and, only if necessary, the repo code, so the SAME command can run. This is a SMOKE \
TEST only — do NOT run real experiments, do NOT change eval parameters that affect \
numbers (seqlen, calib, dataset, sample count beyond the tiny smoke scale).

Failing command:
    {command}

Captured output (read the traceback):
{output}

For EACH change you make (a pinned version, an added package, a patched API line), \
append one line describing it to `{patch_note}` (create the file; one line per change). \
Do not run the command yourself — the loop will re-run it to check. Report 'fixed' when done."""


def build_fixer_prompt(command: str, output: str, patch_note_path: str) -> str:
    """Build the headless-claude instruction for one fix turn on a failed smoke run."""
    return _FIXER_TEMPLATE.format(
        command=command, output=output, patch_note=patch_note_path
    )
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_setuploop.py -v`
Expected: PASS, all setuploop tests green (7 total now)

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/setuploop.py tests/test_setuploop.py
git commit -m "feat(setup): fixer prompt builder (command + traceback + patch contract)"
```

---

## Task 4: the loop — success on first smoke pass (orchestration, all I/O injected)

**Files:**
- Modify: `src/paper_reprise/setuploop.py`
- Test: `tests/test_setuploop.py`

`run_setup_loop` is the orchestrator. This task covers the happy path: create env → smoke passes immediately → freeze snapshot → write `env_snapshot.json` → `SetupResult(ok=True)`. Every I/O seam is a keyword-only injected callable; the real defaults arrive in Task 7. Time is injected via `now`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_setuploop.py`:
```python
import json

from paper_reprise.setupstage import SetupResult
from paper_reprise.setuploop import run_setup_loop


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
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_setuploop.py::test_loop_success_on_first_smoke_pass -v`
Expected: FAIL, `ImportError: cannot import name 'run_setup_loop'`

- [ ] **Step 3: Write the implementation**

Add to `src/paper_reprise/setuploop.py` (add these imports at the top: `import json`, `import time`, `from typing import Callable`, and `from paper_reprise.setupstage import SetupResult`):

> Note on import direction: `setuploop` imports `SetupResult` from `setupstage`, and `setupstage` imports `make_setup_executor`-related helpers from `setuploop`. To avoid a circular import at module load, `setupstage.make_setup_executor` imports `run_setup_loop` *inside the factory body* (Task 6), not at module top. `setuploop` importing the plain dataclass `SetupResult` at top level is fine because `setupstage` does not import `setuploop` at top level.

```python
import json
import time
from typing import Callable

from paper_reprise.setupstage import SetupResult


def _write_log(rd: RunDir, name: str, text: str) -> None:
    (rd.setup_log_dir / name).write_text(text)


def run_setup_loop(
    rd: RunDir,
    spec: Spec,
    *,
    manager: str = "uv",
    max_retries: int = 6,
    timeout_s: float = 3600.0,
    now: Callable[[], float] = time.monotonic,
    create_env: Callable[[Path, str], tuple[int, str]],
    run_smoke: Callable[[str, Path, Path], tuple[int, str]],
    freeze_env: Callable[[Path], dict],
    run_fixer: Callable[[str, Path, Path], None],
) -> SetupResult:
    """Drive the env-debug loop until the smoke command passes once, or a guardrail
    (retry cap / timeout) is exceeded. On exhaustion: ok=False, full log handed off."""
    env_dir = rd.root / "env"
    command = select_smoke_command(rd, spec)
    start = now()
    seen_patches: set[str] = set()
    patches: list[str] = []

    # --- build env once ---
    code, env_log = create_env(env_dir, manager)
    _write_log(rd, "create_env.log", env_log)
    if code != 0:
        return SetupResult(ok=False, patches=patches,
                           error=f"env creation failed (exit {code}); see setup_log/")

    # --- smoke → fix → retry loop ---
    attempt = 0
    while True:
        if now() - start >= timeout_s:
            return SetupResult(ok=False, patches=patches,
                               error=f"setup timed out after {timeout_s}s "
                                     f"({attempt} attempts); see setup_log/")
        code, out = run_smoke(command, rd.repo_dir, env_dir)
        _write_log(rd, f"smoke_{attempt}.log", out)
        if code == 0:
            snapshot = assemble_snapshot(freeze_env(env_dir))
            (rd.root / "env_snapshot.json").write_text(json.dumps(snapshot, indent=2))
            return SetupResult(ok=True, env_snapshot=snapshot, patches=patches)
        if attempt >= max_retries:
            return SetupResult(ok=False, patches=patches,
                               error=f"smoke test still failing after {max_retries} "
                                     f"retries; see setup_log/")
        # hand the traceback to the agent; record what it changed
        note_rel = f"setup_patches/patch_{attempt}.txt"
        prompt = build_fixer_prompt(command, out, note_rel)
        run_fixer(prompt, rd.root, rd.setup_patches_dir / f"patch_{attempt}.txt")
        patches.extend(collect_new_patches(rd.setup_patches_dir, seen_patches))
        attempt += 1
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_setuploop.py::test_loop_success_on_first_smoke_pass -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/setuploop.py tests/test_setuploop.py
git commit -m "feat(setup): loop orchestrator happy path (env→smoke pass→freeze snapshot)"
```

---

## Task 5: the loop — fails-twice-then-succeeds + records the patch trail

**Files:**
- Test: `tests/test_setuploop.py` (implementation already complete from Task 4)

This task adds no new production code — it proves the fix→retry path and the patch-trail capture work. The fake `run_smoke` fails twice then succeeds; the fake `run_fixer` writes a patch note each turn (simulating the agent). We assert the loop retries, captures both patch notes in order, and ends `ok=True`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_setuploop.py`:
```python
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
```

- [ ] **Step 2: Run the test, confirm it passes**

Run: `uv run pytest tests/test_setuploop.py::test_loop_fails_twice_then_succeeds_and_records_patches -v`
Expected: PASS (no new code needed — Task 4's loop already implements this). If it fails, the loop logic is wrong; fix `run_setup_loop`, do not change the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_setuploop.py
git commit -m "test(setup): fails-twice-then-succeeds path records ordered patch trail"
```

---

## Task 6: the loop — retry cap and timeout both surface ok=False (never silent)

**Files:**
- Test: `tests/test_setuploop.py` (implementation already complete from Task 4)

Both guardrails must end the loop with `ok=False`, a descriptive `error`, the partial patch trail preserved, and the full log handed off — per the standing rule "never silently skip/cap expensive work". Two tests: smoke never passes → hits the retry cap; and `now` advances past `timeout_s` → times out mid-loop.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_setuploop.py`:
```python
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


def test_loop_env_creation_failure_is_surfaced(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    f = _fakes()
    f.update(create_env=lambda env_dir, manager: (1, "conda not found"))
    res = run_setup_loop(rd, _spec(), manager="conda", max_retries=3, timeout_s=100.0, **f)

    assert res.ok is False
    assert "env creation failed" in res.error
    assert (rd.setup_log_dir / "create_env.log").read_text() == "conda not found"
```

- [ ] **Step 2: Run the tests, confirm they pass**

Run: `uv run pytest tests/test_setuploop.py -v`
Expected: PASS, all setuploop tests green. Note the `now` for the timeout test: `start=0.0`, first loop check `5.0 < 100` (proceeds, smoke fails, no fixer-clock use because `run_fixer` does not call `now`), second check `999.0 >= 100` → times out. If the timeout test fails, verify the loop checks `now()` once at the top of each iteration only.

- [ ] **Step 3: Commit**

```bash
git add tests/test_setuploop.py
git commit -m "test(setup): retry cap + timeout + env-create failure all surface ok=False with log"
```

---

## Task 7: real I/O seams + make_setup_executor factory

**Files:**
- Modify: `src/paper_reprise/setuploop.py` (real `_create_env`/`_run_smoke`/`_freeze_env`/`_run_fixer`)
- Modify: `src/paper_reprise/setupstage.py` (add `make_setup_executor`)
- Test: `tests/test_setupstage.py`

Now wire the real subprocess/headless implementations as the injectable defaults, and add the `make_setup_executor` factory that the pipeline/CLI will use. The factory returns `executor(rd, spec)` matching the existing `run_setup(rd, spec, executor=...)` seam.

- [ ] **Step 1: Write the failing test**

`tests/test_setupstage.py`:
```python
from pathlib import Path

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
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_setupstage.py -v`
Expected: FAIL, `ImportError: cannot import name 'make_setup_executor'`

- [ ] **Step 3: Write the implementation**

Add the real I/O seams to `src/paper_reprise/setuploop.py` (add `import os` and `import subprocess` to the top imports, and `from paper_reprise.headless import run_headless`):
```python
import os
import subprocess

from paper_reprise.headless import run_headless

_SMOKE_TIMEOUT_S = 1800   # per-attempt cap so one hung smoke run can't block forever


def _create_env(env_dir: Path, manager: str) -> tuple[int, str]:
    """Create a conda/uv env at env_dir. Returns (exit_code, combined log)."""
    if manager == "uv":
        cmd = ["uv", "venv", str(env_dir)]
    else:
        cmd = ["conda", "create", "-y", "-p", str(env_dir), "python=3.11"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr)


def _run_smoke(command: str, cwd: Path, env_dir: Path) -> tuple[int, str]:
    """Run the smoke command inside the repo, USING the created env.

    The created venv/conda env is activated by prepending its bin/ to PATH and
    setting VIRTUAL_ENV, so `python`/`pip` in the command resolve to env_dir's
    interpreter (not the ambient one — otherwise building the env is pointless).
    Returns (exit_code, combined output).
    """
    env = dict(os.environ)
    env["PATH"] = f"{env_dir / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = str(env_dir)
    try:
        proc = subprocess.run(command, shell=True, cwd=str(cwd), env=env,
                              capture_output=True, text=True, timeout=_SMOKE_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        return 124, f"smoke command timed out after {_SMOKE_TIMEOUT_S}s\n{e}"
    return proc.returncode, (proc.stdout + proc.stderr)


def _freeze_env(env_dir: Path) -> dict:
    """Capture pip freeze + torch/transformers/CUDA versions from env_dir."""
    proc = subprocess.run([str(env_dir / "bin" / "python"), "-m", "pip", "freeze"],
                          capture_output=True, text=True)
    freeze_text = proc.stdout
    versions: dict = {"pip_freeze": freeze_text}
    for line in freeze_text.splitlines():
        for pkg in ("torch", "transformers"):
            if line.lower().startswith(f"{pkg}=="):
                versions[pkg] = line.split("==", 1)[1].strip()
    cuda = subprocess.run(
        [str(env_dir / "bin" / "python"), "-c",
         "import torch; print(torch.version.cuda)"],
        capture_output=True, text=True)
    if cuda.returncode == 0 and cuda.stdout.strip() not in ("", "None"):
        versions["cuda"] = cuda.stdout.strip()
    return versions


def _run_fixer(prompt: str, cwd: Path, patch_note: Path) -> None:
    """One headless-claude fix turn. Success is judged by the loop re-running the
    smoke command, so we ignore the HeadlessResult here (the patch note is the
    only artifact we read back, via collect_new_patches)."""
    run_headless(prompt=prompt, allowed_tools=["Read", "Write", "Edit", "Bash"],
                 cwd=cwd, expect_file=patch_note)
```

Then change `run_setup_loop`'s defaults to reference these real seams. Replace the four seam parameters' (now-absent) defaults so they default to the module functions:
```python
def run_setup_loop(
    rd: RunDir,
    spec: Spec,
    *,
    manager: str = "uv",
    max_retries: int = 6,
    timeout_s: float = 3600.0,
    now: Callable[[], float] = time.monotonic,
    create_env: Callable[[Path, str], tuple[int, str]] = _create_env,
    run_smoke: Callable[[str, Path, Path], tuple[int, str]] = _run_smoke,
    freeze_env: Callable[[Path], dict] = _freeze_env,
    run_fixer: Callable[[str, Path, Path], None] = _run_fixer,
) -> SetupResult:
```

> The four `_`-prefixed functions must be defined ABOVE `run_setup_loop` in the file so they can be referenced as defaults. Move `run_setup_loop` to the bottom of the module. The Task 4–6 loop body is unchanged.

Now add the factory to `src/paper_reprise/setupstage.py`:
```python
def make_setup_executor(*, manager: str = "uv", max_retries: int = 6,
                        timeout_s: float = 3600.0) -> Callable[[RunDir, Spec], SetupResult]:
    """Build the executor(rd, spec) the pipeline injects into run_setup. Imported
    lazily to avoid a circular import (setuploop imports SetupResult from here)."""
    from paper_reprise.setuploop import run_setup_loop

    def executor(rd: RunDir, spec: Spec) -> SetupResult:
        return run_setup_loop(rd, spec, manager=manager,
                              max_retries=max_retries, timeout_s=timeout_s)

    return executor
```

- [ ] **Step 4: Run the tests, confirm they pass**

Run: `uv run pytest tests/test_setupstage.py tests/test_setuploop.py -v`
Expected: PASS, all setup tests green. The autouse `_block_real_claude` fixture guarantees `_run_fixer`'s real path is never hit unstubbed; `test_make_setup_executor_runs_loop_with_injected_io` stubs all four seams.

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/setuploop.py src/paper_reprise/setupstage.py tests/test_setupstage.py
git commit -m "feat(setup): real conda/uv + smoke + freeze + fixer seams; make_setup_executor factory"
```

---

## Task 8: wire make_setup_executor into the CLI

**Files:**
- Modify: `src/paper_reprise/cli.py`
- Test: `tests/test_cli.py`

The CLI currently passes `setup_executor=None` (the stub). Switch it to `make_setup_executor()`. Because the run aborts at `specextract` in the existing CLI tests (no real spec), setup is not actually reached there — so the wiring change is asserted by checking the executor is constructed and passed, via monkeypatch.

- [ ] **Step 1: Read the current CLI run command**

Run: `uv run python -c "import paper_reprise.cli, inspect; print(inspect.getsource(paper_reprise.cli.run))"`
Expected: shows `setup_executor=None` in the `run_pipeline(...)` call. Confirm the exact surrounding lines before editing.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_cli.py`:
```python
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
```

- [ ] **Step 3: Run the test, confirm it fails**

Run: `uv run pytest tests/test_cli.py::test_cli_run_passes_real_setup_executor -v`
Expected: FAIL — `setup_executor` is currently `None`, not the sentinel (and `make_setup_executor` is not imported in cli.py).

- [ ] **Step 4: Write the implementation**

In `src/paper_reprise/cli.py`, add to the imports near the existing `from paper_reprise.fetch import ...` line:
```python
from paper_reprise.setupstage import make_setup_executor
```

In the `run` command's `run_pipeline(...)` call, replace:
```python
        fetch_sources=make_fetch_sources(), setup_executor=None, run_executor=run_executor,
```
with:
```python
        fetch_sources=make_fetch_sources(), setup_executor=make_setup_executor(),
        run_executor=run_executor,
```

> If `run_pipeline` is imported locally inside `run` (Plan 2a did `from paper_reprise.pipeline import run_pipeline` inside the function), the monkeypatch on `paper_reprise.pipeline.run_pipeline` still works because the local import resolves the attribute at call time. Keep the import where it is.

- [ ] **Step 5: Run the tests, confirm they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS, all CLI tests green (pre-existing ones + the new one).

- [ ] **Step 6: Commit**

```bash
git add src/paper_reprise/cli.py tests/test_cli.py
git commit -m "feat(setup): wire make_setup_executor into CLI run (replaces stub)"
```

---

## Task 9: full suite + ruff + offline guarantee

**Files:** reuse existing

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS, all tests green (Plan 1 + 2a + the new setuploop/setupstage/cli tests). No test invokes real conda/uv/git/claude — every seam is injected or monkeypatched, and the autouse `_block_real_claude` fixture backstops the headless path.

- [ ] **Step 2: Confirm tests are fast (no accidental real subprocess/network)**

Run: `uv run pytest tests/test_setuploop.py tests/test_setupstage.py -q`
Expected: PASS in well under a second. A slow run means a real subprocess leaked — find the test that forgot to inject a fake.

- [ ] **Step 3: ruff check**

Run: `uv run ruff check src/ tests/`
Expected: "All checks passed!" Fix any F401 (unused imports) — in particular confirm `os`, `time`, `json`, `subprocess`, `Callable`, `run_headless` are all actually used in `setuploop.py`, and `make_setup_executor` is imported and used in `cli.py`.

- [ ] **Step 4: Commit (if ruff made changes)**

```bash
git add -A
git commit -m "chore: ruff clean for Plan 2b setup loop"
```

---

## Self-Review

**1. Spec coverage** (design §4.1 Setup):

- "Single decidable goal: fix the env until the repo's own eval command runs successfully ONCE (smoke test)" → `run_setup_loop` exits `ok=True` the first time `run_smoke` returns exit code 0 (Task 4). Machine-decidable by exit code, no parsing. ✓
- "Smoke-test input: (a) repo's own example/test; (b) fall back to a spec claim command shrunk to tiny scale" → `select_smoke_command` checks `_EXAMPLE_CANDIDATES` first, else `shrink_command(first claim command)` (Task 1). ✓
- "Loop: create env → install deps → smoke-run → on failure agent reads traceback and patches → retry → on success freeze snapshot → exit" → Task 4 loop; the fixer prompt carries the captured traceback (Task 3). Note: dependency *install* is folded into the agent's fix turns (the agent runs pip/uv add as part of fixing), and the first smoke run after `create_env` surfaces the missing-deps traceback that drives the first fix — see ambiguity note below. ✓
- "Guardrails: retry cap AND total timeout; on exceeding, do NOT silently give up — return ok=False with the full setup log handed off" → both guardrails return `ok=False` with a descriptive error citing `setup_log/`, partial patches preserved, logs written to `rd.setup_log_dir` (Task 6). Matches the standing "never silently skip/cap" rule. ✓
- "Env snapshot: on success write env_snapshot.json with pip freeze + CUDA/torch/transformers" → `_freeze_env` captures all four; `assemble_snapshot` normalizes; written to `rd.root / env_snapshot.json` only on success (Tasks 2, 4, 7). ✓
- "Patch trail: record each change into setup_patches/, surface in SetupResult.patches" → fixer writes `patch_<n>.txt` notes; `collect_new_patches` diffs them into `SetupResult.patches` (Tasks 2, 3, 4). ✓
- "setup/run separation: setup only makes the env runnable; must NOT run real experiments" → smoke command is tiny-scale (`--limit 8 --batch-size 1`); fixer prompt explicitly forbids changing number-affecting params or running real experiments; module never imports grade/report or reads expected/tolerance (§2.2/§5 boundary respected). ✓

**Deferred to later (not omissions):**
- GPU executor for the `run` stage (Plan 2c) — `cli.py` still raises `"real GPU executor not implemented (Plan 2c)"` in `run_executor`.
- From-scratch provider (interface only, design §6).
- Reuse of `env_snapshot`/`setup_patches` across runs (design §5.3 accumulation) — directory-by-convention, no code this phase.
- Smarter example/test discovery (parsing repo README for the actual command, Makefile targets, `pytest` discovery) — `_EXAMPLE_CANDIDATES` covers the common `examples/*.sh` cases; richer discovery is a follow-up, not blocking.

**2. Placeholder scan:** No TBD/TODO. Every code step has complete runnable code. The only "not implemented" is the carried-over `run_executor` raise (correctly labeled Plan 2c, untouched by this plan).

**3. Type consistency** (signatures used in later tasks match earlier definitions):
- `SetupResult(ok, env_snapshot, patches, error)` — unchanged from `setupstage.py`; `run_setup_loop` constructs it with these exact kwargs everywhere (Tasks 4, 6). ✓
- Seam signatures are consistent across definition, defaults, and test fakes:
  - `create_env: (env_dir: Path, manager: str) -> (int, str)` — `_create_env`, the fake in `_fakes()`, and the `monkeypatch` in test_setupstage all use `(env_dir, manager)`. ✓
  - `run_smoke: (command: str, cwd: Path, env_dir: Path) -> (int, str)` — `_run_smoke` and all fakes use `(command, cwd, env_dir)` / `(c, cwd, e)`. ✓
  - `freeze_env: (env_dir: Path) -> dict` — returns a dict with `torch/transformers/cuda/pip_freeze`; `assemble_snapshot` reads exactly those keys. ✓
  - `run_fixer: (prompt: str, cwd: Path, patch_note: Path) -> None` — `_run_fixer`, the loop call site (`run_fixer(prompt, rd.root, rd.setup_patches_dir / f"patch_{attempt}.txt")`), and the fakes all use `(prompt, cwd, patch_note)`. The loop passes `rd.root` as cwd and the absolute patch-note path; the prompt embeds the *relative* `setup_patches/patch_<n>.txt` (Task 3) so the agent writes to the right place from cwd=`rd.root`. ✓
  - `now: () -> float` — default `time.monotonic`; tests inject `iter([...]).__next__`. The loop calls `now()` once at the top of each iteration plus once for `start`; the timeout test's clock sequence (`0.0, 5.0, 999.0`) is sized for exactly that call pattern. ✓
- `run_setup_loop`'s I/O defaults reference the `_`-prefixed module functions, which must be defined above it (Task 7 moves the orchestrator to the bottom). ✓
- `make_setup_executor(*, manager, max_retries, timeout_s) -> Callable[[RunDir, Spec], SetupResult]` — returns `executor(rd, spec)` matching `run_setup(rd, spec, executor=...)` (pipeline.py:62) and the CLI passes it as `setup_executor=` (Task 8). ✓
- Circular import: `setuploop` imports `SetupResult` from `setupstage` at top level; `setupstage.make_setup_executor` imports `run_setup_loop` lazily inside the factory body — no import cycle at module load. ✓

**4. Ambiguities in §4.1 and decisions made:**
- *"install deps" as a distinct loop step.* §4.1 lists "install deps" between "build env" and "smoke-run", but `create_env` (uv venv / conda create) produces a bare interpreter, and the repo's deps are exactly what's rotted. Decision: do NOT bake a fixed `pip install -r requirements.txt` step into the loop — that would hardcode an assumption many rotted repos break on. Instead the first `run_smoke` fails with the missing-dep traceback, which the agent resolves (pip/uv install, pinning) inside its fix turn. This keeps the loop a single uniform smoke→fix→retry cycle and lets the agent handle whatever install mechanism the repo actually uses (requirements.txt / pyproject / setup.py / conda env.yml). Recorded as a fix in the patch trail like any other change.
- *Per-attempt vs total timeout.* §4.1 says "total timeout"; a single hung smoke run could still block forever inside one `run_smoke` call before the loop-level `now()` check fires. Decision: keep the loop-level total timeout (the spec's requirement, injectable for tests) AND add a real per-attempt `subprocess` timeout (`_SMOKE_TIMEOUT_S`) inside `_run_smoke` so a hung process is killed and surfaced as exit 124, which the loop treats as a normal failure. The per-attempt cap is not exercised by offline tests (the fake `run_smoke` returns instantly) and does not affect loop logic.
- *What counts as the repo's "own example/test".* §4.1 says "the repo's own example/test if available" without defining detection. Decision: a small priority list of conventional shell entry points (`examples/smoke.sh`, etc.). This is deliberately conservative — when nothing matches we fall back to the shrunk claim command, which is always available. Richer discovery (README parsing, Makefile, pytest) is deferred (noted above) rather than guessed at.
- *Snapshot on failure.* §4.1 says write the snapshot "on success". Decision: `env_snapshot.json` is written only on `ok=True`; on failure we hand off `setup_log/` instead (asserted in Task 6). This keeps "the env that actually ran" the only thing the report's env line ever reflects.

**5. Known risks carried by this design (flagged, not resolved here):**
- *Untrusted code execution / no sandbox.* This is the stage where the cloned repo's code and the agent's `Bash` fix turns actually execute on the user's machine — the exact RCE surface the Plan 2a security review flagged as "out of scope for ingest, must be sandboxed downstream". Plan 2b runs `_run_smoke` (shell=True on a repo-derived command) and `_run_fixer` (agent with Bash) directly on the host with no container/VM/user isolation. For a single-user local research tool against arxiv papers this matches the design's manual-CLI assumption, but it is a real risk. A future hardening pass (run the loop inside a container, drop privileges, network-restrict) is out of scope for 2b and should be its own plan. The `_block_real_claude` test fixture ensures no real execution happens in CI; the risk is only at real `run` time.
- *Shrink-flag fragility.* `_TINY_FLAGS = "--limit 8 --batch-size 1"` is an lm-eval/HF-style convention. A repo with a custom eval script that doesn't accept those flags will fail its first smoke run on an *argument* error rather than a genuine env error — wasting a fix turn (the agent will usually adapt, since the traceback shows the bad flag). This is a heuristic, not a guarantee; richer per-repo smoke-command inference is deferred along with the example/test discovery above.
