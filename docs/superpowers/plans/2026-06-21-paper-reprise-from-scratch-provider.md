# Plan 2d: From-Scratch Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the **from-scratch provider** (design §6): reproduce papers with NO official repo by having headless Claude IMPLEMENT the paper's quantization method from its LaTeX + spec, instead of cloning + running an official repo. Build the deterministic skeleton — provider plumbing, the scaffold agentic seam (injected in tests), provider selection, and CLI wiring so a `repo: null` paper routes here — exactly as Plan 2b/2c shipped the setup/run skeletons with their real GPU/Claude work deferred behind injectable seams.

**Architecture:** Reuse the EXISTING pipeline seams — **no new pipeline stage**. The from-scratch path is just a different pair of `setup_executor` / `run_executor` implementations injected into the SAME `run_setup` / `run_claims` seams the official path already uses. A new `provider.py` holds a **dispatcher**: `make_setup_dispatcher` / `make_run_dispatcher` build executors that, at call time, inspect the run dir to decide whether an official repo was cloned (`rd.repo_dir` non-empty) and route to the official executors (`make_setup_executor` / `make_run_executor`) or to the new from-scratch executors. This keeps `run_pipeline`'s signature untouched (the instruction's preferred option: dispatch on `rd` at call time, do not change the pipeline contract). A new `fromscratch.py` holds the from-scratch executors and their injectable seams, mirroring `setuploop.py`/`runexec.py` line-for-line in style.

**Tech Stack:** Python 3.11, stdlib `subprocess`/`json`/`pathlib`/`time`, pydantic (existing models), pytest (offline via injected fakes), uv, ruff (line-length 100). `from __future__ import annotations` in every module.

Design doc: `docs/superpowers/specs/2026-06-19-paper-reprise-agent-design.md` (§6 From-Scratch Path; §2.2 isolation; §4.1 setup loop discipline; §4.2 run).

---

## Context for the implementer

Plan 1 shipped the deterministic skeleton; 2a made ingest fetch; 2b implemented the agentic setup loop; 2c the GPU run executor. This plan (2d) implements the from-scratch provider §6 reserved interface as a deterministic skeleton.

**The seams you MUST reuse unchanged (do NOT change signatures):**

- `setupstage.run_setup(rd, spec, executor=...) -> SetupResult` — pipeline.py:70 calls it with the injected `setup_executor`. The from-scratch setup executor is just another `executor(rd, spec) -> SetupResult`.
- `runstage.run_claims(rd, spec, executor=...) -> (list[RunResult], dict)` — pipeline.py:75. The from-scratch run executor is just another `executor(claim, artifact, claim_dir) -> dict` with keys `stdout_path, actual_config, gpu, seed, minutes`. A non-zero exit must RAISE so `run_claims` marks the claim BLOCKED (runstage.py:33-38). **Do NOT touch runstage.py.**
- `SetupResult(ok, env_snapshot, patches, error)` (setupstage.py) — reuse exactly; report.py reads `env_snapshot` (keys `torch`/`transformers`/`cuda`) and `patches`.
- `run_pipeline(...)` / `resume_pipeline(...)` signatures — unchanged. They already accept `setup_executor` / `run_executor` as injected callables. Provider selection happens by what the CLI injects (a dispatcher), NOT by changing the pipeline.

**Existing helpers to REUSE (do not reimplement):**

- `setuploop._create_env(env_dir, manager) -> (int, str)`, `setuploop._run_smoke(command, cwd, env_dir) -> (int, str)`, `setuploop._freeze_env(env_dir) -> dict`, `setuploop.assemble_snapshot(freeze) -> dict`, `setuploop.collect_new_patches(dir, seen) -> list[str]` — the from-scratch setup executor reuses these for env build, smoke test, freeze, snapshot normalization, and patch-trail capture.
- `runexec.resolve_actual_config(claim, artifact) -> dict`, `runexec.extract_seed(command) -> int|None`, `runexec._detect_gpu() -> str`, `runexec._rundir_paths(claim_dir) -> (root, env_dir, repo_dir)`, `runexec._run_eval(command, cwd, env_dir, log_path) -> (int, str)` — the from-scratch run executor reuses these for config resolution, seed/GPU metadata, path derivation, and the activated-env subprocess.
- `headless.run_headless(prompt, allowed_tools, cwd, expect_file, timeout) -> HeadlessResult` — the scaffold call wraps this; injectable via `headless._call_claude` (the autouse fixture blocks the real binary).
- `RunDir`: `rd.root`, `rd.repo_dir`, `rd.setup_log_dir`, `rd.setup_patches_dir`, `rd.claim_dir(id)`.

**Provider selection signal (decided):** at executor call time the only deterministic, already-available signal for "official repo found?" is **whether `rd.repo_dir` contains a cloned repo**. `fetch.make_fetch_sources` clones into `rd.repo_dir` ONLY when it finds a GitHub url in the paper; otherwise that dir stays empty (it is always `mkdir`-ed by `RunDir.create`, so existence alone is not the signal — non-emptiness is). `ingest.json`'s `repo` field is only populated at report time on the current code path, so it is NOT reliable at setup/run time; `rd.repo_dir` non-emptiness is. The dispatcher uses `repo_present(rd)` = "repo_dir has any entry". This matches the design's "Papers ingest-marked `repo: null` … take the from-scratch path".

**Isolation discipline (design §2.2 / §6):** `fromscratch.py` must NOT import `grade`/`report`; must NOT read `expected`/`tolerance` for control flow. It only scaffolds (agent implements the method), runs the scaffolded entrypoint, and persists raw output. Grade stays pure and unchanged.

**Setup/run separation:** the from-scratch SETUP executor builds the env + scaffolds `impl/` + smoke-tests the entrypoint at tiny scale (it makes the impl *runnable*, never computes real numbers). The from-scratch RUN executor runs the scaffolded entrypoint per claim at full scale and persists raw stdout. Same split as official.

**Everything offline-testable:** the scaffold headless call, env build, smoke/eval subprocess, and clock are all behind injectable seams; tests inject fakes; the autouse `_block_real_network` fixture backstops anything that forgot.

---

## File Structure

```
src/paper_reprise/
  fromscratch.py   # NEW — from-scratch setup+run executors, scaffold prompt + seams.
  provider.py      # NEW — repo_present(rd) + make_setup_dispatcher / make_run_dispatcher.
  cli.py           # MODIFY — inject the dispatchers instead of the official executors directly.
tests/
  test_fromscratch.py  # NEW — offline tests: scaffold prompt, setup executor, run executor.
  test_provider.py     # NEW — offline tests: repo_present + dispatch routing.
  test_cli.py          # MODIFY — assert the CLI injects the dispatchers.
```

**Responsibility split inside `fromscratch.py`:**

- Pure logic (offline-testable directly, no I/O):
  - `build_scaffold_prompt(rd, spec) -> str` — instruct headless Claude to read `paper/` LaTeX + the spec, implement the paper's quant method as a SELF-CONTAINED impl under `<run>/impl/`, exposing ONE entrypoint `impl/run_eval.sh <claim_id>` that prints the metric to stdout; forbid fabricating numbers; record a one-line note per file (patch-note discipline). Pure string builder.
  - `fromscratch_smoke_command() -> str` — the tiny-scale entrypoint invocation used to smoke-test the scaffold (`bash impl/run_eval.sh --smoke`).
  - `fromscratch_eval_command(claim) -> str` — the per-claim entrypoint invocation (`bash impl/run_eval.sh <claim.id>`).
- Injectable I/O seams (the only functions touching the outside world):
  - `_run_scaffold(prompt, cwd, expect_file, timeout) -> bool` — one headless "implement the method" turn, wrapping `run_headless`; returns whether the expected entrypoint appeared.
- Orchestration (testable via injected fakes + a `now` clock):
  - `run_fromscratch_setup(rd, spec, *, max_retries, timeout_s, now, create_env, run_scaffold, run_smoke, freeze_env) -> SetupResult` — build env → scaffold `impl/` → smoke-test the entrypoint → retry/timeout guardrails identical in spirit to the setup loop; never silently give up; freeze snapshot on success.
- Factories (the executors the dispatcher/pipeline inject):
  - `make_fromscratch_setup_executor(*, manager, max_retries, timeout_s) -> Callable[[RunDir, Spec], SetupResult]`
  - `make_fromscratch_run_executor(*, run_eval, detect_gpu, now) -> Callable[[Claim, Artifact, Path], dict]`

**How the hard parts are solved (read before coding):**

1. **No new stage.** The from-scratch path reuses `run_setup` / `run_claims` verbatim. The pipeline does not know which provider it is using; the CLI injects a dispatcher that picks per run dir.
2. **Provider selection is deterministic and call-time.** `repo_present(rd)` checks `rd.repo_dir` non-emptiness. The dispatcher closes over BOTH executors and calls the right one. `run_pipeline`'s contract is untouched.
3. **Scaffold success is machine-decidable.** Like the setup loop judging smoke by exit code, the from-scratch setup judges scaffold success by (a) the entrypoint file `impl/run_eval.sh` appearing AND (b) the smoke run of that entrypoint exiting 0. Failure hands the captured output back to the next scaffold turn. Bounded by `max_retries` + `timeout_s` (injected clock).
4. **Run executor reuses the official subprocess path.** `make_fromscratch_run_executor` is structurally `make_run_executor` but builds the command from the scaffolded entrypoint (`fromscratch_eval_command`) and resolves cwd to `rd.root` (the impl lives under the run dir, not the empty repo dir). Non-zero exit raises → BLOCKED. Persists `stdout.log` + `actual_config.json`, returns the same metadata dict.

---

## Task 1: scaffold prompt + entrypoint command builders (pure logic)

**Files:**
- Create: `src/paper_reprise/fromscratch.py`
- Test: `tests/test_fromscratch.py`

`build_scaffold_prompt` is the instruction handed to headless Claude: read the paper LaTeX + spec, implement the method self-contained under `impl/`, expose ONE entrypoint, forbid fabricating numbers, record a one-line note per file. `fromscratch_smoke_command` / `fromscratch_eval_command` produce the tiny-scale and per-claim entrypoint invocations. All pure → assertable offline.

- [ ] **Step 1: Write the failing test**

`tests/test_fromscratch.py`:
```python
from pathlib import Path

from paper_reprise.models import Artifact, Claim, EvalProtocol, Spec
from paper_reprise.rundir import RunDir
from paper_reprise.fromscratch import (
    build_scaffold_prompt,
    fromscratch_eval_command,
    fromscratch_smoke_command,
)


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
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_fromscratch.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.fromscratch'`

- [ ] **Step 3: Write the implementation**

`src/paper_reprise/fromscratch.py`:
```python
"""From-scratch provider: reproduce papers with NO official repo by implementing
the paper's quantization method from its description (design §6).

This is the sibling of the official-repo path (setuploop + runexec). Instead of
cloning + running an existing repo, headless Claude IMPLEMENTS the paper's method
as a self-contained `impl/` under the run dir, exposing ONE runnable entrypoint;
the same env-build + smoke-test + retry/timeout guardrails make it runnable, then
the run executor runs that entrypoint per claim and persists raw output. Grade is
untouched and never sees this module — isolation per §2.2.

Every real-world action (the scaffold headless call, env build, smoke/eval
subprocess, the clock) is behind an injectable seam so the whole thing is
offline-testable; the autouse fixture backstops anything that forgets.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from paper_reprise.headless import run_headless
from paper_reprise.models import Artifact, Claim, Spec
from paper_reprise.rundir import RunDir
from paper_reprise.runexec import (
    _detect_gpu,
    _rundir_paths,
    _run_eval,
    extract_seed,
    resolve_actual_config,
)
from paper_reprise.setuploop import (
    _create_env,
    _freeze_env,
    _run_smoke,
    assemble_snapshot,
    collect_new_patches,
)
from paper_reprise.setupstage import SetupResult

# The single runnable entrypoint the scaffold MUST produce (design §6: "executable
# eval commands"). Kept conventional so the smoke + run commands are deterministic.
_ENTRYPOINT = "impl/run_eval.sh"

# Guardrails mirroring the setup loop's philosophy (bounded, never silent give-up).
_SCAFFOLD_TIMEOUT_S = 1800


def fromscratch_smoke_command() -> str:
    """Tiny-scale invocation of the scaffolded entrypoint for the smoke test."""
    return f"bash {_ENTRYPOINT} --smoke"


def fromscratch_eval_command(claim: Claim) -> str:
    """Per-claim invocation of the scaffolded entrypoint (prints the metric)."""
    return f"bash {_ENTRYPOINT} {claim.id}"


_SCAFFOLD_TEMPLATE = """No official repo exists for this paper. IMPLEMENT the \
paper's quantization method ({methods}) from scratch, as a SELF-CONTAINED \
implementation under `impl/` in this run directory.

Read the paper LaTeX source in `paper/` and the extracted reproduction spec in \
`spec.yaml` (artifacts = quantized products, claims = one metric each). Implement \
exactly the method and eval protocol the spec describes.

Expose EXACTLY ONE runnable entrypoint `{entrypoint}` that:
  - takes a single argument: a claim id (e.g. `c1`), or `--smoke` for a tiny-scale \
self-test (a few samples, batch 1) used only to prove the code runs;
  - quantizes per the claim's artifact config and runs its eval protocol;
  - prints the resulting metric value to stdout in a parseable form \
(e.g. `perplexity: 5.80`).

HONESTY RULES (mandatory):
  - Do NOT fabricate, invent, or hard-code any result number. The entrypoint must \
COMPUTE the metric. A run that cannot compute must exit non-zero, never print a \
made-up value.
  - Do NOT read the paper's expected values or tolerances to shortcut the result.

For EACH file you create under `impl/`, append ONE line describing what it \
implements to `{patch_note}` (create the file; one line per file). When `impl/` \
and `{entrypoint}` exist and `--smoke` runs, you are done."""


def build_scaffold_prompt(rd: RunDir, spec: Spec) -> str:
    """Build the headless-claude instruction to implement the paper's method as a
    self-contained `impl/` with one runnable entrypoint. Pure string builder."""
    methods = ", ".join(sorted({a.method for a in spec.artifacts})) or "the paper's method"
    return _SCAFFOLD_TEMPLATE.format(
        methods=methods, entrypoint=_ENTRYPOINT,
        patch_note="setup_patches/scaffold.txt",
    )
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_fromscratch.py -v`
Expected: PASS, 3 green.

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/fromscratch.py tests/test_fromscratch.py
git commit -m "feat(fromscratch): scaffold prompt + entrypoint command builders (pure logic)"
```

---

## Task 2: from-scratch setup executor — scaffold → smoke (orchestration, all I/O injected)

**Files:**
- Modify: `src/paper_reprise/fromscratch.py`
- Test: `tests/test_fromscratch.py`

`run_fromscratch_setup` is the orchestrator. Happy path: build env → scaffold `impl/` → smoke-run the entrypoint → exit 0 → freeze snapshot → `SetupResult(ok=True)`. Plus the guardrail paths: scaffold fails → retry → cap; timeout via injected clock; env-build failure surfaced. Same "never silently give up, hand off the log" discipline as the setup loop. Every I/O seam is an injected keyword-only callable; the `_run_scaffold` seam is added in Task 4 as the real default.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fromscratch.py`:
```python
import json

from paper_reprise.setupstage import SetupResult
from paper_reprise.fromscratch import run_fromscratch_setup


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
```

- [ ] **Step 2: Run the tests, confirm they fail**

Run: `uv run pytest tests/test_fromscratch.py -v`
Expected: FAIL, `ImportError: cannot import name 'run_fromscratch_setup'`

- [ ] **Step 3: Write the implementation**

Add to `src/paper_reprise/fromscratch.py` (the helper + orchestrator; the real `_run_scaffold` default lands in Task 4 — for now the seam is required-by-keyword, resolved at call time like setuploop):

```python
def _write_log(rd: RunDir, name: str, text: str) -> None:
    (rd.setup_log_dir / name).write_text(text)


def run_fromscratch_setup(
    rd: RunDir,
    spec: Spec,
    *,
    manager: str = "uv",
    max_retries: int = 6,
    timeout_s: float = 3600.0,
    now: Callable[[], float] = time.monotonic,
    create_env: Callable[[Path, str], tuple[int, str]] | None = None,
    run_scaffold: Callable[[str, Path, Path, float], bool] | None = None,
    run_smoke: Callable[[str, Path, Path], tuple[int, str]] | None = None,
    freeze_env: Callable[[Path], dict] | None = None,
) -> SetupResult:
    """Build the env, let the agent IMPLEMENT the method under impl/, smoke-test the
    entrypoint until it runs once — under a retry cap AND a total timeout. On
    exhaustion: ok=False with the full log handed off; never a silent give-up. Never
    lets an exception escape (a crash becomes ok=False so the pipeline continues)."""
    create_env = create_env or _create_env
    run_scaffold = run_scaffold or _run_scaffold
    run_smoke = run_smoke or _run_smoke
    freeze_env = freeze_env or _freeze_env
    try:
        return _fromscratch_setup_body(
            rd, spec, manager=manager, max_retries=max_retries, timeout_s=timeout_s,
            now=now, create_env=create_env, run_scaffold=run_scaffold,
            run_smoke=run_smoke, freeze_env=freeze_env)
    except Exception as e:  # never crash the pipeline; setup failure must be ok=False
        return SetupResult(ok=False, error=f"from-scratch setup crashed: {e!r}; see setup_log/")


def _fromscratch_setup_body(
    rd: RunDir,
    spec: Spec,
    *,
    manager: str,
    max_retries: int,
    timeout_s: float,
    now: Callable[[], float],
    create_env: Callable[[Path, str], tuple[int, str]],
    run_scaffold: Callable[[str, Path, Path, float], bool],
    run_smoke: Callable[[str, Path, Path], tuple[int, str]],
    freeze_env: Callable[[Path], dict],
) -> SetupResult:
    env_dir = rd.root / "env"
    smoke_cmd = fromscratch_smoke_command()
    entrypoint = rd.root / _ENTRYPOINT
    prompt = build_scaffold_prompt(rd, spec)
    start = now()
    seen_patches: set[str] = set()
    patches: list[str] = []

    # --- build env once ---
    code, env_log = create_env(env_dir, manager)
    _write_log(rd, "create_env.log", env_log)
    if code != 0:
        return SetupResult(ok=False, patches=patches,
                           error=f"env creation failed (exit {code}); see setup_log/")

    # --- scaffold → smoke → retry loop ---
    attempt = 0
    while True:
        if now() - start >= timeout_s:
            return SetupResult(ok=False, patches=patches,
                               error=f"from-scratch setup timed out after {timeout_s}s "
                                     f"({attempt} attempts); see setup_log/")
        produced = run_scaffold(prompt, rd.root, entrypoint, _SCAFFOLD_TIMEOUT_S)
        patches.extend(collect_new_patches_scaffold(rd.setup_patches_dir, seen_patches))
        if produced:
            code, out = run_smoke(smoke_cmd, rd.root, env_dir)
            _write_log(rd, f"smoke_{attempt}.log", out)
            if code == 0:
                try:
                    snapshot = assemble_snapshot(freeze_env(env_dir))
                except Exception:
                    snapshot = assemble_snapshot({})
                try:
                    (rd.root / "env_snapshot.json").write_text(json.dumps(snapshot, indent=2))
                except OSError:
                    pass
                return SetupResult(ok=True, env_snapshot=snapshot, patches=patches)
        else:
            _write_log(rd, f"scaffold_{attempt}.log",
                       f"scaffold turn {attempt} did not produce {_ENTRYPOINT}")
        if attempt >= max_retries:
            return SetupResult(ok=False, patches=patches,
                               error=f"impl smoke still failing after {max_retries} "
                                     f"retries; see setup_log/")
        attempt += 1
```

For patch collection, the scaffold uses `scaffold_*.txt` notes (distinct from the setup loop's `patch_*.txt`), so add a thin wrapper reusing the same set-diff logic but over the scaffold glob:

```python
def collect_new_patches_scaffold(patches_dir: Path, seen: set[str]) -> list[str]:
    """Set-diff scaffold_*.txt patch notes — the from-scratch analogue of
    setuploop.collect_new_patches (which globs patch_*.txt)."""
    new: list[str] = []
    for p in sorted(patches_dir.glob("scaffold*.txt")):
        if p.name in seen:
            continue
        seen.add(p.name)
        new.append(p.read_text().strip())
    return new
```

> Note: import `collect_new_patches` from setuploop is kept in the top imports for reuse symmetry, but the scaffold notes use a distinct glob (`scaffold*.txt`) so a `scaffold.txt` written by the real agent (Task 1's prompt points at `setup_patches/scaffold.txt`) and per-turn `scaffold_<n>.txt` notes are both captured. If you prefer to avoid the unused import, drop `collect_new_patches` from the import list — ruff will flag it.

- [ ] **Step 4: Run the tests, confirm they pass**

Run: `uv run pytest tests/test_fromscratch.py -v`
Expected: PASS. The timeout clock sequence `0.0, 5.0, 999.0`: `start=0.0`, first check `5.0 < 100` (proceeds, scaffold ok, smoke fails), second check `999.0 >= 100` → times out. If the retry-cap test's patch assertion is brittle, keep the assertion on `res.ok is False` + error text only.

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/fromscratch.py tests/test_fromscratch.py
git commit -m "feat(fromscratch): setup executor (scaffold→smoke) with retry/timeout guardrails"
```

---

## Task 3: from-scratch run executor — run the scaffolded entrypoint per claim

**Files:**
- Modify: `src/paper_reprise/fromscratch.py`
- Test: `tests/test_fromscratch.py`

`make_fromscratch_run_executor` mirrors `runexec.make_run_executor` but (a) builds the command from `fromscratch_eval_command(claim)` and (b) runs it with cwd = the run root (the impl lives under the run dir, not the empty repo dir). It persists `stdout.log` + `actual_config.json`, returns `{stdout_path, actual_config, gpu, seed, minutes}`, and RAISES on non-zero exit so `run_claims` marks it BLOCKED.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fromscratch.py`:
```python
from paper_reprise.fromscratch import make_fromscratch_run_executor
from paper_reprise.runstage import run_claims


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
```

- [ ] **Step 2: Run the tests, confirm they fail**

Run: `uv run pytest tests/test_fromscratch.py -v`
Expected: FAIL, `ImportError: cannot import name 'make_fromscratch_run_executor'`

- [ ] **Step 3: Write the implementation**

Add to `src/paper_reprise/fromscratch.py`:
```python
def make_fromscratch_run_executor(
    *,
    run_eval: Callable[[str, Path, Path, Path], tuple[int, str]] | None = None,
    detect_gpu: Callable[[], str] | None = None,
    now: Callable[[], float] | None = None,
) -> Callable[[Claim, Artifact, Path], dict]:
    """Build the executor(claim, artifact, claim_dir) -> dict that run_claims injects
    on the from-scratch path. Runs the scaffolded entrypoint for the claim in the
    setup-built env (cwd = the run root, where impl/ lives), persists raw stdout +
    the resolved actual_config, and returns run metadata. A non-zero exit raises so
    run_claims marks the claim BLOCKED (the eval did not run — not 'failed to
    reproduce'). Mirrors runexec.make_run_executor, reusing its seams."""
    run_eval = run_eval or _run_eval
    detect_gpu = detect_gpu or _detect_gpu
    now = now or time.monotonic

    def executor(claim: Claim, artifact: Artifact, claim_dir: Path) -> dict:
        root, env_dir, _repo_dir = _rundir_paths(claim_dir)
        command = fromscratch_eval_command(claim)
        log_path = claim_dir / "stdout.log"
        gpu = detect_gpu()
        start = now()
        code, _out = run_eval(command, root, env_dir, log_path)
        minutes = (now() - start) / 60.0

        actual_config = resolve_actual_config(claim, artifact)
        (claim_dir / "actual_config.json").write_text(json.dumps(actual_config, indent=2))

        if code != 0:
            raise RuntimeError(f"from-scratch eval exited {code}; see {log_path}")

        return {
            "stdout_path": str(log_path),
            "actual_config": actual_config,
            "gpu": gpu,
            "seed": extract_seed(command),
            "minutes": minutes,
        }

    return executor
```

- [ ] **Step 4: Run the tests, confirm they pass**

Run: `uv run pytest tests/test_fromscratch.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/fromscratch.py tests/test_fromscratch.py
git commit -m "feat(fromscratch): run executor — run scaffolded entrypoint per claim, BLOCKED on failure"
```

---

## Task 4: real scaffold seam + factories (make_fromscratch_setup_executor)

**Files:**
- Modify: `src/paper_reprise/fromscratch.py`
- Test: `tests/test_fromscratch.py`

Add the real `_run_scaffold` seam (thin wrapper over `run_headless`, judging success by the entrypoint file appearing) and the `make_fromscratch_setup_executor` factory returning `executor(rd, spec) -> SetupResult` (matching `run_setup`'s seam). The run-executor factory already exists (Task 3).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fromscratch.py`:
```python
import paper_reprise.fromscratch as fromscratch
from paper_reprise.fromscratch import make_fromscratch_setup_executor


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
    # _run_scaffold returns True iff the expected entrypoint appears (run_headless
    # contract). We stub run_headless to simulate the agent writing the file.
    expect = tmp_path / "impl" / "run_eval.sh"

    def fake_run_headless(prompt, allowed_tools, cwd, expect_file, timeout=None):
        Path(expect_file).parent.mkdir(parents=True, exist_ok=True)
        Path(expect_file).write_text("echo perplexity: 5.8")
        from paper_reprise.headless import HeadlessResult
        return HeadlessResult(ok=True, output_path=expect_file)

    monkeypatch.setattr(fromscratch, "run_headless", fake_run_headless)
    assert fromscratch._run_scaffold("prompt", tmp_path, expect, 5.0) is True
```

- [ ] **Step 2: Run the tests, confirm they fail**

Run: `uv run pytest tests/test_fromscratch.py -v`
Expected: FAIL, `ImportError: cannot import name 'make_fromscratch_setup_executor'` (and `_run_scaffold` not yet defined).

- [ ] **Step 3: Write the implementation**

Add the real seam + factory to `src/paper_reprise/fromscratch.py`. Place `_run_scaffold` ABOVE `run_fromscratch_setup` (so the `run_scaffold or _run_scaffold` default resolves), or keep the call-time `or _run_scaffold` resolution (preferred, matches setuploop — then position does not matter):

```python
def _run_scaffold(prompt: str, cwd: Path, expect_file: Path, timeout: float) -> bool:
    """One headless-claude 'implement the method' turn. Success = the expected
    entrypoint file appeared (run_headless's own contract). The loop re-runs the
    smoke command to judge whether the impl actually works; this only reports that
    the agent produced the entrypoint."""
    res = run_headless(prompt=prompt, allowed_tools=["Read", "Write", "Edit", "Bash"],
                       cwd=cwd, expect_file=expect_file, timeout=timeout)
    return res.ok


def make_fromscratch_setup_executor(*, manager: str = "uv", max_retries: int = 6,
                                    timeout_s: float = 3600.0
                                    ) -> Callable[[RunDir, Spec], SetupResult]:
    """Build the executor(rd, spec) the pipeline injects into run_setup on the
    from-scratch path."""
    def executor(rd: RunDir, spec: Spec) -> SetupResult:
        return run_fromscratch_setup(rd, spec, manager=manager,
                                     max_retries=max_retries, timeout_s=timeout_s)

    return executor
```

> Confirm `run_fromscratch_setup` resolves `run_scaffold = run_scaffold or _run_scaffold` at call time (Task 2), so monkeypatching `fromscratch._run_scaffold` takes effect (the test relies on this).

- [ ] **Step 4: Run the tests, confirm they pass**

Run: `uv run pytest tests/test_fromscratch.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/fromscratch.py tests/test_fromscratch.py
git commit -m "feat(fromscratch): real scaffold headless seam + make_fromscratch_setup_executor"
```

---

## Task 5: provider dispatcher — repo_present + route official vs from-scratch

**Files:**
- Create: `src/paper_reprise/provider.py`
- Test: `tests/test_provider.py`

`repo_present(rd)` returns whether `rd.repo_dir` was actually populated by a clone (non-empty). `make_setup_dispatcher` / `make_run_dispatcher` build executors that, at call time, route to the official executors when a repo is present, else the from-scratch executors. This is where provider selection lives — the pipeline stays provider-agnostic.

- [ ] **Step 1: Write the failing test**

`tests/test_provider.py`:
```python
from pathlib import Path

from paper_reprise.models import Artifact, Claim, EvalProtocol, Spec
from paper_reprise.rundir import RunDir
from paper_reprise.provider import (
    make_run_dispatcher,
    make_setup_dispatcher,
    repo_present,
)


def _spec():
    return Spec(
        paper="2401.00001", repo=None,
        artifacts=[Artifact(id="a1", base_model="m", method="AWQ", quant_config={"wbits": 4})],
        claims=[Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="custom", command="x",
                                                 metric="perplexity", dataset="wikitext2"),
                      expected=5.78, tolerance=0.05, source="T")],
    )


def test_repo_present_false_when_repo_dir_empty(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")   # repo_dir mkdir-ed, empty
    assert repo_present(rd) is False


def test_repo_present_true_when_repo_dir_has_content(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    (rd.repo_dir / "README.md").write_text("cloned repo")
    assert repo_present(rd) is True


def test_setup_dispatcher_routes_to_official_when_repo_present(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    (rd.repo_dir / "setup.py").write_text("x")
    called = {}
    dispatcher = make_setup_dispatcher(
        official=lambda rd, spec: called.setdefault("which", "official"),
        fromscratch=lambda rd, spec: called.setdefault("which", "fromscratch"))
    dispatcher(rd, _spec())
    assert called["which"] == "official"


def test_setup_dispatcher_routes_to_fromscratch_when_no_repo(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")   # empty repo_dir
    called = {}
    dispatcher = make_setup_dispatcher(
        official=lambda rd, spec: called.setdefault("which", "official"),
        fromscratch=lambda rd, spec: called.setdefault("which", "fromscratch"))
    dispatcher(rd, _spec())
    assert called["which"] == "fromscratch"


def test_run_dispatcher_routes_by_repo_presence(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    claim_dir = rd.claim_dir("c1")
    called = {}
    dispatcher = make_run_dispatcher(
        official=lambda claim, art, cd: called.setdefault("which", "official"),
        fromscratch=lambda claim, art, cd: called.setdefault("which", "fromscratch"))
    # no repo content → from-scratch
    dispatcher(_spec().claims[0], _spec().artifacts[0], claim_dir)
    assert called["which"] == "fromscratch"
    # now add a repo → official
    (rd.repo_dir / "main.py").write_text("x")
    called.clear()
    dispatcher(_spec().claims[0], _spec().artifacts[0], claim_dir)
    assert called["which"] == "official"
```

- [ ] **Step 2: Run the tests, confirm they fail**

Run: `uv run pytest tests/test_provider.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.provider'`

- [ ] **Step 3: Write the implementation**

`src/paper_reprise/provider.py`:
```python
"""Provider selection: route the setup/run executors to the official-repo path or
the from-scratch path (design §6) by whether an official repo was cloned.

No new pipeline stage. The pipeline injects ONE setup_executor and ONE
run_executor; these dispatchers wrap BOTH provider implementations and pick at
call time per run dir, so run_pipeline's contract is untouched. Selection signal:
rd.repo_dir is non-empty iff ingest cloned an official repo (fetch clones there
only when it finds a GitHub url; otherwise the dir stays empty). A paper with
repo: null therefore routes to from-scratch.
"""
from __future__ import annotations

from typing import Callable

from paper_reprise.models import Artifact, Claim, Spec
from paper_reprise.rundir import RunDir
from paper_reprise.runexec import _rundir_paths
from paper_reprise.setupstage import SetupResult


def repo_present(rd: RunDir) -> bool:
    """True iff an official repo was cloned into rd.repo_dir (the dir is always
    mkdir-ed, so we test non-emptiness, not existence)."""
    return rd.repo_dir.is_dir() and any(rd.repo_dir.iterdir())


def make_setup_dispatcher(
    *,
    official: Callable[[RunDir, Spec], SetupResult],
    fromscratch: Callable[[RunDir, Spec], SetupResult],
) -> Callable[[RunDir, Spec], SetupResult]:
    """Build the setup executor the pipeline injects: official path when a repo was
    cloned, from-scratch otherwise."""
    def executor(rd: RunDir, spec: Spec) -> SetupResult:
        return (official if repo_present(rd) else fromscratch)(rd, spec)

    return executor


def make_run_dispatcher(
    *,
    official: Callable[[Claim, Artifact, "Path"], dict],
    fromscratch: Callable[[Claim, Artifact, "Path"], dict],
) -> Callable[[Claim, Artifact, "Path"], dict]:
    """Build the run executor the pipeline injects: routes per run dir, derived from
    the claim_dir, by the same repo-presence signal as setup."""
    def executor(claim: Claim, artifact: Artifact, claim_dir) -> dict:
        root, _env, repo_dir = _rundir_paths(claim_dir)
        present = repo_dir.is_dir() and any(repo_dir.iterdir())
        chosen = official if present else fromscratch
        return chosen(claim, artifact, claim_dir)

    return executor
```

> `Path` is referenced only in annotations (string-quoted) — keep `from __future__ import annotations` and you do not need to import `Path`. If ruff prefers a real import for clarity, `from pathlib import Path` and unquote; either is fine, just keep it import-clean.

- [ ] **Step 4: Run the tests, confirm they pass**

Run: `uv run pytest tests/test_provider.py -v`
Expected: PASS, 5 green.

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/provider.py tests/test_provider.py
git commit -m "feat(provider): repo-presence dispatcher routing official vs from-scratch executors"
```

---

## Task 6: wire the dispatchers into the CLI

**Files:**
- Modify: `src/paper_reprise/cli.py`
- Test: `tests/test_cli.py`

The CLI currently injects the official executors directly (`make_setup_executor()` / `make_run_executor()`). Replace each with a dispatcher that wraps the official executor AND the from-scratch executor, so a `repo: null` paper routes to from-scratch automatically. Both `run` and `resume` commands.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:
```python
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
    from paper_reprise.rundir import RunDir
    from paper_reprise.models import Artifact, Claim, EvalProtocol, Spec
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
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_cli.py::test_cli_run_injects_dispatching_executors -v`
Expected: FAIL — the CLI currently injects the official executor directly, which returns a `SetupResult`/dict, not `"fromscratch-setup"`; and `make_fromscratch_setup_executor` is not imported.

- [ ] **Step 3: Write the implementation**

In `src/paper_reprise/cli.py`, add imports:
```python
from paper_reprise.fromscratch import (
    make_fromscratch_run_executor,
    make_fromscratch_setup_executor,
)
from paper_reprise.provider import make_run_dispatcher, make_setup_dispatcher
```

Add a small helper near the top (after `_timestamp`) so `run` and `resume` build the same dispatchers:
```python
def _setup_executor():
    return make_setup_dispatcher(
        official=make_setup_executor(),
        fromscratch=make_fromscratch_setup_executor())


def _run_executor():
    return make_run_dispatcher(
        official=make_run_executor(),
        fromscratch=make_fromscratch_run_executor())
```

In the `run` command's `run_pipeline(...)` call replace:
```python
        fetch_sources=make_fetch_sources(), setup_executor=make_setup_executor(),
        run_executor=make_run_executor(),
```
with:
```python
        fetch_sources=make_fetch_sources(), setup_executor=_setup_executor(),
        run_executor=_run_executor(),
```

In the `resume` command's `resume_pipeline(...)` call replace:
```python
        setup_executor=make_setup_executor(), run_executor=make_run_executor(),
```
with:
```python
        setup_executor=_setup_executor(), run_executor=_run_executor(),
```

> The existing `test_cli_run_passes_real_setup_executor` / `..._run_executor` tests monkeypatch `cli_mod.make_setup_executor` / `cli_mod.make_run_executor` to a sentinel and assert the injected executor IS that sentinel. After this change the injected executor is a DISPATCHER wrapping the sentinel, not the sentinel itself — those two tests will break. Update them to assert the dispatcher routes to the (sentinel) official executor when a repo is present, mirroring the new test. Adjust them in this same step so the suite stays green:
> - `test_cli_run_passes_real_setup_executor`: stub `make_setup_executor`/`make_fromscratch_setup_executor` to labeled lambdas, create a run dir WITH repo content, assert `captured["setup_executor"](rd, spec) == "official-setup"`.
> - `test_cli_run_passes_real_run_executor`: same shape for the run executor with a repo-present claim dir → `"official-run"`.

- [ ] **Step 4: Run the tests, confirm they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS, all CLI tests green (the two updated ones + the new dispatch test + the untouched rest).

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/cli.py tests/test_cli.py
git commit -m "feat(provider): wire repo-presence dispatchers into CLI run + resume"
```

---

## Task 7: full suite + ruff + offline guarantee

**Files:** reuse existing

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS, all tests green (134 baseline + the new fromscratch/provider/cli tests). No test invokes real conda/uv/git/claude — every seam is injected or monkeypatched; the autouse `_block_real_network` fixture backstops the headless/HTTP paths.

- [ ] **Step 2: Confirm the new tests are fast (no leaked real subprocess)**

Run: `uv run pytest tests/test_fromscratch.py tests/test_provider.py -q`
Expected: PASS in well under a second.

- [ ] **Step 3: ruff check**

Run: `uv run ruff check src/ tests/`
Expected: "All checks passed!" Fix any F401 (unused imports) — in particular confirm every name imported into `fromscratch.py` (the reused setuploop/runexec helpers) is actually used, and drop `collect_new_patches` from the import if you used only the scaffold-glob wrapper.

- [ ] **Step 4: Commit (if ruff made changes)**

```bash
git add -A
git commit -m "chore: ruff clean for from-scratch provider"
```

---

## Self-Review

**1. Spec coverage** (design §6 From-Scratch Path + the architecture decisions):

- "Papers without an official repo take the from-scratch path; both providers implement the same interface (produce quantized products + executable eval commands), converging on the same grade/report" → the from-scratch executors satisfy the SAME `run_setup` / `run_claims` seams as the official ones (`SetupResult` and the `{stdout_path, actual_config, gpu, seed, minutes}` dict), so grade/report consume them unchanged. ✓
- "Provider selection based on whether an official repo was found" → `repo_present(rd)` (repo_dir non-empty) drives `make_setup_dispatcher` / `make_run_dispatcher`; a `repo: null` paper (empty repo_dir) routes to from-scratch (Task 5). ✓
- "Reuse the existing pipeline seams — add NO new pipeline stages" → no change to pipeline.py / runstage.py; the from-scratch path is purely a different injected executor pair behind a dispatcher (Tasks 5–6). ✓
- "build_scaffold_prompt PURE, offline-testable; implement the method self-contained under impl/; ONE entrypoint; forbid fabricating numbers; record a one-line note per file" → Task 1: pure builder, asserts `impl/run_eval.sh`, the no-fabrication rule, and the one-line-per-file note discipline. ✓
- "injectable headless seam wrapping run_headless" → `_run_scaffold` (Task 4), success keyed on the entrypoint appearing (run_headless's own contract). ✓
- "make_fromscratch_setup_executor: build env, run scaffold to produce impl/, smoke-test the entrypoint at tiny scale, SAME retry/timeout guardrails, never silently give up, log handoff; return SetupResult" → Task 2 (`run_fromscratch_setup`) + Task 4 (factory): bounded retry + injected-clock timeout, `ok=False` with `setup_log/` handoff on exhaustion, snapshot frozen on success. ✓
- "make_fromscratch_run_executor: run the entrypoint for the claim in the env, persist stdout.log, write actual_config.json, return the metadata dict (reuse resolve_actual_config); non-zero exit raises → BLOCKED" → Task 3, mirroring `runexec.make_run_executor`, reusing its seams; non-zero raises → `run_claims` marks BLOCKED. ✓
- "Isolation: fromscratch.py must NOT import grade/report; must not read expected/tolerance for control flow" → `fromscratch.py` imports only models/rundir/headless/runexec/setuploop/setupstage; `expected`/`tolerance` are never read; the scaffold prompt explicitly forbids the agent reading them too. ✓
- "Everything offline-testable; injected now clock for timeouts" → all four setup seams + the run seams + `now` are injectable; tests inject fakes and an iterator clock (Tasks 2–4). ✓

**2. Deferred (not omissions) — the agentic + GPU work, exactly as 2b/2c deferred theirs:**
- **The real agentic implementation** — "Claude correctly implements AWQ/GPTQ from the paper LaTeX" is the `_run_scaffold` seam itself. This phase builds + tests the seam with fakes; real execution needs a real Claude + GPU and is deferred, identical to how the official setup loop's real fix turns and the run executor's real GPU eval are deferred behind their seams.
- **Real GPU execution** of the scaffolded entrypoint — `_run_eval` (reused from runexec) is the real subprocess; not exercised in offline tests.
- **Verifying the impl matches the paper's method** beyond "the entrypoint runs and prints a parseable metric" — grade's faithfulness check on the from-scratch path compares the spec-resolved config against itself (same limitation runexec/grade already carry, see grade.py:42-49); a future pass should parse the impl's actual config from its output. Not introduced or worsened here.
- **Richer entrypoint contract** (multiple entrypoints, quant/eval split, structured result file) — kept to ONE shell entrypoint per §6's "executable eval commands"; structured I/O is a follow-up.

**3. Placeholder scan:** No TBD/TODO. Every code step has complete runnable code. The only "not implemented for real" is the agentic scaffold body, which is the injectable seam by design (matches 2b/2c).

**4. Type consistency** (signatures match across definition, defaults, fakes, and the seams they reuse):
- `SetupResult(ok, env_snapshot, patches, error)` — constructed with these exact kwargs throughout `run_fromscratch_setup` (Task 2); identical to setupstage's dataclass. ✓
- from-scratch setup seams: `create_env: (env_dir, manager)->(int,str)`, `run_smoke: (command, cwd, env_dir)->(int,str)`, `freeze_env: (env_dir)->dict` — IDENTICAL to setuploop's, reused as defaults; `run_scaffold: (prompt, cwd, expect_file, timeout)->bool` — new, consistent across `_run_scaffold`, the loop call site, and all fakes. ✓
- from-scratch run executor returns `{stdout_path, actual_config, gpu, seed, minutes}` and reuses `runexec._run_eval`/`_detect_gpu`/`extract_seed`/`resolve_actual_config`/`_rundir_paths` — same shape `run_claims` already consumes (runstage.py:28-32). ✓
- dispatchers: `make_setup_dispatcher(official, fromscratch) -> (rd, spec)->SetupResult` matches `run_setup(rd, spec, executor=...)`; `make_run_dispatcher(official, fromscratch) -> (claim, artifact, claim_dir)->dict` matches `run_claims(... executor=...)`. ✓
- circular imports: `fromscratch` imports `SetupResult` from `setupstage` (a plain dataclass, no cycle — `setupstage` does not import `fromscratch`); `provider` imports from `runexec`/`setupstage`/`models`/`rundir` (no cycle); the CLI imports the factories + dispatchers at module top (cli is a leaf). ✓

**5. Ambiguities and decisions made:**
- *Provider-selection signal.* The instruction says "ingest.repo tells you whether an official repo was found", but on the current code path `ingest.json`'s `repo` field is only populated at REPORT time (`pipeline.py:85 ingest.repo = spec.repo`), so it is `null` at setup/run time and unusable for routing. Decision: route on `rd.repo_dir` non-emptiness, which `fetch.make_fetch_sources` sets deterministically (it clones there iff it found a GitHub url). This is the faithful runtime expression of "an official repo was found", and it is available at exactly the call sites that need it. Documented in `provider.py`'s docstring. A future cleanup could populate `ingest.repo` at ingest time and switch the signal to that; not needed now and out of scope.
- *Where provider selection lives.* The instruction offered "a new provider.py OR cli.py". Decision: `provider.py` (the dispatcher logic) + thin CLI wiring. This keeps the routing testable in isolation (`test_provider.py`) and keeps the pipeline contract untouched (a dispatching executor inspecting `rd` at call time, the instruction's stated preference over changing the pipeline).
- *Scaffold patch-note glob.* The setup loop uses `patch_*.txt`; the scaffold uses `scaffold*.txt` so the two trails don't collide if a paper somehow exercised both (it can't in one run, but the namespacing is free and clearer in the report's patch trail). `collect_new_patches_scaffold` is a 6-line analogue of setuploop's `collect_new_patches`.
- *cwd for the from-scratch eval.* The official run executor uses `rd.repo_dir` as cwd; from-scratch has no repo, the impl lives at `rd.root/impl/`. Decision: cwd = `rd.root` so `bash impl/run_eval.sh` resolves. Surfaced explicitly in `make_fromscratch_run_executor` and asserted in Task 3.

**6. Known risks carried (flagged, not resolved here):**
- *Untrusted code execution / no sandbox.* The from-scratch path runs agent-authored code (`_run_smoke` / `_run_eval`, `shell=True`) and an agent with `Bash` (`_run_scaffold`) directly on the host — the same RCE surface the setup loop already carries (Plan 2b self-review §5). For a single-user local research tool this matches the manual-CLI assumption; container/privilege-drop hardening is a separate plan, unchanged by this one. The `_block_real_network` fixture ensures no real execution in tests.
- *Scaffold quality is unverified by the skeleton.* "The entrypoint runs and prints a parseable metric" is a necessary but not sufficient condition for a faithful reproduction; a wrong-but-runnable impl would pass setup and produce a number grade then judges. This is intrinsic to the from-scratch path (you are trusting the agent's implementation) and is exactly why the report flags `runner: custom` as "unofficial implementation" (design §3.3). The skeleton does not and cannot close this gap; the real `_run_scaffold` + a future faithfulness-from-output check are where it's addressed.
