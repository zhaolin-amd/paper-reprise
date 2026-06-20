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
)
from paper_reprise.setupstage import SetupResult

# Guardrails mirroring the setup loop's philosophy (bounded, never silent give-up).
_SCAFFOLD_TIMEOUT_S = 1800

# The single runnable entrypoint the scaffold MUST produce (design §6: "executable
# eval commands"). Kept conventional so the smoke + run commands are deterministic.
_ENTRYPOINT = "impl/run_eval.sh"


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


def _run_scaffold(prompt: str, cwd: Path, expect_file: Path, timeout: float) -> bool:
    """One headless-claude 'implement the method' turn. Success = the expected
    entrypoint file appeared (run_headless's own contract). The loop re-runs the
    smoke command to judge whether the impl actually works; this only reports that
    the agent produced the entrypoint."""
    res = run_headless(prompt=prompt, allowed_tools=["Read", "Write", "Edit", "Bash"],
                       cwd=cwd, expect_file=expect_file, timeout=timeout)
    return res.ok


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
    # Resolve seams at call time so monkeypatching the module globals takes effect.
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
        return SetupResult(ok=False,
                           error=f"from-scratch setup crashed: {e!r}; see setup_log/")


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
    """The scaffold→smoke→retry loop with seams already resolved. Wrapped by
    run_fromscratch_setup so any exception becomes ok=False rather than a crash."""
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
                # smoke genuinely passed; freeze/snapshot are best-effort and must
                # NOT turn a success into a crash.
                try:
                    snapshot = assemble_snapshot(freeze_env(env_dir))
                except Exception:
                    snapshot = assemble_snapshot({})
                try:
                    (rd.root / "env_snapshot.json").write_text(json.dumps(snapshot, indent=2))
                except OSError:
                    pass  # snapshot is best-effort; the impl genuinely ran
                return SetupResult(ok=True, env_snapshot=snapshot, patches=patches)
        else:
            _write_log(rd, f"scaffold_{attempt}.log",
                       f"scaffold turn {attempt} did not produce {_ENTRYPOINT}")
        if attempt >= max_retries:
            return SetupResult(ok=False, patches=patches,
                               error=f"impl smoke still failing after {max_retries} "
                                     f"retries; see setup_log/")
        attempt += 1


def make_fromscratch_setup_executor(*, manager: str = "uv", max_retries: int = 6,
                                    timeout_s: float = 3600.0
                                    ) -> Callable[[RunDir, Spec], SetupResult]:
    """Build the executor(rd, spec) the pipeline injects into run_setup on the
    from-scratch path."""
    def executor(rd: RunDir, spec: Spec) -> SetupResult:
        return run_fromscratch_setup(rd, spec, manager=manager,
                                     max_retries=max_retries, timeout_s=timeout_s)

    return executor


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
