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

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable

from paper_reprise.headless import run_headless
from paper_reprise.models import Spec
from paper_reprise.modelpaths import hf_env_overlay, resolved_command
from paper_reprise.runexec import _reap_process_group
from paper_reprise.rundir import RunDir
from paper_reprise.setupstage import SetupResult

# Smoke runs must be cheap; guard against a hung command (design §4.1).
_SMOKE_TIMEOUT_S = 1800

# Guardrails so env creation, the headless fixer, and pip freeze can't hang forever.
_CREATE_ENV_TIMEOUT_S = 1800
_FIXER_TIMEOUT_S = 1800
_FREEZE_TIMEOUT_S = 300

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


def assemble_snapshot(freeze: dict) -> dict:
    """Normalize a raw freeze dict into the report's expected keys, keeping the
    full pip freeze. Missing versions become 'unknown' (never silently dropped)."""
    return {
        "torch": freeze.get("torch") or "unknown",
        "transformers": freeze.get("transformers") or "unknown",
        "cuda": freeze.get("cuda") or "unknown",
        "pip_freeze": freeze.get("pip_freeze", ""),
    }


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


def _write_log(rd: RunDir, name: str, text: str) -> None:
    (rd.setup_log_dir / name).write_text(text)


def _create_env(env_dir: Path, manager: str) -> tuple[int, str]:
    """Create a conda/uv env at env_dir. Returns (exit_code, combined log).

    Idempotent: if an interpreter already exists at env_dir (e.g. a re-`resume`
    after a prior run), reuse it instead of failing — `uv venv` and `conda create`
    both refuse to clobber an existing env. The smoke test still re-validates it."""
    if (env_dir / "bin" / "python").exists():
        return 0, f"env already exists at {env_dir}; reusing"
    if manager == "uv":
        cmd = ["uv", "venv", str(env_dir)]
    else:
        cmd = ["conda", "create", "-y", "-p", str(env_dir), "python=3.11"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=_CREATE_ENV_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return 124, "env creation timed out"
    except FileNotFoundError as e:
        return 127, str(e)
    return proc.returncode, (proc.stdout + proc.stderr)


def _run_smoke(command: str, cwd: Path, env_dir: Path) -> tuple[int, str]:
    """Run the smoke command inside the repo, USING the created env.

    The created venv/conda env is activated by prepending its bin/ to PATH and
    setting VIRTUAL_ENV, so `python`/`pip` in the command resolve to env_dir's
    interpreter (not the ambient one — otherwise building the env is pointless).
    Returns (exit_code, combined output).

    NOTE: uses a temp file for output rather than capture_output=True (pipes).
    Some repos (e.g. those that `disown` a background vLLM server) spawn
    background processes that inherit the pipe fds and hold them open after the
    shell exits, causing subprocess.run to block forever waiting for pipe EOF.
    Writing to a temp file avoids this: the shell's stdout/stderr go to the file;
    disowned children may keep writing there but the shell exits independently.

    Runs in a NEW SESSION and reaps the whole process group on return/timeout, so a
    background server the smoke command leaves behind (e.g. vLLM + EngineCore) is
    not leaked — same lifecycle guarantee as the eval path (runexec._run_eval).
    """
    import tempfile
    env = dict(os.environ)
    env["PATH"] = f"{env_dir / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = str(env_dir)
    # Read models from the shared cache / download missing ones to scratch.
    env.update(hf_env_overlay())
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as out_f:
        out_path = Path(out_f.name)
    proc = None
    try:
        with open(out_path, "w") as out_f:
            proc = subprocess.Popen(command, shell=True, executable="/bin/bash",
                                    cwd=str(cwd), env=env,
                                    stdout=out_f, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            code = proc.wait(timeout=_SMOKE_TIMEOUT_S)
        output = out_path.read_text(errors="replace")
        return code, output
    except subprocess.TimeoutExpired:
        output = out_path.read_text(errors="replace") if out_path.exists() else ""
        return 124, f"smoke command timed out after {_SMOKE_TIMEOUT_S}s\n{output[-2000:]}"
    finally:
        if proc is not None:
            _reap_process_group(proc.pid)
            try:
                proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, ChildProcessError, OSError):
                pass
        try:
            out_path.unlink()
        except OSError:
            pass


def _freeze_env(env_dir: Path) -> dict:
    """Capture pip freeze + torch/transformers/CUDA versions from env_dir.

    Best-effort: a hung/missing python binary must not break a passing smoke run,
    so any failure degrades to an empty freeze (→ assemble_snapshot fills 'unknown').
    """
    try:
        proc = subprocess.run([str(env_dir / "bin" / "python"), "-m", "pip", "freeze"],
                              capture_output=True, text=True, timeout=_FREEZE_TIMEOUT_S)
    except Exception:
        return {"pip_freeze": ""}
    freeze_text = proc.stdout
    versions: dict = {"pip_freeze": freeze_text}
    for line in freeze_text.splitlines():
        for pkg in ("torch", "transformers"):
            if line.lower().startswith(f"{pkg}=="):
                versions[pkg] = line.split("==", 1)[1].strip()
    try:
        cuda = subprocess.run(
            [str(env_dir / "bin" / "python"), "-c",
             "import torch; print(torch.version.cuda)"],
            capture_output=True, text=True, timeout=_FREEZE_TIMEOUT_S)
        if cuda.returncode == 0 and cuda.stdout.strip() not in ("", "None"):
            versions["cuda"] = cuda.stdout.strip()
    except Exception:
        pass  # cuda probe is best-effort
    return versions


def _run_fixer(prompt: str, cwd: Path, patch_note: Path) -> None:
    """One headless-claude fix turn. Success is judged by the loop re-running the
    smoke command, so we ignore the HeadlessResult here (the patch note is the
    only artifact we read back, via collect_new_patches)."""
    run_headless(prompt=prompt, allowed_tools=["Read", "Write", "Edit", "Bash"],
                 cwd=cwd, expect_file=patch_note, timeout=_FIXER_TIMEOUT_S)


def run_setup_loop(
    rd: RunDir,
    spec: Spec,
    *,
    manager: str = "uv",
    max_retries: int = 6,
    timeout_s: float = 3600.0,
    now: Callable[[], float] = time.monotonic,
    create_env: Callable[[Path, str], tuple[int, str]] | None = None,
    run_smoke: Callable[[str, Path, Path], tuple[int, str]] | None = None,
    freeze_env: Callable[[Path], dict] | None = None,
    run_fixer: Callable[[str, Path, Path], None] | None = None,
) -> SetupResult:
    """Drive the env-debug loop until the smoke command passes once, or a guardrail
    (retry cap / timeout) is exceeded. On exhaustion: ok=False, full log handed off.

    Contract: this never lets an exception escape — any crash in the loop body is
    caught and reported as ok=False so the pipeline keeps going (design §4.1)."""
    # Resolve seams at call time so monkeypatching the module globals takes effect
    # (Python default args are bound at def time, so we can't default to the names).
    create_env = create_env or _create_env
    run_smoke = run_smoke or _run_smoke
    freeze_env = freeze_env or _freeze_env
    run_fixer = run_fixer or _run_fixer

    try:
        return _run_loop_body(rd, spec, manager=manager, max_retries=max_retries,
                              timeout_s=timeout_s, now=now, create_env=create_env,
                              run_smoke=run_smoke, freeze_env=freeze_env,
                              run_fixer=run_fixer)
    except Exception as e:  # never crash the pipeline; setup failure must be ok=False
        return SetupResult(ok=False, error=f"setup crashed: {e!r}; see setup_log/")


def _run_loop_body(
    rd: RunDir,
    spec: Spec,
    *,
    manager: str,
    max_retries: int,
    timeout_s: float,
    now: Callable[[], float],
    create_env: Callable[[Path, str], tuple[int, str]],
    run_smoke: Callable[[str, Path, Path], tuple[int, str]],
    freeze_env: Callable[[Path], dict],
    run_fixer: Callable[[str, Path, Path], None],
) -> SetupResult:
    """The actual env-debug loop, with seams already resolved. Wrapped by
    run_setup_loop so any exception here becomes ok=False rather than a crash."""
    env_dir = rd.root / "env"
    command = select_smoke_command(rd, spec)
    # Resolve the model the same way the real eval will, so the smoke proves the
    # SAME command (per the fixer prompt) — else a {model}/$PAPER_REPRISE_MODEL
    # reference would fail smoke but pass eval (false negative).
    if command and spec.claims:
        artifacts = {a.id: a for a in spec.artifacts}
        art = artifacts.get(spec.claims[0].artifact)
        if art is not None:
            command = resolved_command(command, art.base_model)
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
            # smoke genuinely passed; freeze/snapshot are best-effort and must NOT
            # turn a success into a crash.
            try:
                snapshot = assemble_snapshot(freeze_env(env_dir))
            except Exception:
                snapshot = assemble_snapshot({})  # unknown versions; smoke still passed
            try:
                (rd.root / "env_snapshot.json").write_text(json.dumps(snapshot, indent=2))
            except OSError:
                pass  # snapshot is best-effort; the env genuinely ran
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
