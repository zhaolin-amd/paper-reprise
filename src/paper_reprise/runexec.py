"""Run stage: the real GPU executor (design §4.2).

Runs each claim's eval command inside the setup-built env, in the cloned repo,
persists raw stdout, and reports run metadata + the resolved actual_config. It
only RUNS and records — it never parses metrics or computes verdicts (grade does
that later, from the persisted output). Every real-world action (the eval
subprocess, GPU detection, the clock) is behind an injectable seam so the
orchestration is offline-testable.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from paper_reprise.models import Artifact, Claim

_SEED_PATTERNS = (
    r"--seed[=\s]+(\d+)",
    r"\bseed=(\d+)",
)

_EVAL_TIMEOUT_S = 7200


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


_NVIDIA_SMI_CANDIDATES = ("/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi")


def _detect_gpu() -> str:
    """Best-effort GPU label: nvidia-smi name, else CUDA_VISIBLE_DEVICES, else 'unknown'.
    Checks shutil.which first, then falls back to known absolute paths so it works even
    when nvidia-smi is not on PATH (common in some HPC environments)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        for candidate in _NVIDIA_SMI_CANDIDATES:
            if shutil.which(candidate) or __import__("pathlib").Path(candidate).is_file():
                smi = candidate
                break
    if smi:
        try:
            proc = subprocess.run(
                [smi, "--query-gpu=name", "--format=csv,noheader"],
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
