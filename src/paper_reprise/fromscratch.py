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
import re
import time
from pathlib import Path
from typing import Callable

from paper_reprise.headless import run_headless
from paper_reprise.models import Artifact, Claim, Spec
from paper_reprise.modelpaths import resolved_command, with_gpus, with_tasks
from paper_reprise.rundir import RunDir
from paper_reprise.runexec import (
    EvalFailed,
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

# The redacted spec the agent is pointed at (expected/tolerance/source stripped).
# The full spec.yaml is never named to the agent — see _SCAFFOLD_TEMPLATE.
_PUBLIC_SPEC = "spec.public.yaml"

# A smoke run that exits 0 must also PRINT a parseable metric line; a silent exit 0
# (e.g. `--smoke) exit 0`) proves nothing computed and must not pass setup. Requires
# a STANDALONE `metric: number` line (single-word name, value is the whole line) —
# the format the prompt asks for. Anchored so diagnostic prose like `exit code: 0`,
# `batch size: 8` or a `foo.py:123` traceback line does NOT count as a metric.
_METRIC_LINE = re.compile(
    r"^[ \t]*[A-Za-z][A-Za-z0-9_]*[ \t]*:[ \t]*[-+]?\d+(?:\.\d+)?[ \t]*$",
    re.MULTILINE,
)


def _smoke_reported_metric(output: str) -> bool:
    """True iff the smoke output contains at least one `metric: number` line."""
    return bool(_METRIC_LINE.search(output))


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
`{public_spec}` (artifacts = quantized products, claims = one metric each). The \
paper's expected values and tolerances are intentionally REDACTED from it. \
Implement exactly the method and eval protocol the spec describes.

Expose EXACTLY ONE runnable entrypoint `{entrypoint}` that:
  - takes a single argument: a claim id (e.g. `c1`), or `--smoke` for a tiny-scale \
self-test (a few samples, batch 1) used only to prove the code runs;
  - quantizes per the claim's artifact config and runs its eval protocol;
  - prints the resulting metric value to stdout in a parseable form \
(e.g. `perplexity: 5.80`);
  - reads these OVERRIDE env vars (so the operator can steer a run without editing \
code): use `${{PAPER_REPRISE_TASKS:-<the claim's eval tasks>}}` for the eval task \
list, `${{PAPER_REPRISE_GPUS:-1}}` for the GPU/process count, and \
`$PAPER_REPRISE_MODEL` (exported) for the base-model path — each falling back to the \
spec's value when unset.

HONESTY RULES (mandatory):
  - Do NOT fabricate, invent, or hard-code any result number. The entrypoint must \
COMPUTE the metric. A run that cannot compute must exit non-zero, never print a \
made-up value.
  - Do NOT read the paper's expected values or tolerances to shortcut the result. \
`{public_spec}` is the only spec you need; do NOT read any other spec file \
(e.g. `spec.yaml`) — the expected numbers are deliberately withheld.

For EACH file you create under `impl/`, append ONE line describing what it \
implements to `{patch_note}` (create the file; one line per file). When `impl/` \
and `{entrypoint}` exist and `--smoke` runs, you are done."""

# Appended when the spec lists prerequisite-method repos: the current paper builds on
# them, so they are offered as READ-ONLY references to disambiguate details the paper
# leaves underspecified — paper stays source of truth, never a way to back into a number.
_SCAFFOLD_REFERENCES_TEMPLATE = """

REFERENCE REPOS (read-only) — this paper's method builds on the prior method(s) below. \
You MAY `git clone` and read them (e.g. into a `refs/` dir; you have Bash) to disambiguate \
details this paper leaves underspecified:
{refs}
Rules for using them:
  - The PAPER is the source of truth. Where the paper gives an explicit definition or \
formula, implement EXACTLY that — even if the reference repo does it differently (the \
reference may use a different convention than this paper restates). Note in {patch_note} \
any place you followed the paper over a reference repo.
  - Use them ONLY to clarify HOW to implement; NEVER to obtain or back into a result \
number. The honesty rules above still hold.
  - They are optional aids, not required reading."""


def _references_block(spec: Spec, patch_note_path: str) -> str:
    """The read-only reference-repo section, or "" when the spec lists none."""
    if not spec.references:
        return ""
    lines = []
    for r in spec.references:
        note = f" — {r.note}" if r.note else ""
        lines.append(f"  - {r.method}: {r.repo_url}{note}")
    return _SCAFFOLD_REFERENCES_TEMPLATE.format(refs="\n".join(lines),
                                                patch_note=patch_note_path)


# Appended only on retry turns: the prior smoke run failed, so hand the agent the
# failing command + its captured output (the traceback) to debug — mirrors
# setuploop.build_fixer_prompt. The initial scaffold turn has no failure context.
_SCAFFOLD_FAILURE_TEMPLATE = """

The previous smoke attempt FAILED. Fix `impl/` so the SAME smoke command can run.

Failing command:
    {command}

Captured output (read the traceback):
{output}"""


def build_scaffold_prompt(
    rd: RunDir,
    spec: Spec,
    patch_note_path: str,
    *,
    failure: tuple[str, str] | None = None,
) -> str:
    """Build the headless-claude instruction to implement the paper's method as a
    self-contained `impl/` with one runnable entrypoint. Pure string builder.

    `patch_note_path` is the PER-TURN note path (e.g. `setup_patches/scaffold_0.txt`)
    embedded in the prompt so each turn's notes land in a distinct file and
    collect_new_patches_scaffold can capture every turn. `failure`, when given, is the
    (failing smoke command, captured output) from the prior turn — embedded so the
    agent can debug. None on the initial scaffold (no failure yet)."""
    methods = ", ".join(sorted({a.method for a in spec.artifacts})) or "the paper's method"
    prompt = _SCAFFOLD_TEMPLATE.format(
        methods=methods, entrypoint=_ENTRYPOINT, patch_note=patch_note_path,
        public_spec=_PUBLIC_SPEC,
    )
    prompt += _references_block(spec, patch_note_path)
    if failure is not None:
        command, output = failure
        prompt += _SCAFFOLD_FAILURE_TEMPLATE.format(command=command, output=output)
    return prompt


def collect_new_patches_scaffold(patches_dir: Path, seen: set[str]) -> list[str]:
    """Set-diff scaffold_*.txt patch notes — the from-scratch analogue of
    setuploop.collect_new_patches (which globs patch_*.txt)."""
    new: list[str] = []
    for p in sorted(patches_dir.glob("scaffold_*.txt")):
        if p.name in seen:
            continue
        seen.add(p.name)
        new.append(p.read_text().strip())
    return new


def _impl_tree_state(impl_dir: Path) -> dict[str, tuple[int, int]]:
    """Map file -> (mtime_ns, size) under impl_dir, for detecting whether a scaffold
    turn actually touched the tree. Empty when impl_dir does not exist yet."""
    state: dict[str, tuple[int, int]] = {}
    if not impl_dir.exists():
        return state
    for p in impl_dir.rglob("*"):
        if p.is_file():
            st = p.stat()
            state[str(p)] = (st.st_mtime_ns, st.st_size)
    return state


def _run_scaffold(prompt: str, cwd: Path, expect_file: Path, timeout: float) -> bool:
    """One headless-claude 'implement the method' turn. Success = the entrypoint
    exists AND this turn actually modified `impl/`.

    run_headless's own contract keys success on the entrypoint FILE existing, but
    once an earlier turn created it that file lingers: a later turn that crashed,
    timed out, or did nothing would still look 'produced', so the loop would burn a
    smoke retry on byte-identical code (and the 'did not produce' branch would be
    dead). We snapshot impl/ before and after and require a real change, so a no-op
    turn is reported as not-produced. The loop still re-runs smoke to judge whether
    the changed impl actually works."""
    impl_dir = expect_file.parent
    before = _impl_tree_state(impl_dir)
    res = run_headless(prompt=prompt, allowed_tools=["Read", "Write", "Edit", "Bash"],
                       cwd=cwd, expect_file=expect_file, timeout=timeout)
    after = _impl_tree_state(impl_dir)
    return res.ok and after != before


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
    # Resolve the model the same way the per-claim eval does (export PAPER_REPRISE_MODEL
    # / {model} substitution), so the smoke proves the SAME conditions — else an impl
    # that locates the model via $PAPER_REPRISE_MODEL would fail smoke but pass eval.
    # Mirrors setuploop._run_loop_body's smoke resolution on the official path.
    if spec.claims:
        artifacts = {a.id: a for a in spec.artifacts}
        art = artifacts.get(spec.claims[0].artifact)
        if art is not None:
            smoke_cmd = resolved_command(smoke_cmd, art.base_model)
    entrypoint = rd.root / _ENTRYPOINT
    start = now()
    seen_patches: set[str] = set()
    patches: list[str] = []
    # Carries the prior turn's failing (command, output) into the next prompt so the
    # agent can debug the traceback; None on the initial scaffold (no failure yet).
    failure: tuple[str, str] | None = None

    # --- redacted spec the agent is allowed to read (honesty barrier, design §6) ---
    rd.write_public_spec(spec)

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
        # Build the prompt PER TURN: a per-turn note path (so each turn's notes land
        # in a distinct file collect_new_patches_scaffold can capture) and, on retries,
        # the prior smoke failure so the agent debugs the traceback (Finding 1+2).
        note_rel = f"setup_patches/scaffold_{attempt}.txt"
        prompt = build_scaffold_prompt(rd, spec, note_rel, failure=failure)
        produced = run_scaffold(prompt, rd.root, entrypoint, _SCAFFOLD_TIMEOUT_S)
        patches.extend(collect_new_patches_scaffold(rd.setup_patches_dir, seen_patches))
        if produced:
            code, out = run_smoke(smoke_cmd, rd.root, env_dir)
            # An exit-0 smoke that printed no metric proves nothing computed — treat
            # it as a failure so the next turn must make the entrypoint emit one,
            # rather than freezing a silent no-op impl as a success (honesty gate).
            if code == 0 and not _smoke_reported_metric(out):
                code = 1
                out = ("smoke exited 0 but printed no parseable metric line "
                       "(expected e.g. `perplexity: 5.80`); the entrypoint must "
                       "COMPUTE and print the metric, not exit silently.\n\n" + out)
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
            # smoke ran but failed: hand the traceback to the next scaffold turn.
            failure = (smoke_cmd, out)
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
    tasks: str | None = None,
    gpus: int | None = None,
    run_eval: Callable[[str, Path, Path, Path], tuple[int, str]] | None = None,
    detect_gpu: Callable[[], str] | None = None,
    now: Callable[[], float] | None = None,
) -> Callable[[Claim, Artifact, Path], dict]:
    """Build the executor(claim, artifact, claim_dir) -> dict that run_claims injects
    on the from-scratch path. Runs the scaffolded entrypoint for the claim in the
    setup-built env (cwd = the run root, where impl/ lives), persists raw stdout +
    the resolved actual_config, and returns run metadata. A non-zero exit raises so
    run_claims marks the claim BLOCKED (the eval did not run — not 'failed to
    reproduce'). Mirrors runexec.make_run_executor, reusing its seams.

    `tasks`/`gpus` (from `run/resume --tasks --gpus`) export PAPER_REPRISE_TASKS /
    PAPER_REPRISE_GPUS for the impl entrypoint, same as the official path."""
    run_eval = run_eval or _run_eval
    detect_gpu = detect_gpu or _detect_gpu
    now = now or time.monotonic

    def executor(claim: Claim, artifact: Artifact, claim_dir: Path) -> dict:
        root, env_dir, _repo_dir = _rundir_paths(claim_dir)
        command = with_gpus(with_tasks(
            resolved_command(fromscratch_eval_command(claim), artifact.base_model), tasks), gpus)
        log_path = claim_dir / "stdout.log"
        gpu = detect_gpu()
        start = now()
        code, _out = run_eval(command, root, env_dir, log_path)
        minutes = (now() - start) / 60.0

        actual_config = resolve_actual_config(claim, artifact)
        (claim_dir / "actual_config.json").write_text(json.dumps(actual_config, indent=2))

        if code != 0:
            raise EvalFailed(command, f"from-scratch eval exited {code}; see {log_path}")

        return {
            "command": command,
            "stdout_path": str(log_path),
            "actual_config": actual_config,
            "gpu": gpu,
            "seed": extract_seed(command),
            "minutes": minutes,
        }

    return executor
