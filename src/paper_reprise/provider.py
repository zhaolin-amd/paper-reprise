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

from pathlib import Path
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
    official: Callable[[Claim, Artifact, Path], dict],
    fromscratch: Callable[[Claim, Artifact, Path], dict],
) -> Callable[[Claim, Artifact, Path], dict]:
    """Build the run executor the pipeline injects: routes per run dir, derived from
    the claim_dir, by the same repo-presence signal as setup."""
    def executor(claim: Claim, artifact: Artifact, claim_dir: Path) -> dict:
        _root, _env, repo_dir = _rundir_paths(claim_dir)
        present = repo_dir.is_dir() and any(repo_dir.iterdir())
        chosen = official if present else fromscratch
        return chosen(claim, artifact, claim_dir)

    return executor
