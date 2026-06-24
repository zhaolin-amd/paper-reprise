"""Deterministic orchestration of the 7 stages with two gates.

Gates and side-effecting stages are injected as callables so tests run offline
and the CLI can supply interactive prompts / real executors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from paper_reprise.grade import grade_claim
from paper_reprise.ingest import normalize_input
from paper_reprise.models import IngestInfo
from paper_reprise.planstage import build_plan
from paper_reprise.report import render_reports
from paper_reprise.rundir import RunDir
from paper_reprise.runstage import run_claims
from paper_reprise.setupstage import run_setup
from paper_reprise.specextract import extract_spec


@dataclass
class PipelineResult:
    root: Path
    aborted_at: Optional[str] = None
    cleaned: list = field(default_factory=list)   # (relpath, bytes) of removed model files


def run_pipeline(
    input_arg: str,
    base_dir: Path,
    timestamp: str,
    available_hardware: list[str],
    approve_spec: Callable,
    approve_plan: Callable,
    fetch_sources: Callable,
    setup_executor: Optional[Callable],
    run_executor: Callable,
    paper_name: Optional[str] = None,
    clean_models: bool = False,
) -> PipelineResult:
    # --- ingest ---
    arxiv_id, url = normalize_input(input_arg)
    rd = RunDir.create(base_dir, arxiv_id=arxiv_id, timestamp=timestamp, name=paper_name)
    fetch_sources(rd, arxiv_id, url)            # fills paper/ and repo/ (network)
    ingest = IngestInfo(arxiv_id=arxiv_id, source_url=url)
    rd.write_ingest(ingest)

    # --- specextract + gate 1 ---
    spec = extract_spec(rd)
    if spec is None:
        return PipelineResult(root=rd.root, aborted_at="specextract")
    if not approve_spec(spec):
        return PipelineResult(root=rd.root, aborted_at="spec-approval")
    rd.write_spec(spec)

    return _finish_pipeline(rd, spec, ingest, available_hardware=available_hardware,
                            approve_plan=approve_plan, setup_executor=setup_executor,
                            run_executor=run_executor, clean_models=clean_models)


def _finish_pipeline(rd, spec, ingest, *, available_hardware, approve_plan,
                     setup_executor, run_executor, clean_models=False) -> PipelineResult:
    # Redirect the repo's bulky output (exported model checkpoints) to scratch via a
    # symlink, before setup/run write anything — keeps runs/ (home) small. Best-effort.
    rd.link_repo_output_to_scratch()

    # --- plan + sentinel ---
    plan = build_plan(spec, available_hardware)
    rd.write_plan(plan)
    if plan.needs_user_decision and not approve_plan(plan):
        return PipelineResult(root=rd.root, aborted_at="plan")

    # --- setup ---
    setup = run_setup(rd, spec, executor=setup_executor)
    if not setup.ok:
        return PipelineResult(root=rd.root, aborted_at="setup")

    # --- run ---
    runs, actual_configs = run_claims(rd, spec, executor=run_executor)

    # --- grade (pure code, isolated) ---
    artifacts = {a.id: a for a in spec.artifacts}
    runs_by_claim = {r.claim_id: r for r in runs}
    grades = [grade_claim(c, artifacts[c.artifact], runs_by_claim[c.id],
                          actual_configs.get(c.id, {}))
              for c in spec.claims]

    # --- report ---
    ingest.repo = spec.repo
    zh, en = render_reports(spec, ingest, grades, runs, setup.env_snapshot,
                            patches=setup.patches)
    (rd.root / "report.zh.md").write_text(zh)
    (rd.root / "report.en.md").write_text(en)

    # --- cleanup: drop the exported quantized model (regenerable), keep records.
    # Only when something was actually verified (a non-BLOCKED grade), so a failed
    # run's model is left for debugging.
    cleaned: list = []
    if clean_models and any(g.verdict != "BLOCKED" for g in grades):
        cleaned = rd.clean_model_artifacts() + rd.clean_scratch_models()
    return PipelineResult(root=rd.root, aborted_at=None, cleaned=cleaned)


def resume_pipeline(run_dir, *, available_hardware, approve_plan,
                    setup_executor, run_executor, clean_models=False) -> PipelineResult:
    """Continue an existing run from the plan stage, using the spec.yaml already on
    disk (the user has reviewed/edited it — resuming IS the approval). Skips ingest
    and specextract."""
    rd = RunDir.open(Path(run_dir))
    spec = rd.read_spec()
    if spec is None:
        return PipelineResult(root=rd.root, aborted_at="no-spec")
    ingest = rd.read_ingest()
    if ingest is None:
        ingest = IngestInfo(arxiv_id=spec.paper,
                            source_url=f"https://arxiv.org/abs/{spec.paper}")
    return _finish_pipeline(rd, spec, ingest, available_hardware=available_hardware,
                            approve_plan=approve_plan, setup_executor=setup_executor,
                            run_executor=run_executor, clean_models=clean_models)
