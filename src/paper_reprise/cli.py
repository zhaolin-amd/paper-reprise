"""paper-reprise CLI: run / resume / report."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from paper_reprise.models import Spec

from paper_reprise.grade import grade_claim
from paper_reprise.report import render_reports
from paper_reprise.rundir import RunDir

from paper_reprise.fetch import (
    fetch_arxiv_title,
    make_fetch_sources,
    resolve_arxiv_id,
    short_name,
)
from paper_reprise.fromscratch import (
    make_fromscratch_run_executor,
    make_fromscratch_setup_executor,
)
from paper_reprise.ingest import normalize_input
from paper_reprise.provider import make_run_dispatcher, make_setup_dispatcher
from paper_reprise.runexec import detect_available_hardware, make_run_executor
from paper_reprise.setupstage import make_setup_executor


def _setup_executor():
    """The setup executor the pipeline injects: a dispatcher that routes to the
    official path when a repo was cloned, else the from-scratch path (design §6)."""
    return make_setup_dispatcher(
        official=make_setup_executor(),
        fromscratch=make_fromscratch_setup_executor())


def _echo_cleaned(cleaned: list) -> None:
    """Report freed space from post-run model cleanup (records were kept)."""
    if not cleaned:
        return
    gb = sum(sz for _, sz in cleaned) / 1e9
    click.echo(f"Cleaned {len(cleaned)} exported model file(s), freed {gb:.1f} GB "
               f"(records kept; pass --keep-models to keep the model).")


def _run_executor(tasks: str | None = None, gpus: int | None = None):
    """The run executor the pipeline injects: the same repo-presence dispatch.
    `tasks`/`gpus` (from --tasks/--gpus) override the lm-eval task list and GPU
    count via PAPER_REPRISE_TASKS / PAPER_REPRISE_GPUS."""
    return make_run_dispatcher(
        official=make_run_executor(tasks=tasks, gpus=gpus),
        fromscratch=make_fromscratch_run_executor(tasks=tasks, gpus=gpus))


def _claim_block(i: int, claim, artifacts: dict) -> str:
    """A multi-line display block for one claim — the COMPLETE chain the user is
    choosing: model + full quant config + eval protocol + target + hardware. A claim
    is reproduced as model×config, so the config is shown explicitly (not abbreviated),
    and paper-agnostically (the whole quant_config dict is dumped)."""
    art = artifacts.get(claim.artifact)
    model = art.base_model if art else "?"
    method = art.method if art else "?"
    qc = ", ".join(f"{k}: {v}" for k, v in art.quant_config.items()) if art else "?"
    ep = claim.eval_protocol
    fewshot = f", few_shot={ep.few_shot}" if ep.few_shot else ""
    seqlen = f", seqlen={ep.seqlen}" if ep.seqlen else ""
    hw = claim.hardware or "—"
    return (
        f"  [{i}] {claim.id}   ({method})\n"
        f"        model:  {model}\n"
        f"        quant:  {qc}\n"
        f"        eval:   {ep.metric} on {ep.dataset}{fewshot}{seqlen}\n"
        f"        target: {claim.expected:g} ±{claim.tolerance:g}   hw: {hw}"
    )


def spec_selection_prompt(spec: "Spec", label: str) -> bool:
    """Print the extracted claims (model + full config) and ask the user to select
    which to reproduce. Mutates spec.claims and spec.artifacts in-place to keep
    only the chosen subset and prune orphaned artifacts. Returns False to abort."""
    claims = spec.claims
    artifacts = {a.id: a for a in spec.artifacts}

    click.echo(f"\nExtracted {len(claims)} claims from {label} — "
               f"pick which to reproduce (model × config is the unit):\n")
    for i, c in enumerate(claims, 1):
        click.echo(_claim_block(i, c, artifacts))
        click.echo("")

    raw = click.prompt(
        '\nEnter numbers (e.g. "1 3"), "all", or "q" to abort',
        default="all",
    )
    raw = raw.strip().lower()
    if raw == "q":
        click.echo("Aborted.")
        return False
    if raw == "all":
        indices = set(range(len(claims)))
    else:
        chosen = []
        for tok in raw.split():
            if tok.isdigit():
                n = int(tok)
                if 1 <= n <= len(claims):
                    chosen.append(n - 1)
        indices = set(chosen)

    if not indices:
        click.echo("No valid claims selected — aborting.")
        return False

    selected_claims = [claims[i] for i in sorted(indices)]
    referenced_artifacts = {c.artifact for c in selected_claims}
    selected_artifacts = [a for a in spec.artifacts if a.id in referenced_artifacts]

    spec.claims = selected_claims
    spec.artifacts = selected_artifacts
    click.echo(f"Selected {len(selected_claims)} claim(s). Continuing…")
    return True


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


@click.group()
def cli() -> None:
    """Reproduce quantization paper results."""


@cli.command()
@click.argument("input_arg")
@click.option("--base-dir", default="runs", help="where run dirs are created")
@click.option("--yes", is_flag=True,
              help="skip interactive claim selection and reproduce all claims end to end")
@click.option("--tasks", default=None,
              help="override the eval task list (comma-separated) via PAPER_REPRISE_TASKS")
@click.option("--gpus", type=int, default=None,
              help="override the GPU count via PAPER_REPRISE_GPUS (how many, not which)")
@click.option("--clean-models/--keep-models", default=False,
              help="delete the exported model weights right after THIS verified run "
                   "[default: keep — use `paper-reprise clean <run_dir>` when you're "
                   "done running, so multiple runs don't re-quantize]")
def run(input_arg: str, base_dir: str, yes: bool, tasks: str | None, gpus: int | None,
        clean_models: bool) -> None:
    """Run the reproduction pipeline for a paper (arxiv id, url, or title).

    By default presents the extracted claims interactively so you can choose which
    to reproduce, then continues the pipeline. Pass --yes to skip selection and
    reproduce all claims end to end.
    """
    from paper_reprise.pipeline import run_pipeline

    # If the input isn't a recognizable arxiv id/url, treat it as a title and
    # resolve it via the arxiv API before handing off to the pipeline.
    try:
        normalize_input(input_arg)
    except ValueError:
        resolved = resolve_arxiv_id(input_arg)
        if resolved is None:
            raise click.ClickException(f"could not resolve title to an arxiv id: {input_arg}")
        click.echo(f"[resolve] '{input_arg}' → {resolved}")
        input_arg = resolved

    def approve_plan(plan):
        if yes:
            return True
        click.echo(f"\nPlan flagged: {plan.decision_reason}")
        return click.confirm("Proceed anyway?", default=False)

    arxiv_id, _url = normalize_input(input_arg)
    _title = fetch_arxiv_title(arxiv_id)
    paper_name = short_name(_title) if _title else None

    display_label = paper_name or arxiv_id

    def approve_spec(spec):
        if yes:
            return True
        return spec_selection_prompt(spec, display_label)

    result = run_pipeline(
        input_arg=input_arg, base_dir=Path(base_dir), timestamp=_timestamp(),
        paper_name=paper_name,
        available_hardware=detect_available_hardware(),
        approve_spec=approve_spec, approve_plan=approve_plan,
        fetch_sources=make_fetch_sources(), setup_executor=_setup_executor(),
        run_executor=_run_executor(tasks, gpus), clean_models=clean_models,
    )
    if result.aborted_at == "spec-approval":
        click.echo(f"\nAborted at claim selection. Run dir: {result.root}")
        click.echo(f"To pick again: paper-reprise run {input_arg}")
    elif result.aborted_at:
        click.echo(f"Aborted at: {result.aborted_at}")
    else:
        _echo_cleaned(result.cleaned)
        click.echo(f"Done. Report: {result.root}/report.zh.md")


@cli.command()
@click.argument("run_dir")
@click.option("--yes", is_flag=True, help="auto-approve gates (non-interactive)")
@click.option("--tasks", default=None,
              help="override the eval task list (comma-separated) via PAPER_REPRISE_TASKS")
@click.option("--gpus", type=int, default=None,
              help="override the GPU count via PAPER_REPRISE_GPUS (how many, not which)")
@click.option("--clean-models/--keep-models", default=False,
              help="delete the exported model weights right after THIS verified run "
                   "[default: keep — use `paper-reprise clean <run_dir>` when you're "
                   "done running, so multiple runs don't re-quantize]")
def resume(run_dir: str, yes: bool, tasks: str | None, gpus: int | None,
           clean_models: bool) -> None:
    """Continue an existing run from its (possibly edited) spec.yaml."""
    from paper_reprise.pipeline import resume_pipeline

    def approve_plan(plan):
        if yes:
            return True
        click.echo(f"\nPlan flagged: {plan.decision_reason}")
        return click.confirm("Proceed anyway?", default=False)

    result = resume_pipeline(
        Path(run_dir), available_hardware=detect_available_hardware(),
        approve_plan=approve_plan,
        setup_executor=_setup_executor(), run_executor=_run_executor(tasks, gpus),
        clean_models=clean_models,
    )
    if result.aborted_at:
        click.echo(f"Aborted at: {result.aborted_at}")
    else:
        _echo_cleaned(result.cleaned)
        click.echo(f"Done. Report: {result.root}/report.zh.md")


@cli.command()
@click.argument("run_dir")
def report(run_dir: str) -> None:
    """Re-render reports from an existing run dir."""
    rd = RunDir.open(Path(run_dir))
    spec = rd.read_spec()
    ingest = rd.read_ingest()
    if spec is None or ingest is None:
        raise click.ClickException("run dir missing spec.yaml or ingest.json")

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

    zh, en = render_reports(spec, ingest, grades, runs, env={}, patches=[])
    (rd.root / "report.zh.md").write_text(zh)
    (rd.root / "report.en.md").write_text(en)
    click.echo(f"Re-rendered: {rd.root}/report.zh.md")


@cli.command()
@click.argument("run_dir")
@click.option("--env/--no-env", default=True,
              help="also remove the per-run environment (env/ and repo/.venv) [default: yes]")
def clean(run_dir: str, env: bool) -> None:
    """Free a finished run's regenerable artifacts (exported model weights, and by
    default the env), keeping all records. Run this once you're done with the run —
    `run`/`resume` no longer auto-delete, so repeated runs don't re-quantize."""
    rd = RunDir.open(Path(run_dir))
    removed = rd.clean_model_artifacts()
    if env:
        removed += rd.clean_env()
    if not removed:
        click.echo("Nothing to clean (no model weights or env found).")
        return
    gb = sum(sz for _, sz in removed) / 1e9
    for rel, sz in removed:
        click.echo(f"  removed {rel}  ({sz/1e9:.2f} GB)")
    click.echo(f"Freed ~{gb:.1f} GB from {rd.root} (records kept).")


if __name__ == "__main__":
    cli()
