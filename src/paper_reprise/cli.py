"""paper-reprise CLI: run / resume / report."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import click

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
from paper_reprise.runexec import make_run_executor
from paper_reprise.setupstage import make_setup_executor


def _setup_executor():
    """The setup executor the pipeline injects: a dispatcher that routes to the
    official path when a repo was cloned, else the from-scratch path (design §6)."""
    return make_setup_dispatcher(
        official=make_setup_executor(),
        fromscratch=make_fromscratch_setup_executor())


def _run_executor():
    """The run executor the pipeline injects: the same repo-presence dispatch."""
    return make_run_dispatcher(
        official=make_run_executor(),
        fromscratch=make_fromscratch_run_executor())


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


@click.group()
def cli() -> None:
    """Reproduce quantization paper results."""


@cli.command()
@click.argument("input_arg")
@click.option("--base-dir", default="runs", help="where run dirs are created")
@click.option("--yes", is_flag=True,
              help="skip the spec review and run straight through (non-interactive)")
def run(input_arg: str, base_dir: str, yes: bool) -> None:
    """Run the reproduction pipeline for a paper (arxiv id, url, or title).

    By default the run STOPS after extracting spec.yaml so you can review and edit
    which models/claims to reproduce, then continue with `resume`. Pass --yes to
    skip that review and run end to end.
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

    # Spec gate: by default require the user to review which models/claims to
    # reproduce. The run halts at "spec-approval" unless --yes is given.
    def approve_spec(spec):
        return yes

    def approve_plan(plan):
        if yes:
            return True
        click.echo(f"\nPlan flagged: {plan.decision_reason}")
        return click.confirm("Proceed anyway?", default=False)

    arxiv_id, _url = normalize_input(input_arg)
    _title = fetch_arxiv_title(arxiv_id)
    paper_name = short_name(_title) if _title else None
    result = run_pipeline(
        input_arg=input_arg, base_dir=Path(base_dir), timestamp=_timestamp(),
        paper_name=paper_name,
        available_hardware=[], approve_spec=approve_spec, approve_plan=approve_plan,
        fetch_sources=make_fetch_sources(), setup_executor=_setup_executor(),
        run_executor=_run_executor(),
    )
    if result.aborted_at == "spec-approval":
        spec = RunDir.open(result.root).read_spec()
        n = len(spec.claims) if spec else "?"
        click.echo(f"\nExtracted {n} claims into {result.root}/spec.yaml")
        click.echo("Review/edit which models & claims to reproduce, then continue with:")
        click.echo(f"  paper-reprise resume {result.root}")
    elif result.aborted_at:
        click.echo(f"Aborted at: {result.aborted_at}")
    else:
        click.echo(f"Done. Report: {result.root}/report.zh.md")


@cli.command()
@click.argument("run_dir")
@click.option("--yes", is_flag=True, help="auto-approve gates (non-interactive)")
def resume(run_dir: str, yes: bool) -> None:
    """Continue an existing run from its (possibly edited) spec.yaml."""
    from paper_reprise.pipeline import resume_pipeline

    def approve_plan(plan):
        if yes:
            return True
        click.echo(f"\nPlan flagged: {plan.decision_reason}")
        return click.confirm("Proceed anyway?", default=False)

    result = resume_pipeline(
        Path(run_dir), available_hardware=[], approve_plan=approve_plan,
        setup_executor=_setup_executor(), run_executor=_run_executor(),
    )
    if result.aborted_at:
        click.echo(f"Aborted at: {result.aborted_at}")
    else:
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


if __name__ == "__main__":
    cli()
