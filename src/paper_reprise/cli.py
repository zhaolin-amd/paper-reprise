"""paper-reprise CLI: run / resume / report."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import click

from paper_reprise.grade import grade_claim
from paper_reprise.report import render_reports
from paper_reprise.rundir import RunDir


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


@click.group()
def cli() -> None:
    """Reproduce quantization paper results."""


@cli.command()
@click.argument("input_arg")
@click.option("--base-dir", default="runs", help="where run dirs are created")
@click.option("--yes", is_flag=True, help="auto-approve all gates (non-interactive)")
def run(input_arg: str, base_dir: str, yes: bool) -> None:
    """Run the reproduction pipeline for a paper (arxiv id or url)."""
    from paper_reprise.pipeline import run_pipeline

    def approve_spec(spec):
        if yes:
            return True
        click.echo(f"\nExtracted {len(spec.claims)} claims. Review spec.yaml.")
        return click.confirm("Approve spec and continue?", default=True)

    def approve_plan(plan):
        if yes:
            return True
        click.echo(f"\nPlan flagged: {plan.decision_reason}")
        return click.confirm("Proceed anyway?", default=False)

    def fetch_sources(rd, arxiv_id, url):
        click.echo(f"[ingest] {arxiv_id} (source fetch deferred to Plan 2)")

    def run_executor(claim, artifact, claim_dir):
        raise RuntimeError("real GPU executor not implemented (Plan 2)")

    result = run_pipeline(
        input_arg=input_arg, base_dir=Path(base_dir), timestamp=_timestamp(),
        available_hardware=[], approve_spec=approve_spec, approve_plan=approve_plan,
        fetch_sources=fetch_sources, setup_executor=None, run_executor=run_executor,
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

    artifacts = {a.id: a for a in spec.artifacts}
    from paper_reprise.models import RunResult
    grades, runs = [], []
    for c in spec.claims:
        log = rd.claim_dir(c.id) / "stdout.log"
        rr = RunResult(claim_id=c.id, command=c.eval_protocol.command,
                       stdout_path=str(log),
                       status="ran" if log.exists() else "blocked",
                       block_reason=None if log.exists() else "no stdout.log")
        runs.append(rr)
        # PLAN-2 TODO: actual_config={} forces the faithfulness check to pass vacuously
        # on re-render, so `report` can never detect a config divergence. Persist each
        # run's actual_config to the run dir (e.g. runs/<claim_id>/actual_config.json)
        # and read it back here once the real executor records it.
        grades.append(grade_claim(c, artifacts[c.artifact], rr, actual_config={}))

    zh, en = render_reports(spec, ingest, grades, runs, env={}, patches=[])
    (rd.root / "report.zh.md").write_text(zh)
    (rd.root / "report.en.md").write_text(en)
    click.echo(f"Re-rendered: {rd.root}/report.zh.md")


if __name__ == "__main__":
    cli()
