"""Render bilingual reproduction reports (zh + en) as two markdown strings.

Iron rules (design §5.2):
  - always show measured (raw) numbers, never paper numbers as substitute
  - every claim carries replay info (command/seed/gpu/commit/env)
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from paper_reprise.models import ClaimGrade, IngestInfo, RunResult, Spec
from paper_reprise.parsers import extract_results_table


def _summary(grades: list[ClaimGrade]) -> str:
    c = Counter(g.verdict for g in grades)
    return f"MATCH {c['MATCH']} / PARTIAL {c['PARTIAL']} / FAIL {c['FAIL']} / BLOCKED {c['BLOCKED']}"


def _env_line(ingest: IngestInfo, env: dict, spec: Spec | None = None) -> str:
    # ingest often doesn't carry the repo even when specextract found one — fall
    # back to spec.repo so a run against an official repo isn't mislabelled.
    repo = ingest.repo or (spec.repo if spec else None)
    repo_s = f"{repo.url}@{repo.commit}" if repo else "(no official repo)"

    def _v(key: str) -> str:
        val = str(env.get(key) or "").strip()
        return val if val and val.lower() != "unknown" else "?"

    return (f"repo: {repo_s} | torch {_v('torch')} / "
            f"transformers {_v('transformers')} / CUDA {_v('cuda')}")


def _artifact(spec: Spec, artifact_id: str):
    return next((a for a in spec.artifacts if a.id == artifact_id), None)


def _config_label(a) -> str:
    """Compact precision/config tag: 16-bit -> BF16, else INT<bits> (+ group size)."""
    if not a:
        return "?"
    bits = a.quant_config.get("wbits", "?")
    if bits == 16:
        return "BF16"
    gs = a.quant_config.get("group_size")
    return f"INT{bits}" + (f" G{gs}" if gs else "")


def _algorithm_label(a) -> str:
    """The quantization algorithm; an uncompressed (BF16/none) artifact has none -> "-"."""
    if not a:
        return "?"
    m = (a.method or "").strip()
    return "-" if m.lower() in ("none", "", "fp16", "bf16", "fp", "full", "baseline") else m


def _measured_cell(g, expected) -> str:
    """measured value annotated with its signed gap vs paper, e.g. 73.79(+0.08)."""
    if not g or g.measured is None:
        return "—"
    return f"{g.measured:.2f}({g.measured - expected:+.2f})"


def _table(spec, grades, header):
    """One row per claim (model × config × metric), with paper, measured(diff),
    verdict and reason all in a single table."""
    runs_by = {g.claim_id: g for g in grades}
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for c in spec.claims:
        g = runs_by.get(c.id)
        measured = _measured_cell(g, c.expected)
        verdict = g.verdict if g else "BLOCKED"
        reason = g.reason if g else "no grade"
        a = _artifact(spec, c.artifact)
        model = a.base_model if a else c.artifact
        lines.append(f"| {model} | {_config_label(a)} | {_algorithm_label(a)} | "
                     f"{c.eval_protocol.metric} | {c.expected:g} | {measured} | "
                     f"{verdict} | {reason} |")
    return "\n".join(lines)


def _replay(runs: list[RunResult]) -> str:
    if not runs:
        return "(none)"
    blocks = []
    for r in runs:
        meta = []
        if r.seed is not None:
            meta.append(f"seed {r.seed}")
        if r.gpu:
            meta.append(str(r.gpu))
        if r.minutes is not None:
            meta.append(f"{r.minutes:.1f} min")
        meta.append(f"`{r.stdout_path}`")
        # command in a fenced block (eval commands are multi-line shell — inline
        # backticks render as a broken blob)
        blocks.append(f"**{r.claim_id}** — " + " · ".join(meta)
                      + f"\n\n```bash\n{r.command.strip()}\n```")
    return "\n\n".join(blocks)


def _raw_scores(runs: list[RunResult]) -> str:
    """The harness's raw per-task results table (verbatim) behind each claim's
    averaged metric, read from its stdout — so a suite-average claim (e.g.
    acc_norm_avg) isn't a black box."""
    out = []
    for r in runs:
        tbl = ""
        try:
            p = Path(r.stdout_path)
            if p.exists():
                tbl = extract_results_table(p.read_text(errors="replace"))
        except OSError:
            pass
        if tbl:
            out.append(f"**{r.claim_id}**\n\n{tbl}")
    return "\n\n".join(out) if out else "(none)"


def _patches(patches: list[str]) -> str:
    return "\n".join(f"- {p}" for p in patches) if patches else "(none)"


def render_reports(spec: Spec, ingest: IngestInfo, grades: list[ClaimGrade],
                   runs: list[RunResult], env: dict, patches: list[str]) -> tuple[str, str]:
    title = ingest.title or ingest.arxiv_id
    summ = _summary(grades)
    envl = _env_line(ingest, env, spec)

    zh = f"""# 复现报告:{title} ({ingest.arxiv_id})
{envl}
判定汇总: {summ}

{_table(spec, grades, "| model | config | algorithm | metric | paper | 实测 | 判定 | 原因 |")}

## 各任务原始分数
{_raw_scores(runs)}

## 复算信息(每条 claim)
{_replay(runs)}

## Setup 改动留痕
{_patches(patches)}
"""

    en = f"""# Reproduction Report: {title} ({ingest.arxiv_id})
{envl}
Verdict summary: {summ}

{_table(spec, grades, "| model | config | algorithm | metric | paper | measured | verdict | reason |")}

## Per-task raw scores
{_raw_scores(runs)}

## Replay info (per claim)
{_replay(runs)}

## Setup patches
{_patches(patches)}
"""
    return zh, en
