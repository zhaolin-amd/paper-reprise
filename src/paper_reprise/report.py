"""Render bilingual reproduction reports (zh + en) as two markdown strings.

Iron rules (design §5.2):
  - always show measured (raw) numbers, never paper numbers as substitute
  - every claim carries replay info (command/seed/gpu/commit/env)
"""
from __future__ import annotations

from collections import Counter

from paper_reprise.models import ClaimGrade, IngestInfo, RunResult, Spec


def _summary(grades: list[ClaimGrade]) -> str:
    c = Counter(g.verdict for g in grades)
    return f"MATCH {c['MATCH']} / PARTIAL {c['PARTIAL']} / FAIL {c['FAIL']} / BLOCKED {c['BLOCKED']}"


def _env_line(ingest: IngestInfo, env: dict) -> str:
    repo = ingest.repo
    repo_s = f"{repo.url}@{repo.commit}" if repo else "(no official repo)"
    return (f"repo: {repo_s} | torch {env.get('torch','?')} / "
            f"transformers {env.get('transformers','?')} / CUDA {env.get('cuda','?')}")


def _artifact_label(spec: Spec, artifact_id: str) -> str:
    a = next((a for a in spec.artifacts if a.id == artifact_id), None)
    if not a:
        return artifact_id
    cfg = a.quant_config
    bits = cfg.get("wbits", "?")
    gs = cfg.get("group_size")
    return f"{a.base_model} W{bits}" + (f"G{gs}" if gs else "")


def _table(spec, grades, header):
    runs_by = {g.claim_id: g for g in grades}
    lines = [header, "|---|---|---|---|---|---|---|"]
    for c in spec.claims:
        g = runs_by.get(c.id)
        measured = "—" if not g or g.measured is None else f"{g.measured:.2f}"
        verdict = g.verdict if g else "BLOCKED"
        reason = g.reason if g else "no grade"
        label = _artifact_label(spec, c.artifact)
        lines.append(f"| {c.id} | {label} | {c.eval_protocol.metric} | "
                     f"{c.expected:g} | {measured} | {verdict} | {reason} |")
    return "\n".join(lines)


def _replay(runs: list[RunResult]) -> str:
    out = []
    for r in runs:
        out.append(f"- {r.claim_id}: `{r.command}` | seed {r.seed} | {r.gpu} | "
                   f"{r.minutes}min | {r.stdout_path}")
    return "\n".join(out) if out else "(none)"


def _patches(patches: list[str]) -> str:
    return "\n".join(f"- {p}" for p in patches) if patches else "(none)"


def render_reports(spec: Spec, ingest: IngestInfo, grades: list[ClaimGrade],
                   runs: list[RunResult], env: dict, patches: list[str]) -> tuple[str, str]:
    title = ingest.title or ingest.arxiv_id
    summ = _summary(grades)
    envl = _env_line(ingest, env)

    zh = f"""# 复现报告:{title} ({ingest.arxiv_id})
{envl}
判定汇总: {summ}

{_table(spec, grades, "| claim | 模型/配置 | 指标 | paper | 实测 | 判定 | 原因 |")}

## 复算信息(每条 claim)
{_replay(runs)}

## Setup 改动留痕
{_patches(patches)}
"""

    en = f"""# Reproduction Report: {title} ({ingest.arxiv_id})
{envl}
Verdict summary: {summ}

{_table(spec, grades, "| claim | model/config | metric | paper | measured | verdict | reason |")}

## Replay info (per claim)
{_replay(runs)}

## Setup patches
{_patches(patches)}
"""
    return zh, en
