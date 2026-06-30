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
    return (f"MATCH {c['MATCH']} · PARTIAL {c['PARTIAL']} · "
            f"FAIL {c['FAIL']} · BLOCKED {c['BLOCKED']}")


def _repo_str(ingest: IngestInfo, spec: Spec | None = None) -> str:
    # ingest often doesn't carry the repo even when specextract found one — fall
    # back to spec.repo so a run against an official repo isn't mislabelled.
    repo = ingest.repo or (spec.repo if spec else None)
    return f"{repo.url}@{repo.commit}" if repo else "(no official repo)"


def _env_str(env: dict) -> str:
    # Only show env components we actually know — an unknown one is dropped rather than
    # rendered as "?" (e.g. a pure-numpy from-scratch run has no torch/transformers/CUDA).
    # CUDA (NVIDIA) and ROCm (AMD) are mutually exclusive per torch build; whichever the
    # snapshot captured is shown, the absent one is simply dropped.
    parts = []
    for label, key in (("torch", "torch"), ("transformers", "transformers"),
                       ("CUDA", "cuda"), ("ROCm", "rocm")):
        val = str(env.get(key) or "").strip()
        if val and val.lower() != "unknown":
            parts.append(f"{label} {val}")
    return " / ".join(parts)


def _title_line(prefix: str, title: str | None, arxiv_id: str) -> str:
    """`# <prefix>: <title> (<arxiv_id>)`, collapsed to just the id when there is no
    distinct title (avoids the ugly `2504.19874 (2504.19874)`)."""
    if title and title != arxiv_id:
        return f"# {prefix}: {title} ({arxiv_id})"
    return f"# {prefix}: {arxiv_id}"


def _meta_block(repo_str: str, env_str: str, summ: str,
                labels: tuple[str, str, str]) -> str:
    """The header metadata as a Markdown bullet list (repo / env / verdict), each on its
    own line. The env bullet is omitted entirely when nothing is known."""
    repo_l, env_l, verdict_l = labels
    lines = [f"- **{repo_l}:** {repo_str}"]
    if env_str:
        lines.append(f"- **{env_l}:** {env_str}")
    lines.append(f"- **{verdict_l}:** {summ}")
    return "\n".join(lines)


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


def _fmt_num(v: float) -> str:
    """Magnitude-adaptive number format. Accuracy/perplexity-scale values (|v|>=1) keep
    two decimals (73.79, 5.80); sub-1 metrics like quantization distortion keep
    significant figures instead of being crushed to 0.00 (0.3633, 0.001033, 3.54e-05)."""
    av = abs(v)
    if av == 0:
        return "0.00"
    if av >= 1:
        return f"{v:.2f}"
    if av >= 1e-3:
        return f"{v:.4g}"
    return f"{v:.3g}"


def _fmt_signed(d: float) -> str:
    """Signed gap with the same magnitude-adaptive precision (e.g. +0.02, -0.01,
    +8.3e-06)."""
    return ("+" if d >= 0 else "-") + _fmt_num(abs(d))


def _measured_cell(g, expected) -> str:
    """measured value annotated with its signed gap vs paper, e.g. 73.79(+0.08)."""
    if not g or g.measured is None:
        return "—"
    return f"{_fmt_num(g.measured)}({_fmt_signed(g.measured - expected)})"


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


def _run_meta(r: RunResult) -> str:
    meta = []
    if r.seed is not None:
        meta.append(f"seed {r.seed}")
    if r.gpu:
        meta.append(str(r.gpu))
    if r.minutes is not None:
        meta.append(f"{r.minutes:.1f} min")
    meta.append(f"`{r.stdout_path}`")
    return " · ".join(meta)


def _replay(spec, runs: list[RunResult]) -> str:
    """Replay script grouped per config (model × config × algorithm), not per claim:
    a config's metrics share one serve+eval script, so identical commands are
    deduped under a single config heading."""
    if not runs:
        return "(none)"
    art_by_claim = {c.id: _artifact(spec, c.artifact) for c in spec.claims}
    groups: dict = {}
    order: list = []
    for r in runs:
        a = art_by_claim.get(r.claim_id)
        model = a.base_model if a else r.claim_id
        key = (model, _config_label(a), _algorithm_label(a))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)

    blocks = []
    for (model, cfg, algo) in order:
        rs = groups[(model, cfg, algo)]
        algo_s = "" if algo in ("-", "?") else f" · {algo}"
        meta = "\n".join(_run_meta(r) for r in rs)
        # dedupe commands (a config's metrics usually share one script); fenced
        # block because eval commands are multi-line shell.
        cmds: list = []
        for r in rs:
            cmd = r.command.strip()
            if cmd not in cmds:
                cmds.append(cmd)
        cmd_s = "\n\n".join(f"```bash\n{cmd}\n```" for cmd in cmds)
        blocks.append(f"**{model} · {cfg}{algo_s}**\n{meta}\n\n{cmd_s}")
    return "\n\n".join(blocks)


def _drop_subgroup_rows(table: str) -> str:
    """Drop sub-group rows from an lm-eval results table — the per-subject MMLU breakdown
    (`| - humanities |…`, `|  - formal_logic |…`) — keeping the top-level `mmlu` row and
    the other tasks. A row is a sub-group when its first cell starts with `- ` (dash +
    space); the markdown separator (all dashes, no space) is preserved."""
    kept = []
    for ln in table.splitlines():
        cells = ln.split("|")
        first = cells[1].strip() if len(cells) > 1 else ""
        if first.startswith("- "):
            continue
        kept.append(ln)
    return "\n".join(kept)


def _raw_scores(runs: list[RunResult]) -> str:
    """The harness's raw per-task results table (per-task, minus the noisy MMLU subject
    breakdown) behind each claim's averaged metric, read from its stdout — so a
    suite-average claim (e.g. acc_norm_avg) isn't a black box."""
    out = []
    for r in runs:
        tbl = ""
        try:
            p = Path(r.stdout_path)
            if p.exists():
                tbl = _drop_subgroup_rows(extract_results_table(p.read_text(errors="replace")))
        except OSError:
            pass
        if tbl:
            out.append(f"**{r.claim_id}**\n\n{tbl}")
    return "\n\n".join(out) if out else "(none)"


def _patches_section(patches: list[str], heading: str) -> str:
    # Provenance of any edits made to get the repo running — valuable when present,
    # noise when empty, so the section is omitted entirely if there were no patches.
    if not patches:
        return ""
    body = "\n".join(f"- {p}" for p in patches)
    return f"\n## {heading}\n{body}\n"


def render_reports(spec: Spec, ingest: IngestInfo, grades: list[ClaimGrade],
                   runs: list[RunResult], env: dict, patches: list[str]) -> tuple[str, str]:
    summ = _summary(grades)
    repo_str = _repo_str(ingest, spec)
    env_str = _env_str(env)

    zh = f"""{_title_line("复现报告", ingest.title, ingest.arxiv_id)}

{_meta_block(repo_str, env_str, summ, ("仓库", "环境", "判定"))}

{_table(spec, grades, "| model | config | algorithm | metric | paper | 实测 | 判定 | 原因 |")}

## 各任务原始分数
{_raw_scores(runs)}

## 复算脚本(每个 config)
{_replay(spec, runs)}
{_patches_section(patches, "Setup 改动留痕")}"""

    en = f"""{_title_line("Reproduction Report", ingest.title, ingest.arxiv_id)}

{_meta_block(repo_str, env_str, summ, ("Repo", "Environment", "Verdict"))}

{_table(spec, grades, "| model | config | algorithm | metric | paper | measured | verdict | reason |")}

## Per-task raw scores
{_raw_scores(runs)}

## Replay script (per config)
{_replay(spec, runs)}
{_patches_section(patches, "Setup patches")}"""
    return zh, en
