"""Render bilingual reproduction reports (zh + en) as two markdown strings.

Iron rules (design §5.2):
  - always show measured (raw) numbers, never paper numbers as substitute
  - every claim carries replay info (command/seed/gpu/commit/env)
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from paper_reprise.models import ClaimGrade, IngestInfo, RunResult, Spec
from paper_reprise.modelpaths import model_base
from paper_reprise.parsers import (
    extract_results_table,
    parse_peak_vram_gb,
    parse_runtime_minutes,
)


def _display_model(base_model: str) -> str:
    """Convert a local snapshot path back to HF-style `org/name` for display.
    `/group/.../Qwen/Qwen3-8B` → `Qwen/Qwen3-8B`. A genuine HF id is returned as-is.
    Strips the shared model cache prefix (model_base()) when it matches; also strips
    /scratch paths from resolve_model's scratch fallback."""
    p = Path(base_model)
    if not p.is_absolute():
        return base_model          # already an HF id like "Qwen/Qwen3-8B"
    for base in (model_base(),):
        try:
            rel = p.relative_to(base)
            parts = rel.parts
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
        except ValueError:
            pass
    # scratch fallback: /scratch/<user>/pretrained_models/<org>/<name>
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "pretrained_models" and i + 2 < len(parts):
            return f"{parts[i+1]}/{parts[i+2]}"
    return base_model              # unknown layout — keep as-is


def _repo_str(ingest: IngestInfo, spec: Spec | None = None) -> str:
    # ingest often doesn't carry the repo even when specextract found one — fall
    # back to spec.repo so a run against an official repo isn't mislabelled.
    repo = ingest.repo or (spec.repo if spec else None)
    return f"{repo.url}@{repo.commit}" if repo else "(no official repo)"


_UNKNOWN = {"", "?", "unknown", "none"}

# Version banners the eval itself prints, used to recover versions the venv
# snapshot missed (e.g. when the eval ran in a shared conda env, or pip freeze
# came back empty). auto-round / GPTQModel print "Torch : X" / "Transformers : X";
# lm-eval prints "Using lm-eval version X". Logs are committed (see .gitignore
# `!runs/**/*.log`), so this stays reproducible.
_VER_PATTERNS = {
    "torch": [r"Torch\s*:\s*([0-9][^\s]+)", r"\btorch==([0-9][^\s,]+)"],
    "transformers": [r"Transformers\s*:\s*([0-9][^\s]+)", r"\btransformers==([0-9][^\s,]+)"],
    "lm_eval": [r"lm[- ]eval(?:uation)?(?:[- ]harness)? version\s+([0-9][^\s]+)",
                r"\blm[_-]eval==([0-9][^\s,]+)"],
}


def _versions_from_logs(runs: list[RunResult]) -> dict:
    """Recover torch/transformers/lm_eval versions from each claim's stdout.log."""
    found: dict = {}
    for r in runs:
        path = getattr(r, "stdout_path", None)
        if not path:
            continue
        try:
            text = Path(path).read_text(errors="ignore")
        except OSError:
            continue
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)  # strip ANSI color codes
        for key, pats in _VER_PATTERNS.items():
            if key in found:
                continue
            for pat in pats:
                m = re.search(pat, text)
                if m:
                    found[key] = m.group(1).rstrip(".,;")
                    break
    return found


def _effective_env(env: dict, runs: list[RunResult]) -> dict:
    """Snapshot env, with any empty/unknown version field filled from the logs."""
    eff = dict(env or {})
    for k, v in _versions_from_logs(runs).items():
        if str(eff.get(k) or "").strip().lower() in _UNKNOWN:
            eff[k] = v
    return eff


def _env_str(env: dict) -> str:
    # Only show env components we actually know — an unknown one is dropped rather than
    # rendered as "?" (e.g. a pure-numpy from-scratch run has no torch/transformers/CUDA).
    # CUDA (NVIDIA) and ROCm (AMD) are mutually exclusive per torch build; whichever the
    # snapshot captured is shown, the absent one is simply dropped.
    # Order follows the stack top-down: hardware runtime -> framework -> library -> eval
    # harness (CUDA/ROCm -> torch -> transformers -> lm_eval), i.e. quantization to eval.
    parts = []
    for label, key in (("CUDA", "cuda"), ("ROCm", "rocm"), ("torch", "torch"),
                       ("transformers", "transformers"), ("lm_eval", "lm_eval")):
        val = str(env.get(key) or "").strip()
        if val and val.lower() not in _UNKNOWN:
            parts.append(f"{label} {val}")
    return " / ".join(parts)


def _title_line(prefix: str, title: str | None, arxiv_id: str) -> str:
    """`# <prefix>: <title> (<arxiv_id>)`, collapsed to just the id when there is no
    distinct title (avoids the ugly `2504.19874 (2504.19874)`)."""
    if title and title != arxiv_id:
        return f"# {prefix}: {title} ({arxiv_id})"
    return f"# {prefix}: {arxiv_id}"


def _meta_block(repo_str: str, env_str: str, labels: tuple[str, str]) -> str:
    """The header metadata as a Markdown bullet list (repo / env), each on its own line.
    The env bullet is omitted entirely when nothing is known. The verdict counts are not
    repeated here — they live in the Conclusion section below the table."""
    repo_l, env_l = labels
    lines = [f"- **{repo_l}:** {repo_str}"]
    if env_str:
        lines.append(f"- **{env_l}:** {env_str}")
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


def _table(spec, grades, header, lang="en"):
    """One row per claim (model × config × metric), with paper, measured(diff),
    verdict and reason all in a single table. `lang` picks the reason language."""
    runs_by = {g.claim_id: g for g in grades}
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for c in spec.claims:
        g = runs_by.get(c.id)
        measured = _measured_cell(g, c.expected)
        verdict = g.verdict if g else "BLOCKED"
        if not g:
            reason = "无评分" if lang == "zh" else "no grade"
        elif lang == "zh":
            reason = g.reason_zh or g.reason
        else:
            reason = g.reason
        a = _artifact(spec, c.artifact)
        model = _display_model(a.base_model) if a else c.artifact
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
        model = _display_model(a.base_model) if a else r.claim_id
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


def _fmt_minutes(m: float | None) -> str:
    if m is None:
        return "—"
    if m < 60:
        return f"{m:.1f} min"
    return f"{m / 60:.1f} h"


def _resources(spec: Spec, runs: list[RunResult], header: str) -> str:
    """Per-claim cost: wall-clock time and peak GPU memory, parsed from each run's log.
    Omitted rows when a claim produced no log (e.g. BLOCKED before running)."""
    runs_by = {r.claim_id: r for r in runs}
    rows = [header, "|---|---|---|---|"]
    any_row = False
    for c in spec.claims:
        r = runs_by.get(c.id)
        text = ""
        if r and r.stdout_path:
            try:
                p = Path(r.stdout_path)
                if p.exists():
                    text = p.read_text(errors="replace")
            except OSError:
                text = ""
        mins = parse_runtime_minutes(text) if text else None
        vram = parse_peak_vram_gb(text) if text else None
        if mins is None and vram is None:
            continue
        any_row = True
        a = _artifact(spec, c.artifact)
        model = _display_model(a.base_model) if a else c.artifact
        vram_s = f"{vram:.1f} GB" if vram is not None else "—"
        rows.append(f"| {model} | {_config_label(a)} | {_fmt_minutes(mins)} | {vram_s} |")
    return "\n".join(rows) if any_row else "(none)"


def _conclusion(spec: Spec, grades: list[ClaimGrade], lang: str) -> str:
    """A short, FACTUAL digest below the verdict table. Computed, not editorialized:
    - verdict counts;
    - if the FP baseline matched but quantized configs are off → the eval protocol is
      validated, so those gaps are a real reproduction gap (the baseline-as-claim logic);
    - else, if every graded value sits on one side of the paper → a systematic offset;
    - a note on any BLOCKED claims."""
    c = Counter(g.verdict for g in grades)
    graded = [g for g in grades if g.measured is not None and g.expected is not None]
    diffs = [g.measured - g.expected for g in graded]
    art = {a.id: a for a in spec.artifacts}
    clm = {cl.id: cl.artifact for cl in spec.claims}

    def _is_baseline(g: ClaimGrade) -> bool:
        a = art.get(clm.get(g.claim_id))
        return a is not None and _algorithm_label(a) == "-"

    base_match = [g for g in graded if _is_baseline(g) and g.checks.get("value")]
    quant_off = [g for g in graded if not _is_baseline(g) and not g.checks.get("value")]
    zh = lang == "zh"
    lines = []
    if zh:
        lines.append(f"- 共 {len(grades)} 个 claim:MATCH {c['MATCH']} · PARTIAL "
                     f"{c['PARTIAL']} · FAIL {c['FAIL']} · BLOCKED {c['BLOCKED']}。")
    else:
        lines.append(f"- {len(grades)} claims: MATCH {c['MATCH']} · PARTIAL {c['PARTIAL']} "
                     f"· FAIL {c['FAIL']} · BLOCKED {c['BLOCKED']}.")

    if base_match and quant_off:
        n = len(quant_off)
        worst = min(quant_off, key=lambda g: g.measured - g.expected)
        d = worst.measured - worst.expected
        if zh:
            lines.append(f"- FP 基线与论文吻合,说明**评测协议可信**;因此 {n} 个超容差的"
                         f"量化配置(最大偏差 {d:+.2f})是**真实的复现差距**(算法/校准/版本所致),"
                         "而非评测口径问题。")
        else:
            lines.append(f"- The FP baseline matches the paper, so the **eval protocol is "
                         f"validated**; the {n} quantized config(s) outside tolerance "
                         f"(worst {d:+.2f}) are therefore a **genuine reproduction gap** "
                         "(algorithm/calibration/version), not an eval-protocol artifact.")
    elif diffs and all(d > 0 for d in diffs):
        if zh:
            lines.append(f"- 实测相对论文**一致偏高**(Δ +{min(diffs):.2f}~+{max(diffs):.2f}),"
                         "更像系统性的评测/环境偏移(如 lm-eval/算法库版本差异),而非逐配置噪声。")
        else:
            lines.append(f"- Measured is **consistently above** the paper "
                         f"(Δ +{min(diffs):.2f}…+{max(diffs):.2f}) — a systematic eval/setup "
                         "offset (e.g. lm-eval/library version drift), not per-config noise.")
    elif diffs and all(d < 0 for d in diffs):
        if zh:
            lines.append(f"- 实测相对论文**一致偏低**(Δ {max(diffs):.2f}~{min(diffs):.2f}),"
                         "更像系统性的评测/环境偏移,而非逐配置噪声。")
        else:
            lines.append(f"- Measured is **consistently below** the paper "
                         f"(Δ {max(diffs):.2f}…{min(diffs):.2f}) — a systematic eval/setup "
                         "offset, not per-config noise.")

    if c["BLOCKED"]:
        if zh:
            lines.append(f"- {c['BLOCKED']} 个 BLOCKED 未产出可比数值(见各自 reason),非「未复现」。")
        else:
            lines.append(f"- {c['BLOCKED']} BLOCKED produced no comparable value "
                         "(see each reason) — not 'failed to reproduce'.")
    return "\n".join(lines)


def _analysis_section(analysis: str, heading: str) -> str:
    """Optional human-written gap analysis appended after the auto-generated Conclusion.
    Written to `analysis.md` in the run dir; never overwritten by re-renders. Empty
    string → section omitted. The heading is bilingual: pass "## Analysis" or "## 分析"."""
    body = (analysis or "").strip()
    return f"\n{heading}\n{body}\n" if body else ""


def render_reports(spec: Spec, ingest: IngestInfo, grades: list[ClaimGrade],
                   runs: list[RunResult], env: dict, patches: list[str],
                   analysis: str = "") -> tuple[str, str]:
    """Render bilingual reports. `analysis` (from analysis.md) is appended verbatim
    to both reports — write it in both languages or whichever you prefer."""
    repo_str = _repo_str(ingest, spec)
    env_str = _env_str(_effective_env(env, runs))

    zh = f"""{_title_line("复现报告", ingest.title, ingest.arxiv_id)}

{_meta_block(repo_str, env_str, ("仓库", "环境"))}

{_table(spec, grades, "| model | config | algorithm | metric | paper | 实测 | 判定 | 原因 |", "zh")}

## 结论
{_conclusion(spec, grades, "zh")}
{_analysis_section(analysis, "## 差距分析")}
## 资源占用(每个 config)
{_resources(spec, runs, "| model | config | 时长 | 峰值显存 |")}

## 各任务原始分数
{_raw_scores(runs)}

## 复算脚本(每个 config)
{_replay(spec, runs)}
{_patches_section(patches, "Setup 改动留痕")}"""

    en = f"""{_title_line("Reproduction Report", ingest.title, ingest.arxiv_id)}

{_meta_block(repo_str, env_str, ("Repo", "Environment"))}

{_table(spec, grades, "| model | config | algorithm | metric | paper | measured | verdict | reason |", "en")}

## Conclusion
{_conclusion(spec, grades, "en")}
{_analysis_section(analysis, "## Analysis")}
## Resources (per config)
{_resources(spec, runs, "| model | config | time | peak VRAM |")}

## Per-task raw scores
{_raw_scores(runs)}

## Replay script (per config)
{_replay(spec, runs)}
{_patches_section(patches, "Setup patches")}"""
    return zh, en
