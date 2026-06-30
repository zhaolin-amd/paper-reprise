"""Metric extraction from raw eval-script stdout.

Returns None when nothing reliable can be extracted — never guess.
Accuracy is always normalized to a percentage in [0, 100].
"""
from __future__ import annotations

import re
from typing import Optional

_PPL_PATTERNS = [
    r"perplexity[:\s=]+([0-9]+\.?[0-9]*)",
    r"\bppl\b[:\s=]+([0-9]+\.?[0-9]*)",
]
_ACC_PATTERNS = [
    r"acc[a-z,_ ]*[:\s=]+([0-9]+\.?[0-9]*)\s*%",   # "acc: 76.3%"
    # lm-eval markdown table row: "|acc     |↑  |0.500|±  |0.189|" — the `acc`
    # metric cell, then the direction cell, then the value cell. \bacc\b so the
    # `acc_norm` row isn't matched by the bare `acc` metric.
    r"\bacc\b[^|\n]*\|[^|\n]*\|\s*([0-9]*\.?[0-9]+)",
    r"acc[a-z,_ ]*[:\s=]+([0-9]*\.?[0-9]+)",        # "acc,none: 0.763"
]
_AVG_ACC_PATTERNS = [
    r"avg[_ ]?acc[a-z]*[:\s=]+([0-9]+\.?[0-9]*)\s*%",   # "avg_acc: 66.5%"
    r"avg[_ ]?acc[a-z]*[:\s=]+([0-9]*\.?[0-9]+)",        # "avg_acc: 0.665"
    r"average[a-z ]*[:\s=]+([0-9]+\.?[0-9]*)\s*%",
    r"average[a-z ]*[:\s=]+([0-9]*\.?[0-9]+)",
]
_SPEEDUP_PATTERNS = [
    r"speedup[:\s=]+([0-9]+\.?[0-9]*)\s*x?",
    r"([0-9]+\.?[0-9]*)\s*x\b",
]

# A signed/decimal/scientific number, e.g. 0.36, -1.5, 3.06e-5 — the value a generic
# scalar metric prints (distortion, recall, bpp, …).
_NUMBER = r"[-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?"


def _parse_generic_scalar(metric: str, text: str) -> Optional[float]:
    """Fallback for from-scratch papers whose metric isn't one of the known families
    (ppl/acc/avg/speedup): match a STANDALONE `<metric>: <number>` line and return the
    number VERBATIM (no %-normalization — a distortion of 0.36 must stay 0.36).

    Anchored to a whole line (like the from-scratch smoke gate) so prose like
    `exit code: 0` or a `foo.py:123` traceback can't be mistaken for the metric."""
    pat = rf"^[ \t]*{re.escape(metric)}[ \t]*:[ \t]*({_NUMBER})[ \t]*$"
    m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
    return float(m.group(1)) if m else None


def _first_match(patterns: list[str], text: str) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def _lm_eval_task_accs(text: str) -> list:
    """Top-level per-task `acc` values from an lm-eval markdown results table, e.g.
    `|arc_challenge|1|none|0|acc|↑|0.4872|±|0.0146|` → 0.4872. Used to average a
    multi-task accuracy metric when the harness prints no single average line.

    Counts only NAMED task rows with the bare `acc` metric — skips continuation rows
    (acc_norm, empty name), sub-group rows (`- subject`), and the header/separator."""
    accs = []
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 4:
            continue
        name = cells[1]
        if not name or name.startswith("-") or name.lower() in ("tasks", "groups"):
            continue
        if "acc" not in cells:                 # bare `acc` metric cell (not acc_norm)
            continue
        mi = cells.index("acc")
        val = None
        for c in cells[mi + 1:]:               # first numeric cell after the metric
            try:
                val = float(c)
                break
            except ValueError:
                continue
        if val is not None and 0.0 <= val <= 1.0:
            accs.append(val)
    return accs


def parse_metric(metric: str, text: str) -> Optional[float]:
    metric = metric.lower()
    if metric in ("perplexity", "ppl"):
        return _first_match(_PPL_PATTERNS, text)
    if metric in ("accuracy", "acc"):
        val = _first_match(_ACC_PATTERNS, text)
        if val is None:
            return None
        return val * 100 if val <= 1.0 else val
    if metric.startswith(("avg_", "average")) or metric.endswith(("_avg", "_average")):
        # benchmark-suite average accuracy under any label the spec uses
        # (avg_acc, acc_norm_avg, average_accuracy, …). Match a line named exactly
        # like the metric first, then the generic avg/average forms.
        pats = [rf"{re.escape(metric)}[:\s=]+([0-9]+\.?[0-9]*)\s*%",
                rf"{re.escape(metric)}[:\s=]+([0-9]*\.?[0-9]+)"] + _AVG_ACC_PATTERNS
        val = _first_match(pats, text)
        if val is not None:
            return val * 100 if val <= 1.0 else val
        # No explicit average line: many harnesses (lm-eval) only print a per-task table.
        # An "average accuracy over N tasks" metric is then the mean of the top-level task
        # `acc` cells — compute it. (e.g. AutoRound's `avg_accuracy_11tasks`.)
        accs = _lm_eval_task_accs(text)
        if accs:
            return sum(accs) / len(accs) * 100.0
        return None
    if metric == "speedup":
        return _first_match(_SPEEDUP_PATTERNS, text)
    # Unknown family: treat it as a generic scalar metric named exactly `metric`.
    return _parse_generic_scalar(metric, text)


# task -> primary score from an eval log's per-task summary lines, e.g.
# "arc_challenge acc_norm: 0.459", "arc_easy: 0.7285", "winogrande acc: 0.692".
_PER_TASK_RE = re.compile(
    r"^[ \t]*([A-Za-z][\w\-]+?)"
    r"(?:[ \t]+(?:acc_norm|acc|pass@1|exact_match|score|em))?"
    r"[ \t]*[:=][ \t]*([-+]?[0-9]*\.?[0-9]+)[ \t]*%?[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)
_PER_TASK_SKIP = {"acc", "acc_norm", "metric", "value", "tasks", "seed", "gpu", "minutes"}


def parse_per_task(text: str) -> dict:
    """Per-task scores from an eval log's summary lines (the per-task primary the
    harness prints, e.g. `arc_challenge: 0.459`). Skips the suite average and
    non-task labels. Values returned as-written (not %-normalized)."""
    out: dict = {}
    for m in re.finditer(_PER_TASK_RE, text or ""):
        name = m.group(1)
        low = name.lower()
        if low in _PER_TASK_SKIP or "avg" in low or "average" in low:
            continue
        out[name] = float(m.group(2))
    return out


_VRAM_RE = re.compile(r"peak_vram['\"]?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*GB", re.IGNORECASE)
_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_RUNTIME_RE = re.compile(r"running time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)


def parse_peak_vram_gb(text: str) -> Optional[float]:
    """Peak GPU memory in GB from an eval log (auto-round logs `peak_vram: 32.84GB`).
    Returns the maximum seen, or None if not reported."""
    vals = [float(m) for m in _VRAM_RE.findall(text or "")]
    return max(vals) if vals else None


def parse_runtime_minutes(text: str) -> Optional[float]:
    """Wall-clock minutes for a run: span between the first and last `YYYY-MM-DD HH:MM:SS`
    log timestamps; falls back to summing `running time=<n>s` lines. None if neither."""
    from datetime import datetime
    stamps = _TS_RE.findall(text or "")
    if len(stamps) >= 2:
        fmt = "%Y-%m-%d %H:%M:%S"
        try:
            ts = [datetime.strptime(s, fmt) for s in stamps]
            return (max(ts) - min(ts)).total_seconds() / 60.0
        except ValueError:
            pass
    secs = [float(s) for s in _RUNTIME_RE.findall(text or "")]
    return sum(secs) / 60.0 if secs else None


def extract_results_table(text: str) -> str:
    """Return the largest contiguous block of markdown table rows (`|...|` lines)
    from an eval log — the harness's raw results table — verbatim. "" if none."""
    best: list = []
    cur: list = []
    for ln in (text or "").splitlines():
        if ln.strip().startswith("|") and ln.count("|") >= 2:
            cur.append(ln.rstrip())
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    return "\n".join(best)
