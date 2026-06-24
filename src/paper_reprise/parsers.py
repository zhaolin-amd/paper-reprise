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


def _first_match(patterns: list[str], text: str) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


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
        if val is None:
            return None
        return val * 100 if val <= 1.0 else val
    if metric == "speedup":
        return _first_match(_SPEEDUP_PATTERNS, text)
    return None


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
