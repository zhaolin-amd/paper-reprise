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
    r"acc[a-z,_ ]*[:\s=]+([0-9]*\.?[0-9]+)",        # "acc,none: 0.763"
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
    if metric == "speedup":
        return _first_match(_SPEEDUP_PATTERNS, text)
    return None
