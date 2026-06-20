"""Ingest stage: normalize input to (arxiv_id, source_url), discover official repo.

Network fetches (latex tarball, git clone) are isolated behind functions that
callers can patch in tests. The parsing/normalization logic is pure.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")
_GITHUB_RE = re.compile(r"https?://github\.com/[\w.\-]+/[\w.\-]+")


def arxiv_id_from_url(url: str) -> Optional[str]:
    m = _ARXIV_RE.search(url)
    return m.group(1) if m else None


def parse_org(text: str) -> dict:
    meta: dict = {"authors": []}
    for line in text.splitlines():
        m = re.match(r"#\+(\w+):\s*(.*)", line)
        if not m:
            continue
        key, val = m.group(1).lower(), m.group(2).strip()
        if key == "authors":
            meta["authors"] = [a.strip() for a in val.split(",") if a.strip()]
        else:
            meta[key] = val
    return meta


def find_repo_url(text: str) -> Optional[str]:
    m = _GITHUB_RE.search(text)
    if not m:
        return None
    return m.group(0).rstrip("/").removesuffix(".git")


def normalize_input(arg: str) -> tuple[str, str]:
    """Return (arxiv_id, source_url) from a .org path, arxiv url, or bare id."""
    p = Path(arg)
    if p.exists() and p.suffix == ".org":
        meta = parse_org(p.read_text())
        url = meta.get("source", "")
        arxiv_id = arxiv_id_from_url(url)
        if not arxiv_id:
            raise ValueError(f"no arxiv source in {arg}")
        return arxiv_id, url
    if arg.startswith("http"):
        arxiv_id = arxiv_id_from_url(arg)
        if not arxiv_id:
            raise ValueError(f"cannot extract arxiv id from {arg}")
        return arxiv_id, f"https://arxiv.org/abs/{arxiv_id}"
    if _ARXIV_RE.fullmatch(arg):
        return arg, f"https://arxiv.org/abs/{arg}"
    raise ValueError(f"unrecognized input: {arg}")
