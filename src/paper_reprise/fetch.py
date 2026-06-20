"""Ingest stage real fetch: latex download+unpack, git clone, arxiv title search.

Low-level I/O (HTTP, subprocess git) is isolated behind injectable functions so
the pure logic and orchestration are offline-testable with fakes.
"""
from __future__ import annotations

import io
import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


def latex_source_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/e-print/{arxiv_id}"


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def unpack_targz(data: bytes, dest: Path) -> None:
    """Unpack a .tar.gz blob into dest, refusing any member that escapes dest."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            out = dest / member.name
            if not _is_within(dest, out):
                raise ValueError(f"unsafe path in archive: {member.name}")
        tar.extractall(dest)


_ARXIV_ABS_RE = re.compile(r"(\d{4}\.\d{4,5})")
_ATOM = "{http://www.w3.org/2005/Atom}"


def parse_arxiv_search(xml_text: str) -> Optional[str]:
    """Return the bare arxiv id of the first <entry> in an arxiv API Atom feed."""
    root = ET.fromstring(xml_text)
    entry = root.find(f"{_ATOM}entry")
    if entry is None:
        return None
    id_el = entry.find(f"{_ATOM}id")
    if id_el is None or not id_el.text:
        return None
    m = _ARXIV_ABS_RE.search(id_el.text)
    return m.group(1) if m else None
