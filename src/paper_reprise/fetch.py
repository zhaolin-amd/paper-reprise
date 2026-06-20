"""Ingest stage real fetch: latex download+unpack, git clone, arxiv title search.

Low-level I/O (HTTP, subprocess git) is isolated behind injectable functions so
the pure logic and orchestration are offline-testable with fakes.
"""
from __future__ import annotations

import io
import re
import tarfile
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional

import httpx


def latex_source_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/e-print/{arxiv_id}"


def _http_get_bytes(url: str) -> bytes:
    resp = httpx.get(url, follow_redirects=True, timeout=60.0)
    resp.raise_for_status()
    return resp.content


def fetch_latex(arxiv_id: str, dest: Path,
                *, http_get: Callable[[str], bytes] = _http_get_bytes) -> Path:
    """Download the arxiv e-print tarball and unpack it into dest. Returns dest."""
    data = http_get(latex_source_url(arxiv_id))
    unpack_targz(data, dest)
    return dest


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


def _http_get_text(url: str) -> str:
    resp = httpx.get(url, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    return resp.text


def arxiv_search_url(query: str) -> str:
    q = urllib.parse.quote(f"ti:{query}")
    return f"http://export.arxiv.org/api/query?search_query={q}&max_results=1"


def resolve_arxiv_id(query: str,
                     *, http_get: Callable[[str], str] = _http_get_text) -> Optional[str]:
    """Resolve a paper title to a bare arxiv id via the arxiv API. None if no match."""
    xml_text = http_get(arxiv_search_url(query))
    return parse_arxiv_search(xml_text)
