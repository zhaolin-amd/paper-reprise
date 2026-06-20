"""Ingest stage real fetch: latex download+unpack, git clone, arxiv title search.

Low-level I/O (HTTP, subprocess git) is isolated behind injectable functions so
the pure logic and orchestration are offline-testable with fakes.
"""
from __future__ import annotations

import io
import re
import subprocess
import tarfile
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional

import httpx

from paper_reprise.ingest import find_repo_url


_MAX_UNPACK_BYTES = 500 * 1024 * 1024   # 500 MB decompressed cap
_MAX_MEMBERS = 10_000


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


def unpack_targz(data: bytes, dest: Path) -> None:
    """Unpack a .tar.gz blob into dest.

    Untrusted input (arxiv e-print tarballs). Uses tarfile's data filter to
    block path traversal, symlink/hardlink escapes, and special-device members,
    and bounds total decompressed size + member count against archive bombs.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = tar.getmembers()
        if len(members) > _MAX_MEMBERS:
            raise ValueError(f"archive has too many members: {len(members)}")
        total = sum(m.size for m in members)
        if total > _MAX_UNPACK_BYTES:
            raise ValueError(f"archive too large when unpacked: {total} bytes")
        try:
            tar.extractall(dest, filter="data")
        except tarfile.FilterError as e:
            raise ValueError(f"unsafe member in archive: {e}") from e


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


def _run_git_clone(url: str, dest: str) -> None:
    subprocess.run(["git", "clone", "--depth", "1", url, dest],
                   check=True, capture_output=True, text=True, timeout=300)


def clone_repo(repo_url: str, dest: Path,
               *, git_clone: Callable[[str, str], None] = _run_git_clone) -> Path:
    """Shallow-clone repo_url into dest. Returns dest."""
    git_clone(repo_url, str(dest))
    return dest


def _concat_latex_text(paper_dir: Path) -> str:
    """Concatenate all .tex files under paper_dir for repo-url scanning."""
    parts = []
    for p in sorted(paper_dir.rglob("*.tex")):
        try:
            parts.append(p.read_text(errors="ignore"))
        except OSError:
            continue
    return "\n".join(parts)


def make_fetch_sources(*, http_get: Callable[[str], bytes] = _http_get_bytes,
                       git_clone: Callable[[str, str], None] = _run_git_clone) -> Callable:
    """Build a fetch_sources(rd, arxiv_id, url) callback for the pipeline.

    Fetches latex into rd.paper_dir, scans it for a GitHub repo url, and clones
    that repo into rd.repo_dir if found. A missing repo link is not an error.
    """
    def fetch_sources(rd, arxiv_id: str, url: str) -> None:
        fetch_latex(arxiv_id, rd.paper_dir, http_get=http_get)
        repo_url = find_repo_url(_concat_latex_text(rd.paper_dir))
        if repo_url:
            clone_repo(repo_url, rd.repo_dir, git_clone=git_clone)

    return fetch_sources
