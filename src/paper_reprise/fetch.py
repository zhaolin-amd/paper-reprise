"""Ingest stage real fetch: latex download+unpack, git clone, arxiv title search.

Low-level I/O (HTTP, subprocess git) is isolated behind injectable functions so
the pure logic and orchestration are offline-testable with fakes.
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path


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
