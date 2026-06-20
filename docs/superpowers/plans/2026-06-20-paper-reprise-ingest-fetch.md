# Plan 2a: Ingest Real Fetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ingest stage's placeholder `fetch_sources` with a real implementation — download + unpack the arxiv LaTeX source into `paper/`, `git clone` the official repo into `repo/`, and resolve a paper title to an arxiv id via the arxiv API — while keeping the whole thing offline-testable through injectable I/O seams.

**Architecture:** A new `fetch.py` module holds the real network/subprocess work. The lowest-level I/O (HTTP GET, subprocess git, arxiv API query) is isolated behind three thin injectable functions so the pure logic (URL building, tar extraction, repo-url discovery, arxiv XML parsing) is tested offline with fakes. A `make_fetch_sources()` factory returns a callable matching Plan 1's `fetch_sources(rd, arxiv_id, url)` seam; the CLI wires it in and adds a title→id resolver step before `run_pipeline`.

**Tech Stack:** Python 3.12, httpx (HTTP, already a dependency), stdlib `tarfile`/`gzip`/`subprocess`/`xml.etree.ElementTree`, pytest (offline via monkeypatch/fakes), uv, ruff.

Design doc: `docs/superpowers/specs/2026-06-19-paper-reprise-agent-design.md` (§3.1 Ingest)

---

## Context for the implementer

Plan 1 shipped a deterministic skeleton. The ingest stage currently has two halves:

- `src/paper_reprise/ingest.py` — pure logic only: `normalize_input(arg)` (arxiv id/url → `(arxiv_id, source_url)`), `arxiv_id_from_url`, `find_repo_url(text)` (scrapes the first GitHub URL out of text). No network.
- The pipeline calls an injected `fetch_sources(rd, arxiv_id, url)` callback that is supposed to fill `rd.paper_dir` and `rd.repo_dir`. In `src/paper_reprise/cli.py` this callback is currently a stub that just prints `"[ingest] ... (source fetch deferred to Plan 2)"`.

Relevant existing API (do not change signatures):
- `RunDir` has `.paper_dir` and `.repo_dir` properties (both already `mkdir`-ed by `RunDir.create`), and `.root`.
- `IngestInfo` (pydantic) has fields: `arxiv_id, title, authors, source_url, repo, latex_path, repo_path`, plus `repo: Optional[RepoInfo]` where `RepoInfo(url, commit)`.
- The pipeline builds `IngestInfo(arxiv_id=..., source_url=...)` itself; Plan 2a does NOT need to change the pipeline. It only needs to (a) provide a real `fetch_sources` in the CLI, and (b) add a title resolver in the CLI before `run_pipeline`.

This plan does NOT touch the setup loop or the GPU executor (those are Plan 2b / 2c). It only makes ingest really fetch.

---

## File Structure

```
src/paper_reprise/
  fetch.py        # NEW — real fetch: latex download+unpack, git clone, arxiv title search.
                  #       low-level I/O isolated behind injectable functions.
  ingest.py       # unchanged (pure normalization/discovery logic stays here)
  cli.py          # MODIFY — wire make_fetch_sources() into run; add title resolver step
tests/
  test_fetch.py   # NEW — offline tests via injected fakes
  fixtures/
    arxiv_search_response.xml   # NEW — sample arxiv API Atom response for the resolver
```

**Responsibility split inside `fetch.py`:**
- Low-level injectable I/O (the only functions that touch the outside world):
  - `_http_get_bytes(url) -> bytes`
  - `_http_get_text(url) -> str`
  - `_run_git_clone(url, dest) -> None`
- Pure logic (offline-testable directly):
  - `latex_source_url(arxiv_id) -> str`
  - `unpack_targz(data: bytes, dest: Path) -> None`
  - `parse_arxiv_search(xml_text) -> Optional[str]` (Atom XML → first arxiv id)
- Orchestration (testable by injecting fakes for the three I/O functions):
  - `fetch_latex(arxiv_id, dest, *, http_get=...) -> Path`
  - `clone_repo(repo_url, dest, *, git_clone=...) -> Path`
  - `resolve_arxiv_id(query, *, http_get=...) -> Optional[str]`
  - `make_fetch_sources(*, http_get=..., git_clone=...) -> Callable` returns `fetch_sources(rd, arxiv_id, url)`

---

## Task 1: latex source URL + tar.gz unpack (pure logic)

**Files:**
- Create: `src/paper_reprise/fetch.py`
- Test: `tests/test_fetch.py`

- [ ] **Step 1: Write the failing test**

`tests/test_fetch.py`:
```python
import io
import tarfile

from paper_reprise.fetch import latex_source_url, unpack_targz


def test_latex_source_url():
    assert latex_source_url("2401.00001") == "https://arxiv.org/e-print/2401.00001"


def _make_targz(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_unpack_targz_writes_files(tmp_path):
    blob = _make_targz({"main.tex": "\\documentclass{article}", "sec/intro.tex": "hi"})
    unpack_targz(blob, tmp_path)
    assert (tmp_path / "main.tex").read_text() == "\\documentclass{article}"
    assert (tmp_path / "sec" / "intro.tex").read_text() == "hi"


def test_unpack_targz_rejects_path_traversal(tmp_path):
    import pytest
    # a malicious member escaping the dest must be refused
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"x"
        info = tarfile.TarInfo(name="../escape.tex")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="unsafe path"):
        unpack_targz(buf.getvalue(), tmp_path)
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_fetch.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.fetch'`

- [ ] **Step 3: Write the implementation**

`src/paper_reprise/fetch.py`:
```python
"""Ingest stage real fetch: latex download+unpack, git clone, arxiv title search.

Low-level I/O (HTTP, subprocess git) is isolated behind injectable functions so
the pure logic and orchestration are offline-testable with fakes.
"""
from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path
from typing import Callable, Optional


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
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_fetch.py -v`
Expected: PASS, 3 tests green

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/fetch.py tests/test_fetch.py
git commit -m "feat: latex source url + safe tar.gz unpack (path-traversal guarded)"
```

---

## Task 2: arxiv API search response parsing (pure logic)

**Files:**
- Modify: `src/paper_reprise/fetch.py`
- Create: `tests/fixtures/arxiv_search_response.xml`
- Test: `tests/test_fetch.py`

The arxiv API (`http://export.arxiv.org/api/query?search_query=ti:...`) returns an Atom feed. We parse the first `<entry>`'s `<id>` (e.g. `http://arxiv.org/abs/2401.00001v1`) into a bare arxiv id.

- [ ] **Step 1: Write the fixture**

`tests/fixtures/arxiv_search_response.xml`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v2</id>
    <title>AWQ: Activation-aware Weight Quantization for LLMs</title>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2306.99999v1</id>
    <title>Some other paper</title>
  </entry>
</feed>
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_fetch.py`:
```python
from pathlib import Path

from paper_reprise.fetch import parse_arxiv_search

FIX = Path(__file__).parent / "fixtures"


def test_parse_arxiv_search_returns_first_id():
    xml = (FIX / "arxiv_search_response.xml").read_text()
    assert parse_arxiv_search(xml) == "2401.00001"


def test_parse_arxiv_search_empty_feed_returns_none():
    xml = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    assert parse_arxiv_search(xml) is None
```

- [ ] **Step 3: Run the test, confirm it fails**

Run: `uv run pytest tests/test_fetch.py::test_parse_arxiv_search_returns_first_id -v`
Expected: FAIL, `ImportError: cannot import name 'parse_arxiv_search'`

- [ ] **Step 4: Write the implementation**

Add to `src/paper_reprise/fetch.py` (imports at top: add `import re` and `import xml.etree.ElementTree as ET`):
```python
import re
import xml.etree.ElementTree as ET

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
```

- [ ] **Step 5: Run the tests, confirm they pass**

Run: `uv run pytest tests/test_fetch.py -v`
Expected: PASS, all fetch tests green (5 total now)

- [ ] **Step 6: Commit**

```bash
git add src/paper_reprise/fetch.py tests/test_fetch.py tests/fixtures/arxiv_search_response.xml
git commit -m "feat: parse arxiv API Atom feed → first arxiv id"
```

---

## Task 3: fetch_latex orchestration (inject fake HTTP)

**Files:**
- Modify: `src/paper_reprise/fetch.py`
- Test: `tests/test_fetch.py`

`fetch_latex` ties together `latex_source_url` + an injected byte-fetcher + `unpack_targz`, returning the destination path. The real HTTP getter defaults to httpx but is injectable.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fetch.py`:
```python
from paper_reprise.fetch import fetch_latex


def test_fetch_latex_downloads_and_unpacks(tmp_path):
    blob = _make_targz({"main.tex": "\\section{Intro}"})
    calls = {}

    def fake_get(url):
        calls["url"] = url
        return blob

    dest = fetch_latex("2401.00001", tmp_path / "paper", http_get=fake_get)
    assert calls["url"] == "https://arxiv.org/e-print/2401.00001"
    assert (dest / "main.tex").read_text() == "\\section{Intro}"
    assert dest == tmp_path / "paper"
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_fetch.py::test_fetch_latex_downloads_and_unpacks -v`
Expected: FAIL, `ImportError: cannot import name 'fetch_latex'`

- [ ] **Step 3: Write the implementation**

Add to `src/paper_reprise/fetch.py`:
```python
import httpx


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
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_fetch.py::test_fetch_latex_downloads_and_unpacks -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/fetch.py tests/test_fetch.py
git commit -m "feat: fetch_latex (download e-print + unpack), httpx getter injectable"
```

---

## Task 4: resolve_arxiv_id (title → id, inject fake HTTP)

**Files:**
- Modify: `src/paper_reprise/fetch.py`
- Test: `tests/test_fetch.py`

`resolve_arxiv_id` builds the arxiv API query URL, fetches text via an injected getter, and parses the first id. Returns None if nothing found.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fetch.py`:
```python
import urllib.parse

from paper_reprise.fetch import resolve_arxiv_id


def test_resolve_arxiv_id_from_title(tmp_path):
    xml = (FIX / "arxiv_search_response.xml").read_text()
    captured = {}

    def fake_get(url):
        captured["url"] = url
        return xml

    got = resolve_arxiv_id("AWQ Activation-aware Weight Quantization", http_get=fake_get)
    assert got == "2401.00001"
    # the title must be url-encoded into a ti: query against the arxiv API
    assert "export.arxiv.org/api/query" in captured["url"]
    assert "ti:" in urllib.parse.unquote(captured["url"])


def test_resolve_arxiv_id_no_match_returns_none():
    empty = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    assert resolve_arxiv_id("nonexistent paper xyz", http_get=lambda u: empty) is None
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_fetch.py::test_resolve_arxiv_id_from_title -v`
Expected: FAIL, `ImportError: cannot import name 'resolve_arxiv_id'`

- [ ] **Step 3: Write the implementation**

Add to `src/paper_reprise/fetch.py`:
```python
import urllib.parse


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
```

- [ ] **Step 4: Run the tests, confirm they pass**

Run: `uv run pytest tests/test_fetch.py -v`
Expected: PASS, all fetch tests green

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/fetch.py tests/test_fetch.py
git commit -m "feat: resolve_arxiv_id (title → arxiv id via arxiv API)"
```

---

## Task 5: clone_repo (inject fake git)

**Files:**
- Modify: `src/paper_reprise/fetch.py`
- Test: `tests/test_fetch.py`

`clone_repo` runs `git clone` into dest via an injected runner, returning dest. The real runner uses subprocess; tests inject a fake that records the call and creates the dir.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fetch.py`:
```python
from paper_reprise.fetch import clone_repo


def test_clone_repo_invokes_git_and_returns_dest(tmp_path):
    calls = {}

    def fake_clone(url, dest):
        calls["url"] = url
        calls["dest"] = dest
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "README.md").write_text("cloned")

    dest = clone_repo("https://github.com/foo/bar", tmp_path / "repo",
                      git_clone=fake_clone)
    assert calls["url"] == "https://github.com/foo/bar"
    assert dest == tmp_path / "repo"
    assert (dest / "README.md").read_text() == "cloned"
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_fetch.py::test_clone_repo_invokes_git_and_returns_dest -v`
Expected: FAIL, `ImportError: cannot import name 'clone_repo'`

- [ ] **Step 3: Write the implementation**

Add to `src/paper_reprise/fetch.py` (add `import subprocess` at top):
```python
import subprocess


def _run_git_clone(url: str, dest: str) -> None:
    subprocess.run(["git", "clone", "--depth", "1", url, dest],
                   check=True, capture_output=True, text=True, timeout=300)


def clone_repo(repo_url: str, dest: Path,
               *, git_clone: Callable[[str, str], None] = _run_git_clone) -> Path:
    """Shallow-clone repo_url into dest. Returns dest."""
    git_clone(repo_url, str(dest))
    return dest
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_fetch.py::test_clone_repo_invokes_git_and_returns_dest -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/fetch.py tests/test_fetch.py
git commit -m "feat: clone_repo (shallow git clone), runner injectable"
```

---

## Task 6: make_fetch_sources factory (the fetch_sources callback)

**Files:**
- Modify: `src/paper_reprise/fetch.py`
- Test: `tests/test_fetch.py`

This assembles the pieces into a callable matching Plan 1's `fetch_sources(rd, arxiv_id, url)` seam: fetch latex into `rd.paper_dir`, then scan the fetched latex for a GitHub repo url (`find_repo_url`), and if found clone it into `rd.repo_dir`. Missing repo is fine (leave `repo/` empty — the pipeline already handles `repo: null`). The factory takes injectable `http_get` / `git_clone` so the whole callback is offline-testable.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fetch.py`:
```python
from paper_reprise.fetch import make_fetch_sources
from paper_reprise.rundir import RunDir


def test_make_fetch_sources_fetches_latex_and_clones_repo(tmp_path):
    latex = _make_targz({"main.tex": "code at https://github.com/foo/bar yay"})
    cloned = {}

    def fake_get(url):
        return latex

    def fake_clone(url, dest):
        cloned["url"] = url
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "x").write_text("ok")

    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    fetch_sources = make_fetch_sources(http_get=fake_get, git_clone=fake_clone)
    fetch_sources(rd, "2401.00001", "https://arxiv.org/abs/2401.00001")

    assert (rd.paper_dir / "main.tex").exists()
    assert cloned["url"] == "https://github.com/foo/bar"
    assert (rd.repo_dir / "x").read_text() == "ok"


def test_make_fetch_sources_no_repo_link_skips_clone(tmp_path):
    latex = _make_targz({"main.tex": "no links in this paper"})
    clone_called = {"n": 0}

    def fake_clone(url, dest):
        clone_called["n"] += 1

    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    fetch_sources = make_fetch_sources(http_get=lambda u: latex, git_clone=fake_clone)
    fetch_sources(rd, "2401.00001", "https://arxiv.org/abs/2401.00001")

    assert (rd.paper_dir / "main.tex").exists()
    assert clone_called["n"] == 0
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_fetch.py::test_make_fetch_sources_fetches_latex_and_clones_repo -v`
Expected: FAIL, `ImportError: cannot import name 'make_fetch_sources'`

- [ ] **Step 3: Write the implementation**

Add to `src/paper_reprise/fetch.py` (add `from paper_reprise.ingest import find_repo_url` at top):
```python
from paper_reprise.ingest import find_repo_url


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
```

- [ ] **Step 4: Run the tests, confirm they pass**

Run: `uv run pytest tests/test_fetch.py -v`
Expected: PASS, all fetch tests green

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/fetch.py tests/test_fetch.py
git commit -m "feat: make_fetch_sources factory (latex fetch + conditional repo clone)"
```

---

## Task 7: wire real fetch + title resolver into the CLI

**Files:**
- Modify: `src/paper_reprise/cli.py`
- Test: `tests/test_cli.py`

Replace the CLI's placeholder `fetch_sources` with `make_fetch_sources()`, and before calling `run_pipeline`, resolve a title-looking input to an arxiv id. "Title-looking" = not an arxiv id and not a URL (so `normalize_input` would otherwise reject it). Add a `--offline` escape hatch is NOT needed; instead the network getters default to the real httpx/git ones. Tests inject fakes via monkeypatch of the `fetch` module functions.

The resolver logic: if `input_arg` is neither a bare arxiv id (`\d{4}\.\d{4,5}`) nor starts with `http`, treat it as a title, call `resolve_arxiv_id`, and replace `input_arg` with the resolved id (or raise a clear error if unresolved).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:
```python
import paper_reprise.cli as cli_mod


def test_cli_run_resolves_title_then_aborts_on_specextract(tmp_path, monkeypatch):
    # title → arxiv id resolution happens; then specextract has no real spec so
    # the pipeline aborts at specextract (no GPU work needed). We assert the
    # resolver was consulted and the run dir is created under the resolved id.
    seen = {}

    def fake_resolve(query, **kwargs):
        seen["query"] = query
        return "2401.00001"

    def fake_fetch_sources_factory(**kwargs):
        def _fs(rd, arxiv_id, url):
            seen["fetched"] = arxiv_id
        return _fs

    monkeypatch.setattr(cli_mod, "resolve_arxiv_id", fake_resolve)
    monkeypatch.setattr(cli_mod, "make_fetch_sources", fake_fetch_sources_factory)

    from click.testing import CliRunner
    res = CliRunner().invoke(
        cli_mod.cli,
        ["run", "AWQ Activation-aware Weight Quantization",
         "--base-dir", str(tmp_path), "--yes"],
    )
    assert res.exit_code == 0
    assert seen["query"] == "AWQ Activation-aware Weight Quantization"
    assert seen["fetched"] == "2401.00001"
    assert "Aborted at: specextract" in res.output


def test_cli_run_unresolvable_title_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "resolve_arxiv_id", lambda q, **k: None)
    from click.testing import CliRunner
    res = CliRunner().invoke(
        cli_mod.cli, ["run", "no such paper", "--base-dir", str(tmp_path), "--yes"]
    )
    assert res.exit_code != 0
    assert "could not resolve" in res.output.lower()
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_cli.py::test_cli_run_resolves_title_then_aborts_on_specextract -v`
Expected: FAIL (the CLI does not import/use `resolve_arxiv_id` or `make_fetch_sources` yet, and the title input currently makes `normalize_input` raise `unrecognized input`)

- [ ] **Step 3: Write the implementation**

Replace the `run` command in `src/paper_reprise/cli.py`. Add these imports near the top (after the existing imports):
```python
import re as _re

from paper_reprise.fetch import make_fetch_sources, resolve_arxiv_id
```

Then replace the body of the `run` function's `fetch_sources` stub and add resolver logic. The full updated `run` command:
```python
@cli.command()
@click.argument("input_arg")
@click.option("--base-dir", default="runs", help="where run dirs are created")
@click.option("--yes", is_flag=True, help="auto-approve all gates (non-interactive)")
def run(input_arg: str, base_dir: str, yes: bool) -> None:
    """Run the reproduction pipeline for a paper (arxiv id, url, or title)."""
    from paper_reprise.pipeline import run_pipeline

    # Title input: not a bare arxiv id and not a URL → resolve via arxiv API.
    if not _re.fullmatch(r"\d{4}\.\d{4,5}", input_arg) and not input_arg.startswith("http"):
        resolved = resolve_arxiv_id(input_arg)
        if resolved is None:
            raise click.ClickException(f"could not resolve title to an arxiv id: {input_arg}")
        click.echo(f"[resolve] '{input_arg}' → {resolved}")
        input_arg = resolved

    def approve_spec(spec):
        if yes:
            return True
        click.echo(f"\nExtracted {len(spec.claims)} claims. Review spec.yaml.")
        return click.confirm("Approve spec and continue?", default=True)

    def approve_plan(plan):
        if yes:
            return True
        click.echo(f"\nPlan flagged: {plan.decision_reason}")
        return click.confirm("Proceed anyway?", default=False)

    def run_executor(claim, artifact, claim_dir):
        raise RuntimeError("real GPU executor not implemented (Plan 2c)")

    result = run_pipeline(
        input_arg=input_arg, base_dir=Path(base_dir), timestamp=_timestamp(),
        available_hardware=[], approve_spec=approve_spec, approve_plan=approve_plan,
        fetch_sources=make_fetch_sources(), setup_executor=None, run_executor=run_executor,
    )
    if result.aborted_at:
        click.echo(f"Aborted at: {result.aborted_at}")
    else:
        click.echo(f"Done. Report: {result.root}/report.zh.md")
```

Note: the old inline `fetch_sources` stub is deleted; `make_fetch_sources()` replaces it. The `run_executor` message updates `Plan 2` → `Plan 2c` for accuracy.

- [ ] **Step 4: Run the tests, confirm they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS, all CLI tests green (the pre-existing `test_cli_help`, `test_cli_run_help_lists_yes_flag`, `test_cli_report_rerenders` still pass; the two new ones pass)

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/cli.py tests/test_cli.py
git commit -m "feat: wire real fetch_sources + title resolver into CLI run"
```

---

## Task 8: full suite + ruff + offline guarantee check

**Files:**
- Test: reuse existing

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS, all tests green (Plan 1's ~59 + the new fetch/cli tests). No test makes a real network call (all inject fakes / monkeypatch).

- [ ] **Step 2: Confirm no accidental real network in tests**

Run: `uv run pytest -q -p no:cacheprovider`
Expected: PASS in well under a few seconds — if any test hit the real network it would be slow/flaky. Visually confirm `tests/test_fetch.py` and the new `tests/test_cli.py` cases all pass `http_get=` / `git_clone=` fakes or monkeypatch `cli_mod.resolve_arxiv_id` / `cli_mod.make_fetch_sources`.

- [ ] **Step 3: ruff check**

Run: `uv run ruff check src/ tests/`
Expected: "All checks passed!" — fix any F401/E702 in the new files (do not introduce unused imports; `re as _re` in cli.py is used by the resolver guard).

- [ ] **Step 4: Commit (if ruff made changes)**

```bash
git add -A
git commit -m "chore: ruff clean for Plan 2a ingest fetch"
```

---

## Self-Review

**1. Spec coverage** (design §3.1 Ingest):

- "Fetch LaTeX source (`arxiv.org/e-print/<id>`), do not OCR the PDF" → Task 1 (`latex_source_url`) + Task 3 (`fetch_latex`). ✓
- "Locate the official repo, priority: GitHub link in the paper > PapersWithCode > GH code search" → Task 6 uses `find_repo_url` over the fetched latex (the GitHub-link-in-paper tier). **PapersWithCode and GH code search tiers are NOT implemented in Plan 2a** — they are a deliberate scope cut: the in-paper GitHub link covers the large majority of cases, and the fallback tiers each need another API integration. This is noted, not an omission. A missing repo simply leaves `repo/` empty (pipeline already handles `repo: null`). ✓ (with documented partial scope)
- Title input deferred from Plan 1 → Task 4 (`resolve_arxiv_id`) + Task 7 (CLI wiring). ✓
- "Network fetches isolated behind injectable functions, offline-testable" → every network/subprocess call is behind `http_get` / `git_clone` injectables; all tests offline. ✓
- Path-traversal safety on tar extraction (not in spec but required for safely unpacking untrusted arxiv tarballs) → Task 1. ✓

**Deferred to later (not omissions):**
- PapersWithCode / GH-code-search repo discovery tiers (only the in-paper GitHub link tier is built).
- Writing `latex_path` / `repo_path` into `IngestInfo` — the pipeline constructs `IngestInfo` itself and Plan 2a intentionally does not change the pipeline signature; populating those fields is a small follow-up if needed. The directories are filled regardless.
- setup loop (Plan 2b), GPU executor (Plan 2c).

**2. Placeholder scan:** No TBD/TODO. Every code step has complete runnable code. The only "not implemented" is the explicit `run_executor` raise (carried over from Plan 1, correctly labeled Plan 2c).

**3. Type consistency:**
- `http_get` byte-getter (`Callable[[str], bytes]`) used by `fetch_latex` and `make_fetch_sources`; text-getter (`Callable[[str], str]`) used by `resolve_arxiv_id`. Two distinct getters by design (bytes for tarball, text for XML) — Task 6's factory only needs the bytes getter (latex) and the git_clone runner, consistent with its signature. ✓
- `git_clone` runner signature `(url: str, dest: str) -> None` consistent across `clone_repo` (Task 5) and `make_fetch_sources` (Task 6) and the CLI test fakes (Task 7). ✓
- `make_fetch_sources()` returns `fetch_sources(rd, arxiv_id, url)` — matches the pipeline's injected seam exactly (pipeline.py:43). ✓
- `resolve_arxiv_id(query, *, http_get=...)` — CLI calls it positionally with one arg (Task 7), matching the signature. ✓
