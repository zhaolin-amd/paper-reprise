import io
import tarfile
from pathlib import Path

from paper_reprise.fetch import (
    fetch_latex,
    latex_source_url,
    parse_arxiv_search,
    unpack_targz,
)

FIX = Path(__file__).parent / "fixtures"


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


def test_parse_arxiv_search_returns_first_id():
    xml = (FIX / "arxiv_search_response.xml").read_text()
    assert parse_arxiv_search(xml) == "2401.00001"


def test_parse_arxiv_search_empty_feed_returns_none():
    xml = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    assert parse_arxiv_search(xml) is None


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
