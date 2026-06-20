import io
import tarfile
import urllib.parse
from pathlib import Path

from paper_reprise.fetch import (
    clone_repo,
    fetch_arxiv_title,
    fetch_latex,
    latex_source_url,
    make_fetch_sources,
    parse_arxiv_search,
    parse_arxiv_title,
    resolve_arxiv_id,
    short_name,
    unpack_targz,
)
from paper_reprise.rundir import RunDir

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
    with pytest.raises(ValueError, match="unsafe"):
        unpack_targz(buf.getvalue(), tmp_path)


def test_unpack_targz_rejects_symlink_escape(tmp_path):
    import pytest
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # a symlink member whose target escapes dest
        info = tarfile.TarInfo(name="evil")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../escape"
        tar.addfile(info)
    with pytest.raises(ValueError, match="unsafe"):
        unpack_targz(buf.getvalue(), tmp_path)


def test_unpack_targz_rejects_too_many_members(tmp_path):
    import pytest
    import paper_reprise.fetch as fetch_mod
    monkey_max = 3
    orig = fetch_mod._MAX_MEMBERS
    fetch_mod._MAX_MEMBERS = monkey_max
    try:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for i in range(monkey_max + 1):
                data = b"x"
                ti = tarfile.TarInfo(name=f"f{i}.tex")
                ti.size = len(data)
                tar.addfile(ti, io.BytesIO(data))
        with pytest.raises(ValueError, match="too many members"):
            unpack_targz(buf.getvalue(), tmp_path)
    finally:
        fetch_mod._MAX_MEMBERS = orig


def test_unpack_targz_rejects_oversize(tmp_path):
    import pytest
    import paper_reprise.fetch as fetch_mod
    orig = fetch_mod._MAX_UNPACK_BYTES
    fetch_mod._MAX_UNPACK_BYTES = 10
    try:
        blob = _make_targz({"main.tex": "x" * 100})
        with pytest.raises(ValueError, match="too large"):
            unpack_targz(blob, tmp_path)
    finally:
        fetch_mod._MAX_UNPACK_BYTES = orig


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


def test_parse_arxiv_title_first_entry():
    xml = (FIX / "arxiv_search_response.xml").read_text()
    assert parse_arxiv_title(xml) == "AWQ: Activation-aware Weight Quantization for LLMs"


def test_parse_arxiv_title_empty_feed_none():
    assert parse_arxiv_title('<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>') is None


def test_fetch_arxiv_title_best_effort_swallows_errors():
    def boom(url):
        raise RuntimeError("network down")
    assert fetch_arxiv_title("2401.00001", http_get=boom) is None


def test_fetch_arxiv_title_happy(tmp_path):
    xml = (FIX / "arxiv_search_response.xml").read_text()
    assert fetch_arxiv_title("2401.00001", http_get=lambda u: xml) == \
        "AWQ: Activation-aware Weight Quantization for LLMs"


def test_short_name_takes_acronym_before_colon():
    assert short_name("GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs") == "GSQ"


def test_short_name_one_word_acronym_title():
    assert short_name("SmoothQuant: Accurate and Efficient Post-Training Quantization") == "SmoothQuant"


def test_short_name_keeps_full_title_when_prefix_is_long():
    t = "The case for 4-bit precision: k-bit inference scaling laws"
    assert short_name(t) == t


def test_short_name_no_colon_returns_title():
    assert short_name("Attention Is All You Need") == "Attention Is All You Need"
