from pathlib import Path

from paper_reprise.ingest import (
    normalize_input, parse_org, find_repo_url, arxiv_id_from_url,
)

FIX = Path(__file__).parent / "fixtures"


def test_parse_org_extracts_source_and_meta():
    meta = parse_org((FIX / "sample.org").read_text())
    assert meta["source"] == "https://arxiv.org/abs/2606.18114"
    assert meta["title"].startswith("Ternary Mamba")
    assert "Alice Smith" in meta["authors"]


def test_arxiv_id_from_abs_url():
    assert arxiv_id_from_url("https://arxiv.org/abs/2606.18114") == "2606.18114"


def test_arxiv_id_from_versioned_url():
    assert arxiv_id_from_url("https://arxiv.org/abs/2401.00001v2") == "2401.00001"


def test_normalize_input_from_org_file(tmp_path):
    f = tmp_path / "x.org"
    f.write_text("#+source: https://arxiv.org/abs/2401.00001\n")
    arxiv_id, url = normalize_input(str(f))
    assert arxiv_id == "2401.00001"
    assert url == "https://arxiv.org/abs/2401.00001"


def test_normalize_input_from_bare_id():
    arxiv_id, url = normalize_input("2401.00001")
    assert arxiv_id == "2401.00001"
    assert url == "https://arxiv.org/abs/2401.00001"


def test_normalize_input_from_abs_url():
    arxiv_id, url = normalize_input("https://arxiv.org/abs/2401.00001")
    assert arxiv_id == "2401.00001"


def test_find_repo_url_picks_github_link():
    text = "We release code at https://github.com/foo/bar for reproduction."
    assert find_repo_url(text) == "https://github.com/foo/bar"


def test_find_repo_url_none_when_absent():
    assert find_repo_url("no links here") is None
