from paper_reprise.ingest import (
    normalize_input, find_repo_url, arxiv_id_from_url,
)


def test_arxiv_id_from_abs_url():
    assert arxiv_id_from_url("https://arxiv.org/abs/2606.18114") == "2606.18114"


def test_arxiv_id_from_versioned_url():
    assert arxiv_id_from_url("https://arxiv.org/abs/2401.00001v2") == "2401.00001"


def test_normalize_input_from_bare_id():
    arxiv_id, url = normalize_input("2401.00001")
    assert arxiv_id == "2401.00001"
    assert url == "https://arxiv.org/abs/2401.00001"


def test_normalize_input_from_abs_url():
    arxiv_id, url = normalize_input("https://arxiv.org/abs/2401.00001")
    assert arxiv_id == "2401.00001"


def test_normalize_input_strips_version_suffix():
    arxiv_id, url = normalize_input("2401.00001v2")
    assert arxiv_id == "2401.00001"
    assert url == "https://arxiv.org/abs/2401.00001"


def test_find_repo_url_picks_github_link():
    text = "We release code at https://github.com/foo/bar for reproduction."
    assert find_repo_url(text) == "https://github.com/foo/bar"


def test_find_repo_url_none_when_absent():
    assert find_repo_url("no links here") is None
