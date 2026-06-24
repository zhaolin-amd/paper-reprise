from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(autouse=True)
def _models_dir_on_tmp(monkeypatch, tmp_path):
    """Point the exported-models scratch base at a tmp dir so tests never create or
    touch real /scratch/$USER/paper-reprise-models (the production default)."""
    monkeypatch.setenv("PAPER_REPRISE_MODELS_DIR", str(tmp_path / "models-scratch"))


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch):
    """Safety net: any test that reaches a real `claude` subprocess or a real HTTP
    fetch fails loudly. Tests that legitimately exercise those paths inject fakes
    (http_get=..., run_eval=...) or monkeypatch run_headless/extract_spec/
    fetch_arxiv_title themselves; this only catches the ones that forgot to.
    Best-effort callers (e.g. fetch_arxiv_title) swallow the raised error and
    degrade, so the suite stays offline and fast either way.
    """
    def _boom_claude(*args, **kwargs):
        raise RuntimeError(
            "real claude subprocess invoked in a test — stub run_headless/extract_spec"
        )

    def _boom_http(*args, **kwargs):
        raise RuntimeError(
            "real HTTP request invoked in a test — inject http_get= or monkeypatch it"
        )

    monkeypatch.setattr("paper_reprise.headless._call_claude", _boom_claude, raising=True)
    monkeypatch.setattr("paper_reprise.fetch._http_get_text", _boom_http, raising=True)
    monkeypatch.setattr("paper_reprise.fetch._http_get_bytes", _boom_http, raising=True)
