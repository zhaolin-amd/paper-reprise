from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(autouse=True)
def _block_real_claude(monkeypatch):
    """Safety net: any test that reaches the real `claude` subprocess fails loudly.

    Tests that legitimately exercise the headless path monkeypatch run_headless or
    extract_spec themselves; this only catches the ones that forgot to.
    """
    def _boom(*args, **kwargs):
        raise RuntimeError(
            "real claude subprocess invoked in a test — stub run_headless/extract_spec"
        )
    monkeypatch.setattr("paper_reprise.headless._call_claude", _boom, raising=True)
