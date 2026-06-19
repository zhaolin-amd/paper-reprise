from pathlib import Path

import paper_repro.headless as headless
from paper_repro.headless import run_headless


def test_success_when_output_file_appears(tmp_path, monkeypatch):
    out = tmp_path / "spec.yaml"

    def fake_call(prompt, allowed_tools, cwd):
        out.write_text("ok")
        return 0

    monkeypatch.setattr(headless, "_call_claude", fake_call)
    res = run_headless(prompt="make spec", allowed_tools=["Write"],
                       cwd=tmp_path, expect_file=out)
    assert res.ok is True
    assert res.output_path == out


def test_failure_when_output_missing(tmp_path, monkeypatch):
    out = tmp_path / "spec.yaml"
    monkeypatch.setattr(headless, "_call_claude", lambda *a, **k: 0)
    res = run_headless(prompt="x", allowed_tools=["Write"], cwd=tmp_path, expect_file=out)
    assert res.ok is False
    assert "did not appear" in res.error


def test_failure_when_nonzero_and_missing(tmp_path, monkeypatch):
    out = tmp_path / "spec.yaml"
    monkeypatch.setattr(headless, "_call_claude", lambda *a, **k: 3)
    res = run_headless(prompt="x", allowed_tools=["Write"], cwd=tmp_path, expect_file=out)
    assert res.ok is False
