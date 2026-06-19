import shutil
from pathlib import Path

import paper_repro.specextract as specextract
from paper_repro.headless import HeadlessResult
from paper_repro.rundir import RunDir

FIX = Path(__file__).parent / "fixtures"


def test_specextract_produces_valid_spec(tmp_path, monkeypatch):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    (rd.paper_dir / "main.tex").write_text("dummy latex")

    def fake_headless(prompt, allowed_tools, cwd, expect_file):
        shutil.copy(FIX / "extracted_spec.yaml", expect_file)
        return HeadlessResult(ok=True, output_path=expect_file)

    monkeypatch.setattr(specextract, "run_headless", fake_headless)
    spec = specextract.extract_spec(rd)
    assert spec is not None
    assert spec.claims[0].id == "c1"
    assert (rd.root / "spec.yaml").exists()


def test_specextract_returns_none_when_headless_fails(tmp_path, monkeypatch):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    monkeypatch.setattr(specextract, "run_headless",
                        lambda **k: HeadlessResult(ok=False, error="boom"))
    assert specextract.extract_spec(rd) is None


def test_specextract_returns_none_on_invalid_yaml(tmp_path, monkeypatch):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")

    def fake_headless(prompt, allowed_tools, cwd, expect_file):
        Path(expect_file).write_text("paper: x\nclaims: [bad]\n")
        return HeadlessResult(ok=True, output_path=expect_file)

    monkeypatch.setattr(specextract, "run_headless", fake_headless)
    assert specextract.extract_spec(rd) is None
