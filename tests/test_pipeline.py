import shutil
from pathlib import Path

import paper_reprise.pipeline as pipeline
from paper_reprise.setupstage import SetupResult

FIX = Path(__file__).parent / "fixtures"


def _fake_specextract(rd):
    shutil.copy(FIX / "extracted_spec.yaml", rd.root / "spec.yaml")
    import yaml
    from paper_reprise.models import Spec
    return Spec.model_validate(yaml.safe_load((rd.root / "spec.yaml").read_text()))


def _fake_setup(rd, spec):
    return SetupResult(ok=True, env_snapshot={"torch": "2.3", "transformers": "4.36",
                                              "cuda": "12.1"}, patches=[])


def _fake_executor(claim, artifact, claim_dir):
    log = Path(claim_dir) / "stdout.log"
    log.write_text("perplexity: 5.80")
    return {"stdout_path": str(log), "actual_config": {"seqlen": 2048},
            "gpu": "A100x1", "seed": 0, "minutes": 1.0}


def test_full_pipeline_produces_match_report(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "extract_spec", _fake_specextract)
    result = pipeline.run_pipeline(
        input_arg="2401.00001", base_dir=tmp_path, timestamp="t",
        available_hardware=["A100-80G"],
        approve_spec=lambda spec: True,
        approve_plan=lambda plan: True,
        fetch_sources=lambda rd, arxiv_id, url: None,
        setup_executor=_fake_setup,
        run_executor=_fake_executor,
    )
    assert (result.root / "report.zh.md").exists()
    assert (result.root / "report.en.md").exists()
    assert "MATCH 1" in (result.root / "report.zh.md").read_text()


def test_pipeline_aborts_when_spec_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "extract_spec", _fake_specextract)
    result = pipeline.run_pipeline(
        input_arg="2401.00001", base_dir=tmp_path, timestamp="t",
        available_hardware=["A100-80G"],
        approve_spec=lambda spec: False,
        approve_plan=lambda plan: True,
        fetch_sources=lambda rd, arxiv_id, url: None,
        setup_executor=_fake_setup, run_executor=_fake_executor,
    )
    assert not (result.root / "report.zh.md").exists()
    assert result.aborted_at == "spec-approval"


def test_pipeline_aborts_when_specextract_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "extract_spec", lambda rd: None)
    result = pipeline.run_pipeline(
        input_arg="2401.00001", base_dir=tmp_path, timestamp="t",
        available_hardware=["A100-80G"],
        approve_spec=lambda spec: True,
        approve_plan=lambda plan: True,
        fetch_sources=lambda rd, arxiv_id, url: None,
        setup_executor=_fake_setup, run_executor=_fake_executor,
    )
    assert result.aborted_at == "specextract"
    assert not (result.root / "report.zh.md").exists()


def test_pipeline_aborts_at_plan_gate(tmp_path, monkeypatch):
    def spec_needs_hw(rd):
        import shutil as _sh
        import yaml
        from paper_reprise.models import Spec
        _sh.copy(FIX / "extracted_spec.yaml", rd.root / "spec.yaml")
        spec = Spec.model_validate(yaml.safe_load((rd.root / "spec.yaml").read_text()))
        spec.claims[0].hardware = "H200-141G x8"
        return spec

    monkeypatch.setattr(pipeline, "extract_spec", spec_needs_hw)
    result = pipeline.run_pipeline(
        input_arg="2401.00001", base_dir=tmp_path, timestamp="t",
        available_hardware=["A100-80G"],
        approve_spec=lambda spec: True,
        approve_plan=lambda plan: False,
        fetch_sources=lambda rd, arxiv_id, url: None,
        setup_executor=_fake_setup, run_executor=_fake_executor,
    )
    assert result.aborted_at == "plan"
    assert not (result.root / "report.zh.md").exists()


def test_pipeline_aborts_when_setup_fails(tmp_path, monkeypatch):
    from paper_reprise.setupstage import SetupResult
    monkeypatch.setattr(pipeline, "extract_spec", _fake_specextract)
    result = pipeline.run_pipeline(
        input_arg="2401.00001", base_dir=tmp_path, timestamp="t",
        available_hardware=["A100-80G"],
        approve_spec=lambda spec: True,
        approve_plan=lambda plan: True,
        fetch_sources=lambda rd, arxiv_id, url: None,
        setup_executor=lambda rd, spec: SetupResult(ok=False),
        run_executor=_fake_executor,
    )
    assert result.aborted_at == "setup"
    assert not (result.root / "report.zh.md").exists()


def test_resume_continues_from_existing_spec(tmp_path, monkeypatch):
    # first, create a run dir with spec.yaml + ingest.json already present
    import shutil
    from paper_reprise.models import IngestInfo
    from paper_reprise.rundir import RunDir
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    shutil.copy(FIX / "extracted_spec.yaml", rd.root / "spec.yaml")
    rd.write_ingest(IngestInfo(arxiv_id="2401.00001",
                               source_url="https://arxiv.org/abs/2401.00001"))

    result = pipeline.resume_pipeline(
        rd.root, available_hardware=["A100-80G"],
        approve_plan=lambda plan: True,
        setup_executor=_fake_setup, run_executor=_fake_executor,
    )
    assert result.aborted_at is None
    assert (result.root / "report.zh.md").exists()
    assert "MATCH 1" in (result.root / "report.zh.md").read_text()


def test_resume_aborts_when_no_spec(tmp_path):
    from paper_reprise.rundir import RunDir
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")  # no spec.yaml
    result = pipeline.resume_pipeline(
        rd.root, available_hardware=[], approve_plan=lambda p: True,
        setup_executor=_fake_setup, run_executor=_fake_executor,
    )
    assert result.aborted_at == "no-spec"


def test_pipeline_uses_paper_name_in_run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "extract_spec", _fake_specextract)
    result = pipeline.run_pipeline(
        input_arg="2401.00001", base_dir=tmp_path, timestamp="t",
        paper_name="Awesome Quant Method",
        available_hardware=["A100-80G"],
        approve_spec=lambda spec: True, approve_plan=lambda plan: True,
        fetch_sources=lambda rd, arxiv_id, url: None,
        setup_executor=_fake_setup, run_executor=_fake_executor,
    )
    assert result.root.name == "awesome-quant-method-2401.00001-t"


def _executor_with_model(claim, artifact, claim_dir):
    cd = Path(claim_dir)
    log = cd / "stdout.log"
    log.write_text("perplexity: 5.80")
    # drop a big "exported model" weight under the run dir (repo runtime)
    mdir = cd.parent.parent / "repo" / "runtime" / "checkpoints" / "assembled"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "model.safetensors").write_bytes(b"\0" * (11 * 1024 * 1024))
    return {"stdout_path": str(log), "actual_config": {"seqlen": 2048},
            "gpu": "A100x1", "seed": 0, "minutes": 1.0}


def _run(tmp_path, monkeypatch, *, clean_models):
    monkeypatch.setattr(pipeline, "extract_spec", _fake_specextract)
    return pipeline.run_pipeline(
        input_arg="2401.00001", base_dir=tmp_path, timestamp="t",
        available_hardware=["A100-80G"], approve_spec=lambda s: True,
        approve_plan=lambda p: True, fetch_sources=lambda rd, a, u: None,
        setup_executor=_fake_setup, run_executor=_executor_with_model,
        clean_models=clean_models)


def test_clean_models_removes_exported_weights_keeps_report(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, clean_models=True)
    wt = result.root / "repo/runtime/checkpoints/assembled/model.safetensors"
    assert not wt.exists()                       # exported model cleaned
    assert result.cleaned                        # freed list reported
    assert (result.root / "report.zh.md").exists()   # records kept
    assert (result.root / "runs/c1/stdout.log").exists()


def test_keep_models_leaves_exported_weights(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, clean_models=False)
    wt = result.root / "repo/runtime/checkpoints/assembled/model.safetensors"
    assert wt.exists()                           # kept
    assert not result.cleaned
