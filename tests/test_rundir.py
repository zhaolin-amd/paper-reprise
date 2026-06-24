from paper_reprise.models import IngestInfo, Spec, Artifact, Claim, EvalProtocol
from paper_reprise.rundir import RunDir


def _spec():
    return Spec(
        paper="2401.00001", repo=None,
        artifacts=[Artifact(id="a1", base_model="m", method="AWQ", quant_config={"wbits": 4})],
        claims=[Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="official", command="c",
                                                 metric="perplexity", dataset="wikitext2"),
                      expected=5.78, tolerance=0.05, source="T")],
    )


def test_create_makes_layout(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="20260619-101500")
    assert rd.root.name == "2401.00001-20260619-101500"
    assert rd.root.is_dir()
    assert rd.claim_dir("c1").parent == rd.runs_dir


def test_write_then_read_ingest(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    info = IngestInfo(arxiv_id="2401.00001", source_url="https://arxiv.org/abs/2401.00001")
    rd.write_ingest(info)
    assert (rd.root / "ingest.json").exists()
    got = rd.read_ingest()
    assert got.arxiv_id == "2401.00001"


def test_write_then_read_spec_yaml(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    rd.write_spec(_spec())
    assert (rd.root / "spec.yaml").exists()
    got = rd.read_spec()
    assert got.claims[0].id == "c1"


def test_open_existing(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    rd2 = RunDir.open(rd.root)
    assert rd2.root == rd.root


def test_read_missing_spec_returns_none(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    assert rd.read_spec() is None


def test_create_with_name_prepends_slug(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t",
                       name="GSQ: Gumbel-Softmax Quantization!")
    assert rd.root.name == "gsq-gumbel-softmax-quantization-2401.00001-t"


def test_create_with_empty_name_falls_back(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t", name="   ")
    assert rd.root.name == "2401.00001-t"


def test_create_name_is_truncated(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t",
                       name="a" * 100)
    # slug truncated to 40 chars before the id
    assert rd.root.name == ("a" * 40) + "-2401.00001-t"


def test_clean_model_artifacts_removes_weights_keeps_records(tmp_path):
    from paper_reprise.rundir import RunDir
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    big = b"\0" * (11 * 1024 * 1024)
    ck = rd.repo_dir / "runtime" / "checkpoints" / "m" / "assembled"
    ck.mkdir(parents=True)
    (ck / "model.safetensors").write_bytes(big)
    (ck / "extra.bin").write_bytes(big)
    (ck / "config.json").write_text("{}")        # non-weight → kept
    tiny = ck / "tiny.pt"
    tiny.write_bytes(b"\0" * 1024)               # weight ext but under floor → kept
    (rd.root / "spec.yaml").write_text("paper: p")
    sd = rd.claim_dir("c1")
    (sd / "stdout.log").write_text("perplexity: 5.8")

    removed = rd.clean_model_artifacts()
    names = {n for n, _ in removed}
    assert any(n.endswith("model.safetensors") for n in names)
    assert any(n.endswith("extra.bin") for n in names)
    assert not (ck / "model.safetensors").exists()
    assert not (ck / "extra.bin").exists()
    # kept: under-floor weight, non-weight, and all records
    assert tiny.exists()
    assert (ck / "config.json").exists()
    assert (rd.root / "spec.yaml").read_text() == "paper: p"
    assert (sd / "stdout.log").exists()


def test_clean_env_removes_envs_keeps_records(tmp_path):
    from paper_reprise.rundir import RunDir
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    (rd.root / "env" / "bin").mkdir(parents=True)
    (rd.root / "env" / "bin" / "python").write_bytes(b"\0" * 2048)
    (rd.repo_dir / ".venv" / "lib").mkdir(parents=True)
    (rd.repo_dir / ".venv" / "lib" / "pkg.so").write_bytes(b"\0" * 4096)
    (rd.root / "env_snapshot.json").write_text("{}")       # record, kept

    removed = {r for r, _ in rd.clean_env()}
    assert "env" in removed and "repo/.venv" in removed
    assert not (rd.root / "env").exists()
    assert not (rd.repo_dir / ".venv").exists()
    assert (rd.root / "env_snapshot.json").exists()
