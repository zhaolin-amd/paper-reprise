from paper_reprise.models import Artifact, Claim, EvalProtocol
from paper_reprise.runexec import (
    _detect_gpu,
    _run_eval,
    build_eval_command,
    extract_seed,
    resolve_actual_config,
)


def _claim(command="python eval.py --model m", seqlen=2048, stride=2048, few_shot=0):
    return Claim(
        id="c1", artifact="a1",
        eval_protocol=EvalProtocol(runner="official", command=command,
                                   metric="perplexity", dataset="wikitext2",
                                   seqlen=seqlen, stride=stride, few_shot=few_shot),
        expected=5.78, tolerance=0.05, source="T",
    )


def _artifact(wbits=4, group_size=128):
    qc = {"wbits": wbits}
    if group_size is not None:
        qc["group_size"] = group_size
    return Artifact(id="a1", base_model="m", method="AWQ", quant_config=qc)


def test_build_eval_command_returns_protocol_command():
    assert build_eval_command(_claim("python main.py --reproduce")) == "python main.py --reproduce"


def test_extract_seed_dash_flag():
    assert extract_seed("python eval.py --seed 42 --model m") == 42


def test_extract_seed_equals_form():
    assert extract_seed("python eval.py seed=7") == 7


def test_extract_seed_absent_returns_none():
    assert extract_seed("python eval.py --model m") is None


def test_resolve_actual_config_matches_grade_keys():
    cfg = resolve_actual_config(_claim(seqlen=2048, stride=1024, few_shot=5), _artifact(wbits=4, group_size=128))
    assert cfg == {"seqlen": 2048, "stride": 1024, "few_shot": 5, "wbits": 4, "group_size": 128}


def test_resolve_actual_config_omits_absent_optional_keys():
    # no stride on the protocol, no group_size on the artifact → those keys omitted
    c = _claim(seqlen=2048, stride=None, few_shot=0)
    cfg = resolve_actual_config(c, _artifact(wbits=8, group_size=None))
    assert cfg == {"seqlen": 2048, "few_shot": 0, "wbits": 8}


def test_run_eval_persists_output_and_returns_exit_code(tmp_path):
    log = tmp_path / "stdout.log"
    env_dir = tmp_path / "env"
    (env_dir / "bin").mkdir(parents=True)
    code, out = _run_eval("echo hello-run", cwd=tmp_path, env_dir=env_dir, log_path=log)
    assert code == 0
    assert "hello-run" in out
    assert "hello-run" in log.read_text()   # persisted to disk for grade


def test_run_eval_nonzero_exit_is_captured(tmp_path):
    log = tmp_path / "stdout.log"
    env_dir = tmp_path / "env"
    (env_dir / "bin").mkdir(parents=True)
    code, out = _run_eval("exit 3", cwd=tmp_path, env_dir=env_dir, log_path=log)
    assert code == 3


def test_detect_gpu_returns_string_or_unknown(monkeypatch):
    # with no nvidia-smi and no CUDA_VISIBLE_DEVICES, falls back to "unknown"
    import paper_reprise.runexec as runexec
    monkeypatch.setattr(runexec.shutil, "which", lambda name: None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _detect_gpu() == "unknown"
