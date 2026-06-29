import json
from pathlib import Path

from paper_reprise.models import Artifact, Claim, EvalProtocol, Spec
from paper_reprise.rundir import RunDir
from paper_reprise.runstage import run_claims
from paper_reprise.runexec import (
    _detect_gpu,
    _run_eval,
    build_eval_command,
    extract_seed,
    make_run_executor,
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


def test_run_eval_reaps_orphaned_background_process(tmp_path):
    # The eval command exits immediately but leaves a background "server" alive (the
    # vLLM-leak shape). `set -m` puts that server in its OWN process group within the
    # session — exactly what vLLM does — so a group-only kill on the leader would miss
    # it; _run_eval reaps the whole SESSION, so the orphan must NOT survive the call.
    import subprocess as sp
    import time

    log = tmp_path / "stdout.log"
    env_dir = tmp_path / "env"
    (env_dir / "bin").mkdir(parents=True)
    marker = "paper_reprise_orphan_2718281828"   # unique so we match only our child
    code, _ = _run_eval(f"set -m; ( sleep 120 ; echo {marker} ) & echo started",
                        cwd=tmp_path, env_dir=env_dir, log_path=log)
    assert code == 0
    time.sleep(0.6)
    found = sp.run(["pgrep", "-fa", marker], capture_output=True, text=True)
    if found.returncode == 0:        # safety: don't leak if the assert is about to fail
        for line in found.stdout.splitlines():
            pid = line.split(" ", 1)[0]
            if pid.isdigit():
                sp.run(["kill", "-9", pid])
    assert found.returncode != 0, f"orphan leaked: {found.stdout!r}"


def test_detect_gpu_returns_unknown_when_no_tools(monkeypatch):
    # no nvidia-smi / amd-smi / rocm-smi anywhere and no *_VISIBLE_DEVICES → "unknown"
    import paper_reprise.runexec as runexec
    monkeypatch.setattr(runexec, "_resolve_tool", lambda name, cands: None)
    for var in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    assert _detect_gpu() == "unknown"


def test_detect_gpu_nvidia(monkeypatch):
    import paper_reprise.runexec as runexec
    monkeypatch.setattr(runexec, "_resolve_tool",
                        lambda name, cands: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(runexec, "_run_tool", lambda cmd: "NVIDIA H200\nNVIDIA H200\n")
    assert _detect_gpu() == "NVIDIA H200"


def test_detect_gpu_amd_instinct_via_amd_smi(monkeypatch):
    # AMD box: no nvidia-smi, amd-smi reports an Instinct MI300X
    import paper_reprise.runexec as runexec
    monkeypatch.setattr(runexec, "_resolve_tool",
                        lambda name, cands: "/opt/rocm/bin/amd-smi" if name == "amd-smi" else None)
    monkeypatch.setattr(runexec, "_run_tool",
                        lambda cmd: "ASIC:\n    market_name: Instinct MI300X\n")
    assert _detect_gpu() == "AMD Instinct MI300X"


def test_detect_gpu_amd_via_rocm_smi(monkeypatch):
    import paper_reprise.runexec as runexec
    monkeypatch.setattr(runexec, "_resolve_tool",
                        lambda name, cands: "/opt/rocm/bin/rocm-smi" if name == "rocm-smi" else None)
    monkeypatch.setattr(runexec, "_run_tool", lambda cmd: "Card series: AMD INSTINCT MI355X\n")
    assert _detect_gpu() == "AMD Instinct MI355X"


def test_detect_gpu_amd_via_rocm_smi_csv_form(monkeypatch):
    # --showproductname returns nothing parseable; the --csv form carries the token
    import paper_reprise.runexec as runexec
    monkeypatch.setattr(runexec, "_resolve_tool",
                        lambda name, cands: "/opt/rocm/bin/rocm-smi" if name == "rocm-smi" else None)

    def fake_run(cmd):
        return "name\nAMD Instinct MI355X\n" if "--csv" in cmd else "unsupported\n"

    monkeypatch.setattr(runexec, "_run_tool", fake_run)
    assert _detect_gpu() == "AMD Instinct MI355X"


def test_detect_gpu_nvitop_fallback_normalizes_amd(monkeypatch):
    # no CLI tools resolve → fall through to the optional nvitop backend (NVML+amdsmi)
    import sys
    import types
    import paper_reprise.runexec as runexec
    monkeypatch.setattr(runexec, "_resolve_tool", lambda name, cands: None)
    for var in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)

    class _Dev:
        def name(self):
            return "Instinct MI300X"   # amdsmi market name (no 'AMD' prefix)

    fake = types.ModuleType("nvitop")
    fake.Device = type("Device", (), {"all": staticmethod(lambda: [_Dev()])})
    monkeypatch.setitem(sys.modules, "nvitop", fake)
    assert _detect_gpu() == "AMD Instinct MI300X"


def test_detect_gpu_nvitop_passes_through_nvidia(monkeypatch):
    import sys
    import types
    import paper_reprise.runexec as runexec
    monkeypatch.setattr(runexec, "_resolve_tool", lambda name, cands: None)
    for var in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)

    class _Dev:
        def name(self):
            return "NVIDIA H200"

    fake = types.ModuleType("nvitop")
    fake.Device = type("Device", (), {"all": staticmethod(lambda: [_Dev()])})
    monkeypatch.setitem(sys.modules, "nvitop", fake)
    assert _detect_gpu() == "NVIDIA H200"


def test_detect_available_hardware(monkeypatch):
    import paper_reprise.runexec as runexec
    monkeypatch.setattr(runexec, "_detect_gpu", lambda: "AMD Instinct MI350X")
    assert runexec.detect_available_hardware() == ["AMD Instinct MI350X"]
    monkeypatch.setattr(runexec, "_detect_gpu", lambda: "unknown")
    assert runexec.detect_available_hardware() == []
    # a bare visible-devices hint has no family → treated as unknown
    monkeypatch.setattr(runexec, "_detect_gpu", lambda: "HIP_VISIBLE_DEVICES=0,1")
    assert runexec.detect_available_hardware() == []


def test_executor_runs_persists_and_returns_metadata(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    claim_dir = rd.claim_dir("c1")
    calls = {}

    def fake_run_eval(command, cwd, env_dir, log_path):
        calls["command"] = command
        calls["cwd"] = cwd
        calls["env_dir"] = env_dir
        Path(log_path).write_text("perplexity: 5.80")
        return 0, "perplexity: 5.80"

    executor = make_run_executor(
        run_eval=fake_run_eval, detect_gpu=lambda: "A100",
        now=iter([100.0, 220.0]).__next__,
    )
    out = executor(_claim("python eval.py --seed 42"), _artifact(), claim_dir)

    # derived paths
    assert calls["cwd"] == rd.repo_dir
    assert calls["env_dir"] == rd.root / "env"
    # returned metadata
    assert out["stdout_path"] == str(claim_dir / "stdout.log")
    assert out["gpu"] == "A100"
    assert out["seed"] == 42
    assert out["minutes"] == 2.0                      # (220-100)/60
    assert out["actual_config"]["wbits"] == 4
    # actual_config persisted for the report re-render
    saved = json.loads((claim_dir / "actual_config.json").read_text())
    assert saved["seqlen"] == 2048


def test_executor_resolves_model_into_command(tmp_path, monkeypatch):
    # end-to-end: the command handed to the seam is model-resolved ({model}
    # substituted with the snapshot path + PAPER_REPRISE_MODEL exported), not just
    # the bare protocol command. Guards against dropping resolved_command.
    base = tmp_path / "cache"
    snap = base / "meta-llama" / "Llama-3.2-1B"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    monkeypatch.setenv("PAPER_REPRISE_MODEL_BASE", str(base))
    monkeypatch.setenv("PAPER_REPRISE_DOWNLOAD_DIR", str(tmp_path / "dl"))

    rd = RunDir.create(tmp_path / "run", arxiv_id="p", timestamp="t")
    claim_dir = rd.claim_dir("c1")
    seen = {}

    def fake_run_eval(command, cwd, env_dir, log_path):
        seen["command"] = command
        Path(log_path).write_text("perplexity: 5.80")
        return 0, ""

    artifact = Artifact(id="a1", base_model="meta-llama/Llama-3.2-1B", method="AWQ",
                        quant_config={"wbits": 4})
    executor = make_run_executor(run_eval=fake_run_eval, detect_gpu=lambda: "H200",
                                 now=iter([0.0, 60.0]).__next__)
    executor(_claim("python eval.py --model {model}"), artifact, claim_dir)

    assert seen["command"] == (
        f"export PAPER_REPRISE_MODEL={snap}; python eval.py --model {snap}")


def _spec_one(command="python eval.py"):
    return Spec(paper="p", repo=None, artifacts=[_artifact()],
                claims=[_claim(command)])


def test_executor_injects_tasks_override(tmp_path):
    # run/resume --tasks exports PAPER_REPRISE_TASKS ahead of the eval command
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    claim_dir = rd.claim_dir("c1")
    seen = {}

    def fake_run_eval(command, cwd, env_dir, log_path):
        seen["command"] = command
        Path(log_path).write_text("perplexity: 5.8")
        return 0, ""

    executor = make_run_executor(tasks="arc_easy,piqa", run_eval=fake_run_eval,
                                 detect_gpu=lambda: "X", now=iter([0.0, 1.0]).__next__)
    out = executor(_claim("python eval.py"), _artifact(), claim_dir)
    assert seen["command"].startswith("export PAPER_REPRISE_TASKS=arc_easy,piqa;")
    assert out["command"].startswith("export PAPER_REPRISE_TASKS=arc_easy,piqa;")


def test_executor_injects_gpus_override(tmp_path):
    # run/resume --gpus N exports PAPER_REPRISE_GPUS ahead of the eval command
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    claim_dir = rd.claim_dir("c1")
    seen = {}

    def fake_run_eval(command, cwd, env_dir, log_path):
        seen["command"] = command
        Path(log_path).write_text("perplexity: 5.8")
        return 0, ""

    executor = make_run_executor(gpus=4, run_eval=fake_run_eval,
                                 detect_gpu=lambda: "X", now=iter([0.0, 1.0]).__next__)
    executor(_claim("python eval.py"), _artifact(), claim_dir)
    assert "export PAPER_REPRISE_GPUS=4;" in seen["command"]


def test_nonzero_eval_becomes_blocked_via_run_claims(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")

    def fail_eval(command, cwd, env_dir, log_path):
        Path(log_path).write_text("Traceback: CUDA out of memory")
        return 1, "Traceback: CUDA out of memory"

    executor = make_run_executor(run_eval=fail_eval, detect_gpu=lambda: "A100",
                                 now=iter([0.0, 60.0]).__next__)
    results, configs = run_claims(rd, _spec_one(), executor=executor)

    assert results[0].status == "blocked"
    assert "eval exited 1" in results[0].block_reason
    # stdout was still persisted (grade/report can show the traceback)
    assert (rd.claim_dir("c1") / "stdout.log").read_text().startswith("Traceback")
    # actual_config.json was written before the raise (for the report re-render)
    assert (rd.claim_dir("c1") / "actual_config.json").exists()
    # blocked result records the command that ACTUALLY ran (resolved), carried by
    # EvalFailed — not the unresolved spec command (report replays this verbatim)
    assert results[0].command.startswith("export PAPER_REPRISE_MODEL=")
    assert results[0].command.endswith("python eval.py")


def test_run_claims_records_resolved_command_on_success(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")

    def ok_eval(command, cwd, env_dir, log_path):
        Path(log_path).write_text("perplexity: 5.80")
        return 0, ""

    executor = make_run_executor(run_eval=ok_eval, detect_gpu=lambda: "A100",
                                 now=iter([0.0, 60.0]).__next__)
    results, _ = run_claims(rd, _spec_one("python eval.py"), executor=executor)
    # the model-resolved command, not the bare protocol command
    assert results[0].command.startswith("export PAPER_REPRISE_MODEL=")
    assert results[0].command.endswith("python eval.py")
    assert results[0].command != "python eval.py"


def test_successful_eval_is_ran_via_run_claims(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")

    def ok_eval(command, cwd, env_dir, log_path):
        Path(log_path).write_text("perplexity: 5.80")
        return 0, "perplexity: 5.80"

    executor = make_run_executor(run_eval=ok_eval, detect_gpu=lambda: "A100",
                                 now=iter([0.0, 30.0]).__next__)
    results, configs = run_claims(rd, _spec_one(), executor=executor)

    assert results[0].status == "ran"
    assert results[0].gpu == "A100"
    assert configs["c1"]["wbits"] == 4


def test_activated_env_uses_absolute_env_path(tmp_path, monkeypatch):
    # The eval command may `cd` (e.g. `bash impl/run_eval.sh` -> impl/), so a relative
    # env/bin on PATH would break python/entrypoint resolution. PATH must be absolute.
    import os
    from paper_reprise.runexec import _activated_env
    monkeypatch.chdir(tmp_path)
    env = _activated_env(Path("rundir/env"))            # relative input
    first = env["PATH"].split(os.pathsep)[0]
    assert os.path.isabs(first)
    assert first.endswith("rundir/env/bin")
    assert os.path.isabs(env["VIRTUAL_ENV"])
