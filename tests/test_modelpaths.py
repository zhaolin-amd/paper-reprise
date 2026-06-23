"""Model-path policy: read shared cache first, download missing to scratch."""
from __future__ import annotations

from pathlib import Path

from paper_reprise import modelpaths


def _make_snapshot(base, model_id):
    """Create a fake local snapshot `base/<model_id>/config.json`."""
    d = base / model_id
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    return d


# --- defaults & overrides ---------------------------------------------------

def test_defaults_when_env_unset(monkeypatch):
    # the /group-read /scratch-download contract depends on these literals
    monkeypatch.delenv("PAPER_REPRISE_MODEL_BASE", raising=False)
    monkeypatch.delenv("PAPER_REPRISE_DOWNLOAD_DIR", raising=False)
    monkeypatch.setenv("USER", "alice")
    assert modelpaths.model_base() == Path("/group/amdneuralopt/huggingface/pretrained_models")
    assert modelpaths.download_dir() == Path("/scratch/alice/pretrained_models")


def test_env_overrides_win(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPER_REPRISE_MODEL_BASE", str(tmp_path / "base"))
    monkeypatch.setenv("PAPER_REPRISE_DOWNLOAD_DIR", str(tmp_path / "dl"))
    assert modelpaths.model_base() == tmp_path / "base"
    assert modelpaths.download_dir() == tmp_path / "dl"


# --- resolve_model ----------------------------------------------------------

def test_resolve_model_hits_local_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_REPRISE_MODEL_BASE", str(tmp_path))
    snap = _make_snapshot(tmp_path, "meta-llama/Llama-3.2-1B")
    assert modelpaths.resolve_model("meta-llama/Llama-3.2-1B") == str(snap)


def test_resolve_model_miss_returns_id(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_REPRISE_MODEL_BASE", str(tmp_path))
    assert modelpaths.resolve_model("meta-llama/Llama-3.2-1B") == "meta-llama/Llama-3.2-1B"


def test_resolve_model_partial_dir_does_not_shadow(tmp_path, monkeypatch):
    # a dir with no config.json must NOT shadow a real download
    monkeypatch.setenv("PAPER_REPRISE_MODEL_BASE", str(tmp_path))
    (tmp_path / "org" / "m").mkdir(parents=True)
    assert modelpaths.resolve_model("org/m") == "org/m"


def test_resolve_model_rejects_traversal_and_absolute(tmp_path, monkeypatch):
    # a misextracted id must not read snapshots outside the shared cache
    monkeypatch.setenv("PAPER_REPRISE_MODEL_BASE", str(tmp_path))
    assert modelpaths.resolve_model("../../etc") == "../../etc"
    assert modelpaths.resolve_model("/etc/passwd") == "/etc/passwd"


# --- resolved_command -------------------------------------------------------

def test_resolved_command_substitutes_and_exports(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_REPRISE_MODEL_BASE", str(tmp_path))
    snap = _make_snapshot(tmp_path, "org/m")
    out = modelpaths.resolved_command("python eval.py --model {model}", "org/m")
    assert out == f"export PAPER_REPRISE_MODEL={snap}; python eval.py --model {snap}"


def test_resolved_command_exports_for_whole_compound_command(tmp_path, monkeypatch):
    # the export form (not a `VAR=val cmd` prefix) scopes the var across && / pipes,
    # so the segment that consumes $PAPER_REPRISE_MODEL actually sees it
    monkeypatch.setenv("PAPER_REPRISE_MODEL_BASE", str(tmp_path))
    out = modelpaths.resolved_command(
        "cd repo && GSQ_MODEL_NAME=$PAPER_REPRISE_MODEL python eval.py", "org/m")
    assert out.startswith("export PAPER_REPRISE_MODEL=org/m; ")
    assert out.endswith("cd repo && GSQ_MODEL_NAME=$PAPER_REPRISE_MODEL python eval.py")


def test_resolved_command_quotes_shell_metacharacters(tmp_path, monkeypatch):
    # a missing model id with shell metachars must be quoted, not interpreted
    monkeypatch.setenv("PAPER_REPRISE_MODEL_BASE", str(tmp_path))
    out = modelpaths.resolved_command("python eval.py --model {model}", "m; rm -rf x")
    assert "--model 'm; rm -rf x'" in out      # quoted as one arg, not injected
    assert "--model m; rm -rf x" not in out    # the raw (injectable) form is absent


def test_resolved_command_empty_model_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPER_REPRISE_MODEL_BASE", str(tmp_path))
    assert modelpaths.resolved_command("python main.py --config c.yaml", "") == \
        "python main.py --config c.yaml"


# --- hf_env_overlay ---------------------------------------------------------

def test_hf_env_overlay_points_hub_cache_at_scratch(tmp_path, monkeypatch):
    dl = tmp_path / "scratch" / "pretrained_models"
    monkeypatch.setenv("PAPER_REPRISE_DOWNLOAD_DIR", str(dl))
    monkeypatch.setenv("HF_HOME", "/preset/hf")  # respected, not overridden
    overlay = modelpaths.hf_env_overlay()
    assert overlay["HF_HUB_CACHE"] == str(dl)
    assert dl.is_dir()                       # created
    assert "HF_HOME" not in overlay          # existing HF_HOME respected
    assert overlay["HF_HUB_CACHE"] != str(modelpaths.model_base())  # never the shared cache


def test_hf_env_overlay_defaults_hf_home_when_unset(tmp_path, monkeypatch):
    dl = tmp_path / "scratch" / "pretrained_models"
    monkeypatch.setenv("PAPER_REPRISE_DOWNLOAD_DIR", str(dl))
    monkeypatch.delenv("HF_HOME", raising=False)
    overlay = modelpaths.hf_env_overlay()
    assert overlay["HF_HOME"].endswith("cache/huggingface")


# --- seam wiring (the /scratch-download contract at the real env builder) ----

def test_activated_env_injects_hub_cache(tmp_path, monkeypatch):
    from paper_reprise.runexec import _activated_env
    dl = tmp_path / "dl"
    monkeypatch.setenv("PAPER_REPRISE_DOWNLOAD_DIR", str(dl))
    env_dir = tmp_path / "env"
    env = _activated_env(env_dir)
    assert env["HF_HUB_CACHE"] == str(dl)            # download contract wired in
    assert env["VIRTUAL_ENV"] == str(env_dir)        # venv still activated
    assert str(env_dir / "bin") in env["PATH"]


def test_with_tasks_prepends_export_or_noop():
    from paper_reprise.modelpaths import with_tasks
    assert with_tasks("bash run.sh", "arc_easy,piqa") == \
        "export PAPER_REPRISE_TASKS=arc_easy,piqa; bash run.sh"
    assert with_tasks("bash run.sh", None) == "bash run.sh"   # no override
    assert with_tasks("bash run.sh", "") == "bash run.sh"     # empty → no-op
