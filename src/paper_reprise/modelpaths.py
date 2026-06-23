"""Model-path policy: read pretrained snapshots from a shared cache first, and
download anything missing to a writable scratch dir (never the shared cache).

Two layers, matching how quantization repos actually consume models:

1. **Read-first local snapshots** (`resolve_model`). The shared cache is laid out
   as `<MODEL_BASE>/<org>/<model>` snapshot dirs (config.json + weights), NOT the
   HF hub `models--org--name` layout — so HF can't auto-resolve a bare id against
   it. We instead map a model id (`meta-llama/Llama-3.2-1B`) to its local snapshot
   path when it exists, and hand that path to the eval command. This avoids
   re-downloading (and re-authing for gated models already cached there).

2. **Downloads go to scratch** (`hf_env_overlay`). For any model/dataset NOT in
   the shared cache, the repo falls back to a normal HF download; we point
   HF_HUB_CACHE at a writable scratch dir so those land there, not in $HOME (small
   quota) and not in the read-only shared cache.

Both roots default to this site's paths but are env-overridable so the tool stays
portable (see README "Model cache"). Pure logic + dict building — no subprocess.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path

# Shared, read-mostly pretrained-model cache (org/model snapshot layout). Site
# default; override with PAPER_REPRISE_MODEL_BASE.
_DEFAULT_MODEL_BASE = "/group/amdneuralopt/huggingface/pretrained_models"

# Placeholder an eval command may use for "the model" — substituted with the
# resolved path/id at run time (see resolved_command).
_MODEL_PLACEHOLDERS = ("{model}",)


def model_base() -> Path:
    """The shared pretrained-model cache root (env PAPER_REPRISE_MODEL_BASE)."""
    return Path(os.environ.get("PAPER_REPRISE_MODEL_BASE", _DEFAULT_MODEL_BASE))


def download_dir() -> Path:
    """The scratch dir missing models download into (env PAPER_REPRISE_DOWNLOAD_DIR,
    else /scratch/$USER/pretrained_models — derived, not tied to one account)."""
    env = os.environ.get("PAPER_REPRISE_DOWNLOAD_DIR")
    if env:
        return Path(env)
    user = os.environ.get("USER") or "shared"
    return Path(f"/scratch/{user}/pretrained_models")


def resolve_model(model_id: str) -> str:
    """Map a model id to its local snapshot path under the shared cache when present,
    else return the id unchanged so the repo downloads it via HF.

    Only a relative `org/model` id is resolved: an absolute path or a `..` segment
    is returned verbatim (a misextracted id must not read snapshots outside the
    shared cache). A local snapshot counts only if `<base>/<id>/config.json` exists
    — an empty or partial dir must NOT shadow a real download."""
    if not model_id:
        return model_id
    if model_id.startswith("/") or ".." in Path(model_id).parts:
        return model_id
    candidate = model_base() / model_id
    if (candidate / "config.json").is_file():
        return str(candidate)
    return model_id


def resolved_command(command: str, model_id: str) -> str:
    """Make a command model-aware: substitute `{model}` with the resolved path AND
    `export PAPER_REPRISE_MODEL=<resolved>` so the WHOLE command sees it.

    `export …; cmd` (not the `VAR=val cmd` prefix form) is deliberate: an inline
    assignment only scopes the first simple command, so a compound eval command
    (`cd repo && python eval.py`, a pipeline, …) or spec extra_args like
    `GSQ_MODEL_NAME=$PAPER_REPRISE_MODEL` would otherwise not see it. The resolved
    value is shlex-quoted in both the export and the `{model}` substitution since
    the command runs under shell=True. Empty model_id → command unchanged (a
    misextracted spec then fails loudly on the literal rather than on `''`)."""
    if not model_id:
        return command
    quoted = shlex.quote(resolve_model(model_id))
    for ph in _MODEL_PLACEHOLDERS:
        command = command.replace(ph, quoted)
    return f"export PAPER_REPRISE_MODEL={quoted}; {command}"


def with_tasks(command: str, tasks: str | None) -> str:
    """Prepend `export PAPER_REPRISE_TASKS=<tasks>` so an eval command can pick up a
    user-overridden lm-eval task list (e.g. `run --tasks arc_easy,piqa`). Eval
    commands reference it as `${PAPER_REPRISE_TASKS:-<spec default tasks>}`, so the
    override wins when set and the spec's own tasks apply otherwise. Same
    export-not-prefix rationale as resolved_command (compound commands must see it).
    No-op when tasks is empty/None."""
    if not tasks:
        return command
    return f"export PAPER_REPRISE_TASKS={shlex.quote(tasks)}; {command}"


def hf_env_overlay() -> dict:
    """Env vars to merge into an eval/smoke subprocess so HF model downloads land in
    the scratch dir (created here), not $HOME (small quota) or the read-only shared
    cache. HF_HOME only gets a default if unset — an existing scratch HF_HOME is
    respected. Never points HF caches at the shared (read-only) cache."""
    dl = download_dir()
    dl.mkdir(parents=True, exist_ok=True)
    overlay: dict = {"HF_HUB_CACHE": str(dl)}
    if not os.environ.get("HF_HOME"):
        overlay["HF_HOME"] = str(dl.parent / "cache" / "huggingface")
    return overlay
