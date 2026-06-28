"""Run directory layout and typed artifact I/O.

One RunDir == one paper reproduction run. All stage artifacts live under root.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Optional

import yaml

from paper_reprise.models import IngestInfo, PlanReport, Spec
from paper_reprise.modelpaths import run_models_dir


def _repo_output_subdir() -> str:
    """Subdir under repo/ where the paper's repo writes large artifacts (checkpoints).
    Default `runtime` (GSQ et al.); override with PAPER_REPRISE_REPO_OUTPUT_SUBDIR."""
    return os.environ.get("PAPER_REPRISE_REPO_OUTPUT_SUBDIR", "runtime")


# Exported model-weight file extensions cleaned up after a verified run (the
# quantized model is regenerable; records are kept).
_MODEL_WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx")


def _slug(text: str, max_len: int = 40) -> str:
    """Filesystem-safe slug from a paper title. Titles are usually "ShortName: subtitle",
    so use the part BEFORE the first colon (the authors' short name — e.g. "TurboQuant",
    "GSQ") when it yields a non-empty slug; otherwise fall back to the whole title.
    Truncated to max_len."""
    head = text.split(":", 1)[0]
    s = re.sub(r"[^a-z0-9]+", "-", head.lower()).strip("-")
    if not s:                       # colon-led / punctuation-only head -> use full title
        s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].strip("-")


def public_spec_dict(spec: Spec) -> dict:
    """Redacted view of a spec, safe to expose to the from-scratch implementer.

    Drops each claim's `expected`, `tolerance` and `source` — i.e. the paper's
    reported number, the pass band, and where it came from — so the agent cannot
    read the target it is graded against. Method, eval protocol and quant config
    are preserved (they are what must be implemented). Pure dict transform."""
    d = spec.model_dump()
    for claim in d.get("claims", []):
        for k in ("expected", "tolerance", "source"):
            claim.pop(k, None)
    return d


class RunDir:
    def __init__(self, root: Path):
        self.root = Path(root)

    # ---- lifecycle -------------------------------------------------------
    @classmethod
    def create(cls, base: Path, arxiv_id: str, timestamp: str,
               name: Optional[str] = None) -> "RunDir":
        slug = _slug(name) if name else ""
        stem = f"{slug}-{arxiv_id}-{timestamp}" if slug else f"{arxiv_id}-{timestamp}"
        root = Path(base) / stem
        rd = cls(root)
        for d in (rd.root, rd.paper_dir, rd.repo_dir, rd.runs_dir,
                  rd.setup_log_dir, rd.setup_patches_dir):
            d.mkdir(parents=True, exist_ok=True)
        return rd

    @classmethod
    def open(cls, root: Path) -> "RunDir":
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"run dir not found: {root}")
        return cls(root)

    # ---- subdirs ---------------------------------------------------------
    @property
    def paper_dir(self) -> Path:
        return self.root / "paper"

    @property
    def repo_dir(self) -> Path:
        return self.root / "repo"

    @property
    def runs_dir(self) -> Path:
        # Per-claim execution outputs live under `claims/` (one subdir per claim id) —
        # named so it doesn't read as a second `runs/` nested in the top-level runs/ tree.
        return self.root / "claims"

    @property
    def setup_log_dir(self) -> Path:
        return self.root / "setup_log"

    @property
    def setup_patches_dir(self) -> Path:
        return self.root / "setup_patches"

    def claim_dir(self, claim_id: str) -> Path:
        d = self.runs_dir / claim_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- typed artifact I/O ---------------------------------------------
    def write_ingest(self, info: IngestInfo) -> None:
        (self.root / "ingest.json").write_text(info.model_dump_json(indent=2))

    def read_ingest(self) -> Optional[IngestInfo]:
        p = self.root / "ingest.json"
        if not p.exists():
            return None
        return IngestInfo.model_validate_json(p.read_text())

    def write_spec(self, spec: Spec) -> None:
        (self.root / "spec.yaml").write_text(
            yaml.safe_dump(spec.model_dump(), sort_keys=False, allow_unicode=True)
        )

    def write_public_spec(self, spec: Spec) -> Path:
        """Write the redacted spec the from-scratch agent is allowed to read.

        The paper's expected values, tolerances and source citations are stripped
        from every claim (see public_spec_dict), so an honest implementation is
        built against the method + eval protocol only and never sees the number it
        is supposed to hit — the honesty barrier of the from-scratch path
        (design §6). The full spec.yaml stays the grade/human-review artifact."""
        p = self.root / "spec.public.yaml"
        p.write_text(
            yaml.safe_dump(public_spec_dict(spec), sort_keys=False, allow_unicode=True)
        )
        return p

    def read_spec(self) -> Optional[Spec]:
        p = self.root / "spec.yaml"
        if not p.exists():
            return None
        return Spec.model_validate(yaml.safe_load(p.read_text()))

    def write_plan(self, plan: PlanReport) -> None:
        (self.root / "plan.json").write_text(plan.model_dump_json(indent=2))

    def read_plan(self) -> Optional[PlanReport]:
        p = self.root / "plan.json"
        if not p.exists():
            return None
        return PlanReport.model_validate_json(p.read_text())

    def clean_model_artifacts(self, *, min_bytes: int = 10 * 1024 * 1024
                              ) -> list[tuple[str, int]]:
        """Delete exported model-weight files under the run dir — the quantized model
        the eval produced (e.g. a repo's `runtime/checkpoints/.../*.safetensors`) —
        while KEEPING every other record (logs, ingest/spec/plan json, env snapshot,
        setup_patches, per-claim stdout.log / actual_config, the reports).

        Weights are matched by extension AND a size floor, so configs/tokenizers and
        small fixtures are left alone; base models live in the shared cache (outside
        the run dir) and are never under here. Returns [(relpath, bytes)] removed.
        Best-effort: unreadable/locked files are skipped, then emptied dirs pruned."""
        removed: list[tuple[str, int]] = []
        for p in self.root.rglob("*"):
            if not p.is_file() or p.is_symlink():
                continue
            if p.suffix.lower() not in _MODEL_WEIGHT_EXTS:
                continue
            try:
                size = p.stat().st_size
                if size < min_bytes:
                    continue
                p.unlink()
            except OSError:
                continue
            removed.append((str(p.relative_to(self.root)), size))
        # prune now-empty directories left behind (deepest first), never the root
        for d in sorted((q for q in self.root.rglob("*") if q.is_dir()),
                        key=lambda q: len(q.parts), reverse=True):
            try:
                next(d.iterdir())
            except StopIteration:
                try:
                    d.rmdir()
                except OSError:
                    pass
            except OSError:
                pass
        return removed

    def link_repo_output_to_scratch(self) -> Optional[Path]:
        """Symlink `repo/<subdir>` (default `runtime`) → a per-run scratch dir, so the
        paper's repo writes its quantized model to scratch (big) instead of under
        runs/ (home, small quota), transparently — no command edits. paper-reprise's
        own records stay in the run root. Idempotent; SKIPS if `repo/<subdir>` already
        exists as a real dir (won't clobber data). Best-effort: returns the scratch
        target, or None if skipped/failed (e.g. scratch unwritable → repo falls back
        to its default home path)."""
        src = self.repo_dir / _repo_output_subdir()
        target = run_models_dir(self.root.name)
        try:
            if src.is_symlink():
                target.mkdir(parents=True, exist_ok=True)
                return target
            if src.exists():
                return None          # real dir already present — don't clobber
            target.mkdir(parents=True, exist_ok=True)
            src.parent.mkdir(parents=True, exist_ok=True)
            src.symlink_to(target, target_is_directory=True)
            return target
        except OSError:
            return None

    def clean_scratch_models(self) -> list[tuple[str, int]]:
        """Remove this run's scratch export dir (run_models_dir) and the dangling
        repo symlink, if present. Returns [(abspath, bytes)] removed."""
        removed: list[tuple[str, int]] = []
        target = run_models_dir(self.root.name)
        if target.exists() and not target.is_symlink():
            size = sum(p.stat().st_size for p in target.rglob("*")
                       if p.is_file() and not p.is_symlink())
            try:
                shutil.rmtree(target)
                removed.append((str(target), size))
            except OSError:
                pass
        link = self.repo_dir / _repo_output_subdir()
        if link.is_symlink():
            try:
                link.unlink()
            except OSError:
                pass
        return removed

    def clean_env(self) -> list[tuple[str, int]]:
        """Remove the per-run environment(s) — paper-reprise's `env/` and the cloned
        repo's `.venv` — while keeping `env_snapshot.json` and all records. The env
        is regenerable: a later `resume` rebuilds it in the setup stage. Returns
        [(reldir, bytes)] removed. (Bytes are the dir's own file sizes; actual freed
        space can be less when a uv venv hardlinks packages from the shared uv cache.)"""
        removed: list[tuple[str, int]] = []
        for d in (self.root / "env", self.repo_dir / ".venv"):
            if not d.is_dir():
                continue
            size = sum(p.stat().st_size for p in d.rglob("*")
                       if p.is_file() and not p.is_symlink())
            try:
                shutil.rmtree(d)
            except OSError:
                continue
            removed.append((str(d.relative_to(self.root)), size))
        return removed
