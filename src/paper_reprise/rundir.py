"""Run directory layout and typed artifact I/O.

One RunDir == one paper reproduction run. All stage artifacts live under root.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml

from paper_reprise.models import IngestInfo, PlanReport, Spec


def _slug(text: str, max_len: int = 40) -> str:
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
        return self.root / "runs"

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
