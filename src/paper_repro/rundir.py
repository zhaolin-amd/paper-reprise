"""Run directory layout and typed artifact I/O.

One RunDir == one paper reproduction run. All stage artifacts live under root.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from paper_repro.models import IngestInfo, PlanReport, Spec


class RunDir:
    def __init__(self, root: Path):
        self.root = Path(root)

    # ---- lifecycle -------------------------------------------------------
    @classmethod
    def create(cls, base: Path, arxiv_id: str, timestamp: str) -> "RunDir":
        root = Path(base) / f"{arxiv_id}-{timestamp}"
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
