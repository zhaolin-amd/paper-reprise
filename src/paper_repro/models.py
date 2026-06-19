"""Typed contracts shared across all pipeline stages.

This module depends on nothing else in paper_repro — it is the pure schema.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, model_validator

Runner = Literal["official", "cited-standard", "custom"]
CalibStatus = Literal["known", "UNKNOWN"]
Verdict = Literal["MATCH", "PARTIAL", "FAIL", "BLOCKED"]


class EvalProtocol(BaseModel):
    runner: Runner
    command: str
    metric: str                      # "perplexity" | "accuracy" | "speedup" | ...
    dataset: str
    split: Optional[str] = None
    seqlen: Optional[int] = None
    stride: Optional[int] = None
    few_shot: int = 0
    extra_args: Optional[str] = None


class Artifact(BaseModel):
    id: str
    base_model: str
    method: str
    quant_config: dict
    calib_status: CalibStatus = "known"


class Claim(BaseModel):
    id: str
    artifact: str                    # references Artifact.id
    eval_protocol: EvalProtocol
    expected: float
    tolerance: float
    source: str                      # e.g. "Table 3, row 2, col W4"
    hardware: Optional[str] = None   # null for accuracy claims; pinned for efficiency claims


class RepoInfo(BaseModel):
    url: str
    commit: Optional[str] = None


class Spec(BaseModel):
    paper: str
    repo: Optional[RepoInfo] = None
    artifacts: list[Artifact]
    claims: list[Claim]

    @model_validator(mode="after")
    def _claims_reference_known_artifacts(self) -> "Spec":
        ids = {a.id for a in self.artifacts}
        for c in self.claims:
            if c.artifact not in ids:
                raise ValueError(f"claim {c.id} references unknown artifact {c.artifact!r}")
        return self


class IngestInfo(BaseModel):
    arxiv_id: str
    title: Optional[str] = None
    authors: list[str] = []
    source_url: str
    repo: Optional[RepoInfo] = None
    latex_path: Optional[str] = None
    repo_path: Optional[str] = None


class ClaimPlan(BaseModel):
    claim_id: str
    est_gpus: int = 1
    est_vram_gb: Optional[float] = None
    est_minutes: Optional[float] = None
    required_hardware: Optional[str] = None
    feasible: bool = True
    anomaly: Optional[str] = None    # set when estimate wildly diverges from paper


class PlanReport(BaseModel):
    claims: list[ClaimPlan]
    needs_user_decision: bool = False
    decision_reason: Optional[str] = None


class RunResult(BaseModel):
    claim_id: str
    command: str
    seed: Optional[int] = None
    gpu: Optional[str] = None
    minutes: Optional[float] = None
    stdout_path: str                 # path to raw output file
    status: Literal["ran", "blocked"] = "ran"
    block_reason: Optional[str] = None


class ClaimGrade(BaseModel):
    claim_id: str
    verdict: Verdict
    measured: Optional[float]
    expected: float
    reason: str
    checks: dict                     # {"value": bool, "faithful": bool}
