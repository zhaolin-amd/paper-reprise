"""Typed contracts shared across all pipeline stages.

This module depends on nothing else in paper_reprise — it is the pure schema.
"""
from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

Runner = Literal["official", "cited-standard", "custom"]
CalibStatus = Literal["known", "UNKNOWN"]
Verdict = Literal["MATCH", "PARTIAL", "FAIL", "BLOCKED"]

# specextract (LLM) often phrases calib_status as the calibration situation rather than
# the required enum. Map the common variants so one mislabel doesn't abort the whole run;
# genuinely unrecognized values still fall through and are rejected (loud failure).
# `known` == the calibration config is determinable (incl. data-free / no calibration);
# `UNKNOWN` == it cannot be determined → grade BLOCKS the claim.
_CALIB_KNOWN = {
    "known", "calibrated", "uncalibrated", "calibration-free", "calibration free",
    "data-free", "data free", "datafree", "static", "rtn", "round-to-nearest",
    "none", "n/a", "na",
}
_CALIB_UNKNOWN = {
    "unknown", "unspecified", "unclear", "undetermined", "tbd",
    "not specified", "not stated", "?",
}


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

    @field_validator("few_shot", mode="before")
    @classmethod
    def _few_shot_default(cls, v):
        # specextract (LLM) sometimes emits `few_shot: null`; treat as the 0 default.
        return 0 if v is None else v

    @field_validator("extra_args", mode="before")
    @classmethod
    def _extra_args_to_str(cls, v):
        # extra_args is a free-form string, but specextract sometimes emits a dict
        # (e.g. {"temperature": 1.0, "num_repeats": 10}); JSON-encode it rather than
        # aborting the whole run on a type mismatch.
        if v is None or isinstance(v, str):
            return v
        try:
            return json.dumps(v, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(v)


class Artifact(BaseModel):
    id: str
    base_model: str
    method: str
    quant_config: dict
    calib_status: CalibStatus = "known"

    @field_validator("calib_status", mode="before")
    @classmethod
    def _normalize_calib_status(cls, v):
        if isinstance(v, str):
            u = v.strip().lower()
            if u in _CALIB_KNOWN:
                return "known"
            if u in _CALIB_UNKNOWN:
                return "UNKNOWN"
        return v  # unrecognized -> falls through to the Literal (rejected, loud failure)

    @model_validator(mode="after")
    def _alias_bits_to_wbits(self) -> "Artifact":
        qc = self.quant_config
        if isinstance(qc, dict) and "wbits" not in qc and "bits" in qc:
            qc["wbits"] = qc["bits"]
        return self


class Claim(BaseModel):
    id: str
    artifact: str                    # references Artifact.id
    eval_protocol: EvalProtocol
    expected: float
    tolerance: float
    source: str                      # e.g. "Table 3, row 2, col W4"
    hardware: Optional[str] = None   # null for accuracy claims; pinned for efficiency claims
    no_paper_ref: bool = False       # True → paper column shows "-", no diff annotation


class RepoInfo(BaseModel):
    url: str
    commit: Optional[str] = None


class ReferenceRepo(BaseModel):
    """A prerequisite method THIS paper's algorithm builds on, plus its official repo.

    Surfaced to the from-scratch implementer as a READ-ONLY reference to disambiguate
    details the current paper leaves underspecified — the paper stays the source of truth,
    and these never carry the target numbers (so they survive into the redacted spec)."""
    method: str                      # e.g. "QJL"
    repo_url: str
    note: Optional[str] = None       # what to look at / caveat


class Spec(BaseModel):
    paper: str
    repo: Optional[RepoInfo] = None
    artifacts: list[Artifact]
    claims: list[Claim]
    references: list[ReferenceRepo] = []

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
    reason: str                      # canonical English (shown in README.md)
    reason_zh: Optional[str] = None  # Chinese rendering (README_zh.md); falls back to reason
    checks: dict                     # {"value": bool, "faithful": bool}
