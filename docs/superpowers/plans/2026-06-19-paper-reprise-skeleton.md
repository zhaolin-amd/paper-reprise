# paper-reprise Deterministic Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the deterministic pipeline skeleton for paper-reprise — able to run `ingest → specextract → plan → setup → run → grade → report` end to end, with grade, report, and the data contracts (models) fully TDD-implemented and locked; specextract/setup/run wired through their interfaces with mock claude and fixtures, leaving real GPU quantization/eval to a later Plan 2.

**Architecture:** Deterministic Python orchestration; each stage reads/writes typed artifacts in the run directory (Approach B). The agent's nondeterminism is confined to the setup stage (interface + stub this phase). grade is pure code, isolated from execution — reads only the persisted raw output + spec — implementing the "process-faithful AND value-in-tolerance" double check and the MATCH/PARTIAL/FAIL/BLOCKED four verdicts.

**Tech Stack:** Python 3.12, pydantic v2 (data models), click (CLI), pyyaml (spec), pytest (tests), uv (deps/venv), `claude -p` headless (specextract/setup, reusing llm-paper-radar's invocation pattern).

Design doc: `docs/superpowers/specs/2026-06-19-paper-reprise-agent-design.md`

---

## File Structure

```
paper-reprise/
  pyproject.toml                  # project metadata + deps + pytest/ruff config
  src/paper_reprise/
    __init__.py
    models.py                     # pydantic models: EvalProtocol/Artifact/Claim/Spec/IngestInfo/PlanReport/ClaimGrade/RunResult
    rundir.py                     # RunDir: directory layout + artifact I/O
    parsers.py                    # metric output parsers: PPL / accuracy / speedup
    grade.py                      # pure-code judge: value + faithfulness double check → 4 verdicts
    report.py                     # render report.zh.md / report.en.md
    ingest.py                     # input normalization (arxiv url/id) + latex fetch + repo discovery
    planstage.py                  # feasibility/anomaly sentinel
    headless.py                   # claude -p wrapper, output-file verification (don't trust exit code)
    specextract.py                # specextract stage: call headless → spec.yaml → gate
    setupstage.py                 # setup stage: agentic debug loop (interface + stub)
    runstage.py                   # run stage: quant + eval (interface + stub)
    pipeline.py                   # orchestrate 7 stages + gates
    cli.py                        # click CLI: run / resume / report
  tests/
    conftest.py                   # shared fixtures
    fixtures/                     # spec.yaml / eval output
    test_models.py
    test_rundir.py
    test_parsers.py
    test_grade.py
    test_report.py
    test_ingest.py
    test_planstage.py
    test_headless.py
    test_specextract.py
    test_pipeline.py
    test_cli.py
```

**Responsibility boundaries:** each file has one clear responsibility. `grade.py` never imports `runstage.py` (judge/execution isolation; they communicate only through files persisted by RunDir). `models.py` depends on no other module in this package (pure contract).

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/paper_reprise/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "paper-reprise"
version = "0.1.0"
description = "Reproduce quantization paper results from arxiv / official repos"
requires-python = ">=3.11"
dependencies = [
    "click>=8.1.7",
    "pydantic>=2.7.0",
    "pyyaml>=6.0.1",
    "httpx>=0.27.0",
]

[dependency-groups]
dev = [
    "pytest>=8.3.0",
    "ruff>=0.7.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/paper_reprise"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 2: Create package entry**

`src/paper_reprise/__init__.py`:
```python
"""paper-reprise: reproduce quantization paper results."""

__version__ = "0.1.0"
```

`tests/__init__.py`:
```python
```

- [ ] **Step 3: Sync deps, create venv**

Run: `cd /proj/xcohdstaff7/zhaolin/code/paper-reprise && uv sync`
Expected: creates `.venv/`, installs click/pydantic/pyyaml/httpx/pytest/ruff, no errors.

- [ ] **Step 4: Verify import**

Run: `uv run python -c "import paper_reprise; print(paper_reprise.__version__)"`
Expected: prints `0.1.0`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/paper_reprise/__init__.py tests/__init__.py uv.lock
git commit -m "chore: project scaffold with uv + pytest"
```

---

## Task 2: Data Models (contracts)

**Files:**
- Create: `src/paper_reprise/models.py`
- Test: `tests/test_models.py`

These pydantic models are the contract shared across all stages. Once the field names are set, later tasks must follow them strictly.

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:
```python
import pytest
from pydantic import ValidationError

from paper_reprise.models import (
    EvalProtocol, Artifact, Claim, Spec, IngestInfo, RepoInfo,
    PlanReport, ClaimPlan, ClaimGrade, RunResult, Verdict, Runner, CalibStatus,
)


def test_eval_protocol_minimal():
    ep = EvalProtocol(
        runner="official",
        command="python eval.py --model {model}",
        metric="perplexity",
        dataset="wikitext2",
    )
    assert ep.runner == "official"
    assert ep.split is None
    assert ep.few_shot == 0


def test_runner_must_be_enum():
    with pytest.raises(ValidationError):
        EvalProtocol(runner="bogus", command="x", metric="ppl", dataset="d")


def test_artifact_calib_status_default_known():
    a = Artifact(id="a1", base_model="m", method="AWQ", quant_config={"wbits": 4})
    assert a.calib_status == "known"


def test_claim_carries_expected_and_tolerance():
    ep = EvalProtocol(runner="official", command="c", metric="perplexity", dataset="wikitext2")
    c = Claim(id="c1", artifact="a1", eval_protocol=ep, expected=5.78, tolerance=0.05,
              source="Table 3")
    assert c.expected == 5.78
    assert c.hardware is None


def test_spec_roundtrip_via_dict():
    spec = Spec(
        paper="2401.00001",
        repo=RepoInfo(url="https://github.com/x/y", commit="abc"),
        artifacts=[Artifact(id="a1", base_model="m", method="AWQ", quant_config={"wbits": 4})],
        claims=[Claim(
            id="c1", artifact="a1",
            eval_protocol=EvalProtocol(runner="official", command="c",
                                       metric="perplexity", dataset="wikitext2"),
            expected=5.78, tolerance=0.05, source="Table 3")],
    )
    d = spec.model_dump()
    spec2 = Spec.model_validate(d)
    assert spec2.claims[0].artifact == "a1"


def test_claim_artifact_must_reference_existing_artifact():
    with pytest.raises(ValueError, match="unknown artifact"):
        Spec(
            paper="2401.00001",
            repo=None,
            artifacts=[Artifact(id="a1", base_model="m", method="AWQ", quant_config={})],
            claims=[Claim(
                id="c1", artifact="MISSING",
                eval_protocol=EvalProtocol(runner="official", command="c",
                                           metric="perplexity", dataset="wikitext2"),
                expected=1.0, tolerance=0.05, source="T")],
        )


def test_verdict_enum_values():
    assert set(Verdict.__args__) if hasattr(Verdict, "__args__") else True
    cg = ClaimGrade(claim_id="c1", verdict="MATCH", measured=5.80, expected=5.78,
                    reason="", checks={"value": True, "faithful": True})
    assert cg.verdict == "MATCH"
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.models'`

- [ ] **Step 3: Write the implementation**

`src/paper_reprise/models.py`:
```python
"""Typed contracts shared across all pipeline stages.

This module depends on nothing else in paper_reprise — it is the pure schema.
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
    stdout_path: str                 # path to raw persisted output
    status: Literal["ran", "blocked"] = "ran"
    block_reason: Optional[str] = None


class ClaimGrade(BaseModel):
    claim_id: str
    verdict: Verdict
    measured: Optional[float]
    expected: float
    reason: str
    checks: dict                     # {"value": bool, "faithful": bool}
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/models.py tests/test_models.py
git commit -m "feat: typed contracts (Spec/Claim/Grade/...) with cross-ref validation"
```

---

## Task 3: RunDir Layout and I/O

**Files:**
- Create: `src/paper_reprise/rundir.py`
- Test: `tests/test_rundir.py`

- [ ] **Step 1: Write the failing test**

`tests/test_rundir.py`:
```python
from pathlib import Path

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
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_rundir.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.rundir'`

- [ ] **Step 3: Write the implementation**

`src/paper_reprise/rundir.py`:
```python
"""Run directory layout and typed artifact I/O.

One RunDir == one paper reproduction run. All stage artifacts live under root.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from paper_reprise.models import IngestInfo, PlanReport, Spec


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
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_rundir.py -v`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/rundir.py tests/test_rundir.py
git commit -m "feat: RunDir layout + typed artifact I/O"
```

---

## Task 4: Metric Parsers

**Files:**
- Create: `src/paper_reprise/parsers.py`
- Test: `tests/test_parsers.py`

The run stage persists the eval script's raw stdout; the grade stage uses these parsers to extract the numbers. If it can't parse, return None explicitly (so grade marks it BLOCKED/UNPARSEABLE) — never guess.

- [ ] **Step 1: Write the failing test**

`tests/test_parsers.py`:
```python
from paper_reprise.parsers import parse_metric


def test_parse_perplexity_simple():
    out = "Evaluating...\nwikitext2 perplexity: 5.80\nDone."
    assert parse_metric("perplexity", out) == 5.80


def test_parse_perplexity_ppl_alias():
    out = "final PPL = 7.41"
    assert parse_metric("perplexity", out) == 7.41


def test_parse_accuracy_percent():
    out = "hellaswag acc: 76.3%"
    assert parse_metric("accuracy", out) == 76.3


def test_parse_accuracy_fraction_normalized_to_percent():
    out = "acc,none: 0.763"
    assert parse_metric("accuracy", out) == 76.3


def test_parse_speedup():
    out = "Throughput speedup: 2.1x over fp16"
    assert parse_metric("speedup", out) == 2.1


def test_unparseable_returns_none():
    assert parse_metric("perplexity", "garbage with no number relevant") is None


def test_unknown_metric_returns_none():
    assert parse_metric("bleu", "bleu: 30") is None
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_parsers.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.parsers'`

- [ ] **Step 3: Write the implementation**

`src/paper_reprise/parsers.py`:
```python
"""Metric extraction from raw eval-script stdout.

Returns None when nothing reliable can be extracted — never guess.
Accuracy is always normalized to a percentage in [0, 100].
"""
from __future__ import annotations

import re
from typing import Optional

_PPL_PATTERNS = [
    r"perplexity[:\s=]+([0-9]+\.?[0-9]*)",
    r"\bppl\b[:\s=]+([0-9]+\.?[0-9]*)",
]
_ACC_PATTERNS = [
    r"acc[a-z,_ ]*[:\s=]+([0-9]+\.?[0-9]*)\s*%",   # "acc: 76.3%"
    r"acc[a-z,_ ]*[:\s=]+([0-9]*\.?[0-9]+)",        # "acc,none: 0.763"
]
_SPEEDUP_PATTERNS = [
    r"speedup[:\s=]+([0-9]+\.?[0-9]*)\s*x?",
    r"([0-9]+\.?[0-9]*)\s*x\b",
]


def _first_match(patterns: list[str], text: str) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def parse_metric(metric: str, text: str) -> Optional[float]:
    metric = metric.lower()
    if metric in ("perplexity", "ppl"):
        return _first_match(_PPL_PATTERNS, text)
    if metric in ("accuracy", "acc"):
        val = _first_match(_ACC_PATTERNS, text)
        if val is None:
            return None
        return val * 100 if val <= 1.0 else val
    if metric == "speedup":
        return _first_match(_SPEEDUP_PATTERNS, text)
    return None
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_parsers.py -v`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/parsers.py tests/test_parsers.py
git commit -m "feat: metric parsers (ppl/accuracy/speedup) returning None on miss"
```

---

## Task 5: Grade — Pure-Code Judge (core)

**Files:**
- Create: `src/paper_reprise/grade.py`
- Test: `tests/test_grade.py`

This is the crown of the whole system. Two independent checks (value-in-tolerance + process-faithfulness), four verdicts. grade reads only spec + the run's persisted output; **it does not import runstage, does not re-run, and knows no execution context beyond the target value**.

Verdict rules (consistent with design §5.1):
- **MATCH** = value in tolerance AND process faithful
- **PARTIAL** = value in tolerance but process diverged; or process faithful but value out of tolerance (reason required)
- **FAIL** = value significantly off and unattributable (out of tolerance AND process also diverged)
- **BLOCKED** = run didn't run / output unparseable / calib UNKNOWN → incomparable

- [ ] **Step 1: Write the failing test**

`tests/test_grade.py`:
```python
from paper_reprise.models import (
    Artifact, Claim, EvalProtocol, Spec, RunResult, RepoInfo,
)
from paper_reprise.grade import grade_claim


def _claim(seqlen=2048, calib_status="known", expected=5.78, tol=0.05):
    return Claim(
        id="c1", artifact="a1",
        eval_protocol=EvalProtocol(runner="official", command="c",
                                   metric="perplexity", dataset="wikitext2",
                                   seqlen=seqlen),
        expected=expected, tolerance=tol, source="T",
    )


def _artifact(calib_status="known"):
    return Artifact(id="a1", base_model="m", method="AWQ",
                    quant_config={"wbits": 4, "seqlen": 2048}, calib_status=calib_status)


def _run(stdout_path, status="ran"):
    return RunResult(claim_id="c1", command="c", stdout_path=str(stdout_path),
                     status=status)


def test_match_when_value_in_tol_and_faithful(tmp_path):
    out = tmp_path / "c1.log"; out.write_text("perplexity: 5.80")
    g = grade_claim(_claim(), _artifact(), _run(out), actual_config={"seqlen": 2048})
    assert g.verdict == "MATCH"
    assert g.measured == 5.80


def test_partial_when_value_off_but_faithful(tmp_path):
    out = tmp_path / "c1.log"; out.write_text("perplexity: 6.50")
    g = grade_claim(_claim(), _artifact(), _run(out), actual_config={"seqlen": 2048})
    assert g.verdict == "PARTIAL"
    assert "超容差" in g.reason or "tolerance" in g.reason.lower()


def test_partial_when_value_ok_but_config_diverged(tmp_path):
    out = tmp_path / "c1.log"; out.write_text("perplexity: 5.80")
    g = grade_claim(_claim(seqlen=2048), _artifact(), _run(out),
                    actual_config={"seqlen": 4096})
    assert g.verdict == "PARTIAL"
    assert "seqlen" in g.reason


def test_fail_when_value_off_and_config_diverged(tmp_path):
    out = tmp_path / "c1.log"; out.write_text("perplexity: 9.99")
    g = grade_claim(_claim(seqlen=2048), _artifact(), _run(out),
                    actual_config={"seqlen": 4096})
    assert g.verdict == "FAIL"


def test_blocked_when_run_blocked(tmp_path):
    out = tmp_path / "missing.log"
    g = grade_claim(_claim(), _artifact(), _run(out, status="blocked"),
                    actual_config={"seqlen": 2048})
    assert g.verdict == "BLOCKED"


def test_blocked_when_unparseable(tmp_path):
    out = tmp_path / "c1.log"; out.write_text("no number here")
    g = grade_claim(_claim(), _artifact(), _run(out), actual_config={"seqlen": 2048})
    assert g.verdict == "BLOCKED"
    assert "解析" in g.reason or "parse" in g.reason.lower()


def test_blocked_when_calib_unknown(tmp_path):
    out = tmp_path / "c1.log"; out.write_text("perplexity: 5.80")
    g = grade_claim(_claim(), _artifact(calib_status="UNKNOWN"), _run(out),
                    actual_config={"seqlen": 2048})
    assert g.verdict == "BLOCKED"
    assert "calib" in g.reason.lower()
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_grade.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.grade'`

- [ ] **Step 3: Write the implementation**

`src/paper_reprise/grade.py`:
```python
"""Pure-code judge. Isolated from execution: reads only spec + run's persisted output.

Two independent checks:
  1. value:    |measured - expected| <= tolerance
  2. faithful: actual run config matches the claim's eval_protocol / artifact

Verdict matrix (design §5.1):
  MATCH   = value AND faithful
  PARTIAL = (value AND not faithful) OR (faithful AND not value)  [reason required]
  FAIL    = not value AND not faithful
  BLOCKED = run blocked / unparseable / calib UNKNOWN (not "failed to reproduce")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from paper_reprise.models import Artifact, Claim, ClaimGrade, RunResult
from paper_reprise.parsers import parse_metric

# config keys whose divergence breaks faithfulness
_FAITHFUL_KEYS = ("seqlen", "stride", "wbits", "group_size", "few_shot")


def _faithfulness(claim: Claim, actual_config: dict) -> tuple[bool, list[str]]:
    expected_cfg = {}
    ep = claim.eval_protocol
    if ep.seqlen is not None:
        expected_cfg["seqlen"] = ep.seqlen
    if ep.stride is not None:
        expected_cfg["stride"] = ep.stride
    if ep.few_shot is not None:
        expected_cfg["few_shot"] = ep.few_shot

    diffs = []
    for k in _FAITHFUL_KEYS:
        if k in expected_cfg and k in actual_config:
            if expected_cfg[k] != actual_config[k]:
                diffs.append(f"{k} 不一致 (spec={expected_cfg[k]} actual={actual_config[k]})")
    return (len(diffs) == 0, diffs)


def grade_claim(claim: Claim, artifact: Artifact, run: RunResult,
                actual_config: dict) -> ClaimGrade:
    # --- BLOCKED short-circuits ---
    if run.status == "blocked":
        return ClaimGrade(claim_id=claim.id, verdict="BLOCKED", measured=None,
                          expected=claim.expected,
                          reason=f"run 未跑成: {run.block_reason or 'unknown'}",
                          checks={"value": False, "faithful": False})

    if artifact.calib_status == "UNKNOWN":
        return ClaimGrade(claim_id=claim.id, verdict="BLOCKED", measured=None,
                          expected=claim.expected,
                          reason="calib 配置缺失 (calib_status=UNKNOWN),结果不可比",
                          checks={"value": False, "faithful": False})

    text = ""
    p = Path(run.stdout_path)
    if p.exists():
        text = p.read_text()
    measured = parse_metric(claim.eval_protocol.metric, text)
    if measured is None:
        return ClaimGrade(claim_id=claim.id, verdict="BLOCKED", measured=None,
                          expected=claim.expected,
                          reason=f"无法从输出解析 {claim.eval_protocol.metric}",
                          checks={"value": False, "faithful": False})

    # --- two checks ---
    value_ok = abs(measured - claim.expected) <= claim.tolerance
    faithful_ok, diffs = _faithfulness(claim, actual_config)

    if value_ok and faithful_ok:
        verdict, reason = "MATCH", "—"
    elif value_ok and not faithful_ok:
        verdict, reason = "PARTIAL", "数值达标但过程有偏差: " + "; ".join(diffs)
    elif faithful_ok and not value_ok:
        delta = abs(measured - claim.expected)
        verdict, reason = "PARTIAL", f"过程忠实但数值超容差 {delta:.4g} (>{claim.tolerance})"
    else:
        verdict, reason = "FAIL", "数值超容差且过程有偏差: " + "; ".join(diffs)

    return ClaimGrade(claim_id=claim.id, verdict=verdict, measured=measured,
                      expected=claim.expected, reason=reason,
                      checks={"value": value_ok, "faithful": faithful_ok})
```

> Note: during code review the faithfulness check was strengthened to also pull
> `wbits`/`group_size` from `artifact.quant_config` (the plan version above omitted
> them), closing a false-MATCH hole. See the shipped `grade.py` for the final form.

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_grade.py -v`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/grade.py tests/test_grade.py
git commit -m "feat: pure-code judge with value+faithfulness double check, 4 verdicts"
```

---

## Task 6: Report — Bilingual Rendering

**Files:**
- Create: `src/paper_reprise/report.py`
- Test: `tests/test_report.py`

Render `report.zh.md` and `report.en.md`. Always use the measured raw numbers, never fill in paper numbers; each claim carries its replay info.

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:
```python
from paper_reprise.models import (
    Spec, Artifact, Claim, EvalProtocol, RepoInfo, ClaimGrade, RunResult, IngestInfo,
)
from paper_reprise.report import render_reports


def _ctx():
    spec = Spec(
        paper="2401.00001",
        repo=RepoInfo(url="https://github.com/x/y", commit="abc123"),
        artifacts=[Artifact(id="a1", base_model="Llama2-7B", method="AWQ",
                            quant_config={"wbits": 4, "group_size": 128})],
        claims=[Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="official", command="python e.py",
                                                 metric="perplexity", dataset="wikitext2"),
                      expected=5.78, tolerance=0.05, source="Table 3")],
    )
    ingest = IngestInfo(arxiv_id="2401.00001", title="Test Paper",
                        source_url="https://arxiv.org/abs/2401.00001",
                        repo=RepoInfo(url="https://github.com/x/y", commit="abc123"))
    grades = [ClaimGrade(claim_id="c1", verdict="MATCH", measured=5.80, expected=5.78,
                         reason="—", checks={"value": True, "faithful": True})]
    runs = [RunResult(claim_id="c1", command="python e.py", seed=0, gpu="A100x1",
                      minutes=18.0, stdout_path="runs/c1/stdout.log")]
    env = {"torch": "2.3.0", "transformers": "4.36.0", "cuda": "12.1"}
    return spec, ingest, grades, runs, env


def test_renders_both_languages():
    spec, ingest, grades, runs, env = _ctx()
    zh, en = render_reports(spec, ingest, grades, runs, env, patches=[])
    assert "复现报告" in zh
    assert "Reproduction Report" in en


def test_uses_measured_not_expected_for_actual_column():
    spec, ingest, grades, runs, env = _ctx()
    zh, _ = render_reports(spec, ingest, grades, runs, env, patches=[])
    assert "5.80" in zh        # measured present
    assert "MATCH" in zh


def test_summary_counts_verdicts():
    spec, ingest, grades, runs, env = _ctx()
    zh, _ = render_reports(spec, ingest, grades, runs, env, patches=[])
    assert "MATCH 1" in zh


def test_env_snapshot_in_report():
    spec, ingest, grades, runs, env = _ctx()
    zh, _ = render_reports(spec, ingest, grades, runs, env, patches=[])
    assert "4.36.0" in zh
    assert "abc123" in zh      # repo commit
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.report'`

- [ ] **Step 3: Write the implementation**

`src/paper_reprise/report.py`:
```python
"""Render bilingual reproduction reports (zh + en) as two markdown strings.

Iron rules (design §5.2):
  - always show measured (raw) numbers, never paper numbers as substitute
  - every claim carries replay info (command/seed/gpu/commit/env)
"""
from __future__ import annotations

from collections import Counter

from paper_reprise.models import ClaimGrade, IngestInfo, RunResult, Spec


def _summary(grades: list[ClaimGrade]) -> str:
    c = Counter(g.verdict for g in grades)
    return f"MATCH {c['MATCH']} / PARTIAL {c['PARTIAL']} / FAIL {c['FAIL']} / BLOCKED {c['BLOCKED']}"


def _env_line(ingest: IngestInfo, env: dict) -> str:
    repo = ingest.repo
    repo_s = f"{repo.url}@{repo.commit}" if repo else "(no official repo)"
    return (f"repo: {repo_s} | torch {env.get('torch','?')} / "
            f"transformers {env.get('transformers','?')} / CUDA {env.get('cuda','?')}")


def _artifact_label(spec: Spec, artifact_id: str) -> str:
    a = next((a for a in spec.artifacts if a.id == artifact_id), None)
    if not a:
        return artifact_id
    cfg = a.quant_config
    bits = cfg.get("wbits", "?")
    gs = cfg.get("group_size")
    return f"{a.base_model} W{bits}" + (f"G{gs}" if gs else "")


def _table(spec, grades, header):
    runs_by = {g.claim_id: g for g in grades}
    lines = [header, "|---|---|---|---|---|---|---|"]
    for c in spec.claims:
        g = runs_by.get(c.id)
        measured = "—" if not g or g.measured is None else f"{g.measured:.2f}"
        verdict = g.verdict if g else "BLOCKED"
        reason = g.reason if g else "no grade"
        label = _artifact_label(spec, c.artifact)
        lines.append(f"| {c.id} | {label} | {c.eval_protocol.metric} | "
                     f"{c.expected:g} | {measured} | {verdict} | {reason} |")
    return "\n".join(lines)


def _replay(runs: list[RunResult]) -> str:
    out = []
    for r in runs:
        out.append(f"- {r.claim_id}: `{r.command}` | seed {r.seed} | {r.gpu} | "
                   f"{r.minutes}min | {r.stdout_path}")
    return "\n".join(out) if out else "(none)"


def _patches(patches: list[str]) -> str:
    return "\n".join(f"- {p}" for p in patches) if patches else "(none)"


def render_reports(spec: Spec, ingest: IngestInfo, grades: list[ClaimGrade],
                   runs: list[RunResult], env: dict, patches: list[str]) -> tuple[str, str]:
    title = ingest.title or ingest.arxiv_id
    summ = _summary(grades)
    envl = _env_line(ingest, env)

    zh = f"""# 复现报告:{title} ({ingest.arxiv_id})
{envl}
判定汇总: {summ}

{_table(spec, grades, "| claim | 模型/配置 | 指标 | paper | 实测 | 判定 | 原因 |")}

## 复算信息(每条 claim)
{_replay(runs)}

## Setup 改动留痕
{_patches(patches)}
"""

    en = f"""# Reproduction Report: {title} ({ingest.arxiv_id})
{envl}
Verdict summary: {summ}

{_table(spec, grades, "| claim | model/config | metric | paper | measured | verdict | reason |")}

## Replay info (per claim)
{_replay(runs)}

## Setup patches
{_patches(patches)}
"""
    return zh, en
```

> Note: the measured column uses `{:.2f}` (not `{:g}`) so that `5.80` renders as
> `5.80`, matching the test's `"5.80" in zh` assertion.

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_report.py -v`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/report.py tests/test_report.py
git commit -m "feat: bilingual report rendering (measured-only, with replay info)"
```

---

## Task 7: Ingest — Input Normalization

**Files:**
- Create: `src/paper_reprise/ingest.py`
- Test: `tests/test_ingest.py`

This phase focuses on the purely logic-testable parts: input normalization (arxiv url / bare id → arxiv_id + source_url) and repo link discovery (scrape GitHub links out of latex/readme text). Network fetches (latex/clone) are wrapped as injectable functions, stubbed via monkeypatch in tests.

- [ ] **Step 1: Write the failing test**

`tests/test_ingest.py`:
```python
from paper_reprise.ingest import (
    normalize_input, find_repo_url, arxiv_id_from_url,
)


def test_arxiv_id_from_abs_url():
    assert arxiv_id_from_url("https://arxiv.org/abs/2606.18114") == "2606.18114"


def test_arxiv_id_from_versioned_url():
    assert arxiv_id_from_url("https://arxiv.org/abs/2401.00001v2") == "2401.00001"


def test_normalize_input_from_bare_id():
    arxiv_id, url = normalize_input("2401.00001")
    assert arxiv_id == "2401.00001"
    assert url == "https://arxiv.org/abs/2401.00001"


def test_normalize_input_from_abs_url():
    arxiv_id, url = normalize_input("https://arxiv.org/abs/2401.00001")
    assert arxiv_id == "2401.00001"


def test_find_repo_url_picks_github_link():
    text = "We release code at https://github.com/foo/bar for reproduction."
    assert find_repo_url(text) == "https://github.com/foo/bar"


def test_find_repo_url_none_when_absent():
    assert find_repo_url("no links here") is None
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.ingest'`

- [ ] **Step 3: Write the implementation**

`src/paper_reprise/ingest.py`:
```python
"""Ingest stage: normalize input to (arxiv_id, source_url), discover official repo.

Network fetches (latex tarball, git clone) are isolated behind functions that
callers can patch in tests. The parsing/normalization logic is pure.
"""
from __future__ import annotations

import re
from typing import Optional

_ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")
_GITHUB_RE = re.compile(r"https?://github\.com/[\w.\-]+/[\w.\-]+")


def arxiv_id_from_url(url: str) -> Optional[str]:
    m = _ARXIV_RE.search(url)
    return m.group(1) if m else None


def find_repo_url(text: str) -> Optional[str]:
    m = _GITHUB_RE.search(text)
    if not m:
        return None
    return m.group(0).rstrip("/").removesuffix(".git")


def normalize_input(arg: str) -> tuple[str, str]:
    """Return (arxiv_id, source_url) from an arxiv url or bare arxiv id."""
    if arg.startswith("http"):
        arxiv_id = arxiv_id_from_url(arg)
        if not arxiv_id:
            raise ValueError(f"cannot extract arxiv id from {arg}")
        return arxiv_id, f"https://arxiv.org/abs/{arxiv_id}"
    if _ARXIV_RE.fullmatch(arg):
        return arg, f"https://arxiv.org/abs/{arg}"
    raise ValueError(f"unrecognized input: {arg}")
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/ingest.py tests/test_ingest.py
git commit -m "feat: ingest input normalization + repo discovery"
```

---

## Task 8: Plan — Feasibility/Anomaly Sentinel

**Files:**
- Create: `src/paper_reprise/planstage.py`
- Test: `tests/test_planstage.py`

plan passes silently by default (cost not a constraint). It sets `needs_user_decision=True` only in two cases: infeasible hardware, or estimate wildly diverging from the paper's self-report (a quality signal, usually meaning specextract got something wrong).

- [ ] **Step 1: Write the failing test**

`tests/test_planstage.py`:
```python
from paper_reprise.models import Spec, Artifact, Claim, EvalProtocol
from paper_reprise.planstage import build_plan


def _spec_with_hw(hardware):
    return Spec(
        paper="p", repo=None,
        artifacts=[Artifact(id="a1", base_model="m", method="AWQ", quant_config={})],
        claims=[Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="official", command="c",
                                                 metric="perplexity", dataset="wikitext2"),
                      expected=1.0, tolerance=0.05, source="T", hardware=hardware)],
    )


def test_silent_pass_when_all_feasible():
    plan = build_plan(_spec_with_hw(None), available_hardware=["A100-80G"])
    assert plan.needs_user_decision is False
    assert plan.claims[0].feasible is True


def test_flags_infeasible_hardware():
    plan = build_plan(_spec_with_hw("H200-141G x8"), available_hardware=["A100-80G"])
    assert plan.needs_user_decision is True
    assert plan.claims[0].feasible is False
    assert "H200" in (plan.decision_reason or "")


def test_no_hardware_requirement_is_feasible():
    plan = build_plan(_spec_with_hw(None), available_hardware=[])
    assert plan.claims[0].feasible is True
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_planstage.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.planstage'`

- [ ] **Step 3: Write the implementation**

`src/paper_reprise/planstage.py`:
```python
"""Plan stage: feasibility / anomaly sentinel.

Default: silent pass (compute is not a constraint). Escalates to a user decision
only on (1) infeasible hardware, or (2) estimate wildly diverging from the paper.
"""
from __future__ import annotations

from paper_reprise.models import ClaimPlan, PlanReport, Spec


def _hardware_feasible(required: str | None, available: list[str]) -> bool:
    if not required:
        return True
    # crude check: the required GPU family token must appear in some available entry
    family = required.split("-")[0].split()[0].upper()   # "H200-141G x8" -> "H200"
    return any(family in a.upper() for a in available)


def build_plan(spec: Spec, available_hardware: list[str]) -> PlanReport:
    claims: list[ClaimPlan] = []
    reasons: list[str] = []
    for c in spec.claims:
        feasible = _hardware_feasible(c.hardware, available_hardware)
        anomaly = None
        if not feasible:
            anomaly = f"硬件不可行: 需要 {c.hardware},可用 {available_hardware}"
            reasons.append(f"{c.id}: {anomaly}")
        claims.append(ClaimPlan(claim_id=c.id, required_hardware=c.hardware,
                                feasible=feasible, anomaly=anomaly))

    needs = len(reasons) > 0
    return PlanReport(claims=claims, needs_user_decision=needs,
                      decision_reason="; ".join(reasons) if needs else None)
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_planstage.py -v`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/planstage.py tests/test_planstage.py
git commit -m "feat: plan stage feasibility/anomaly sentinel (silent pass by default)"
```

---

## Task 9: Headless — claude -p Wrapper

**Files:**
- Create: `src/paper_reprise/headless.py`
- Test: `tests/test_headless.py`

Reuse llm-paper-radar's pattern: `claude -p --permission-mode acceptEdits --allowedTools "..."`, prompt via stdin, **don't trust the exit code — judge success by whether the output file appeared**. The subprocess call is wrapped as injectable, monkeypatched in tests.

- [ ] **Step 1: Write the failing test**

`tests/test_headless.py`:
```python
from pathlib import Path

import paper_reprise.headless as headless
from paper_reprise.headless import run_headless


def test_success_when_output_file_appears(tmp_path, monkeypatch):
    out = tmp_path / "spec.yaml"

    def fake_call(prompt, allowed_tools, cwd):
        out.write_text("ok")          # simulate claude writing the file
        return 0

    monkeypatch.setattr(headless, "_call_claude", fake_call)
    res = run_headless(prompt="make spec", allowed_tools=["Write"],
                       cwd=tmp_path, expect_file=out)
    assert res.ok is True
    assert res.output_path == out


def test_failure_when_output_missing(tmp_path, monkeypatch):
    out = tmp_path / "spec.yaml"
    monkeypatch.setattr(headless, "_call_claude", lambda *a, **k: 0)
    res = run_headless(prompt="x", allowed_tools=["Write"], cwd=tmp_path, expect_file=out)
    assert res.ok is False
    assert "did not appear" in res.error


def test_failure_when_nonzero_and_missing(tmp_path, monkeypatch):
    out = tmp_path / "spec.yaml"
    monkeypatch.setattr(headless, "_call_claude", lambda *a, **k: 3)
    res = run_headless(prompt="x", allowed_tools=["Write"], cwd=tmp_path, expect_file=out)
    assert res.ok is False
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_headless.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.headless'`

- [ ] **Step 3: Write the implementation**

`src/paper_reprise/headless.py`:
```python
"""Wrapper around `claude -p` headless invocation (mirrors llm-paper-radar).

Success is determined by the expected output file appearing — NOT by exit code,
because the skill can exit 0 while silently failing to write.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class HeadlessResult:
    ok: bool
    output_path: Optional[Path] = None
    error: str = ""
    exit_code: int = 0


def _call_claude(prompt: str, allowed_tools: list[str], cwd: Path) -> int:
    """Invoke `claude -p`, prompt via stdin. Returns exit code."""
    proc = subprocess.run(
        ["claude", "-p", "--permission-mode", "acceptEdits",
         "--allowedTools", ",".join(allowed_tools)],
        input=prompt, text=True, cwd=str(cwd),
    )
    return proc.returncode


def run_headless(prompt: str, allowed_tools: list[str], cwd: Path,
                 expect_file: Path) -> HeadlessResult:
    code = _call_claude(prompt, allowed_tools, cwd)
    if expect_file.exists():
        return HeadlessResult(ok=True, output_path=expect_file, exit_code=code)
    return HeadlessResult(ok=False, exit_code=code,
                          error=f"expected output {expect_file} did not appear "
                                f"(exit={code})")
```

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_headless.py -v`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/headless.py tests/test_headless.py
git commit -m "feat: headless claude -p wrapper with output-file verification"
```

---

## Task 10: SpecExtract Stage (headless, mockable)

**Files:**
- Create: `src/paper_reprise/specextract.py`
- Create: `tests/fixtures/extracted_spec.yaml`
- Test: `tests/test_specextract.py`

specextract builds the prompt, calls headless to produce `spec.yaml`, and validates it parses into a `Spec`. This phase uses a mock headless that writes the fixture spec to verify the assembly. The gate (waiting for the user to review the spec) lives in the pipeline layer; this module only generates + validates.

- [ ] **Step 1: Write the fixture**

`tests/fixtures/extracted_spec.yaml`:
```yaml
paper: "2401.00001"
repo:
  url: https://github.com/example/awq
  commit: null
artifacts:
  - id: llama2-7b-w4g128
    base_model: meta-llama/Llama-2-7b-hf
    method: AWQ
    quant_config:
      wbits: 4
      group_size: 128
    calib_status: known
claims:
  - id: c1
    artifact: llama2-7b-w4g128
    eval_protocol:
      runner: official
      command: "python eval_ppl.py --model {model} --dataset wikitext2"
      metric: perplexity
      dataset: wikitext2
      seqlen: 2048
      stride: 2048
      few_shot: 0
    expected: 5.78
    tolerance: 0.05
    source: "Table 3, row 2"
    hardware: null
```

- [ ] **Step 2: Write the failing test**

`tests/test_specextract.py`:
```python
import shutil
from pathlib import Path

import paper_reprise.specextract as specextract
from paper_reprise.headless import HeadlessResult
from paper_reprise.rundir import RunDir

FIX = Path(__file__).parent / "fixtures"


def test_specextract_produces_valid_spec(tmp_path, monkeypatch):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    (rd.paper_dir / "main.tex").write_text("dummy latex")

    def fake_headless(prompt, allowed_tools, cwd, expect_file):
        shutil.copy(FIX / "extracted_spec.yaml", expect_file)
        return HeadlessResult(ok=True, output_path=expect_file)

    monkeypatch.setattr(specextract, "run_headless", fake_headless)
    spec = specextract.extract_spec(rd)
    assert spec is not None
    assert spec.claims[0].id == "c1"
    assert (rd.root / "spec.yaml").exists()


def test_specextract_returns_none_when_headless_fails(tmp_path, monkeypatch):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    monkeypatch.setattr(specextract, "run_headless",
                        lambda **k: HeadlessResult(ok=False, error="boom"))
    assert specextract.extract_spec(rd) is None


def test_specextract_returns_none_on_invalid_yaml(tmp_path, monkeypatch):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")

    def fake_headless(prompt, allowed_tools, cwd, expect_file):
        Path(expect_file).write_text("paper: x\nclaims: [bad]\n")  # invalid Spec
        return HeadlessResult(ok=True, output_path=expect_file)

    monkeypatch.setattr(specextract, "run_headless", fake_headless)
    assert specextract.extract_spec(rd) is None
```

- [ ] **Step 3: Run the test, confirm it fails**

Run: `uv run pytest tests/test_specextract.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.specextract'`

- [ ] **Step 4: Write the implementation**

`src/paper_reprise/specextract.py`:
```python
"""SpecExtract stage: one headless claude call → spec.yaml → validated Spec.

The approval gate (user reviews spec.yaml) is enforced by the pipeline, not here.
"""
from __future__ import annotations

from typing import Optional

import yaml

from paper_reprise.headless import run_headless
from paper_reprise.models import Spec
from paper_reprise.rundir import RunDir

_PROMPT_TEMPLATE = """You are extracting a machine-checkable reproduction spec from a \
quantization paper. Read the LaTeX sources in `paper/` and the official repo README in \
`repo/` (if present). Extract ONLY the paper's MAIN results (the headline/bolded claims), \
not every table cell.

Write a YAML file to `{out}` with this exact schema:
- paper: arxiv id
- repo: {{url, commit}} or null
- artifacts: list of {{id, base_model, method, quant_config, calib_status}}
- claims: list of {{id, artifact, eval_protocol, expected, tolerance, source, hardware}}
  where eval_protocol = {{runner, command, metric, dataset, split, seqlen, stride, \
few_shot, extra_args}}

Rules:
- runner: prefer "official" (the repo's own eval script). Use "cited-standard" if the \
paper explicitly cites a standard impl; "custom" only as last resort.
- If calibration config cannot be determined, set calib_status: UNKNOWN.
- Default tolerance: perplexity 0.05, accuracy 0.5. If the paper states one, use it.
- source: pin each claim to its location, e.g. "Table 3, row 2, col W4".

Write ONLY the YAML file. Report 'Saved: {out}' when done."""


def build_prompt(rd: RunDir) -> str:
    return _PROMPT_TEMPLATE.format(out=rd.root / "spec.yaml")


def extract_spec(rd: RunDir) -> Optional[Spec]:
    out = rd.root / "spec.yaml"
    res = run_headless(prompt=build_prompt(rd),
                       allowed_tools=["Read", "Write", "Bash"],
                       cwd=rd.root, expect_file=out)
    if not res.ok:
        return None
    try:
        spec = Spec.model_validate(yaml.safe_load(out.read_text()))
    except Exception:
        return None
    return spec
```

- [ ] **Step 5: Run the test, confirm it passes**

Run: `uv run pytest tests/test_specextract.py -v`
Expected: PASS, all tests green

- [ ] **Step 6: Commit**

```bash
git add src/paper_reprise/specextract.py tests/test_specextract.py tests/fixtures/extracted_spec.yaml
git commit -m "feat: specextract stage (headless → validated Spec)"
```

---

## Task 11: Setup / Run Stage Interfaces and Stubs

**Files:**
- Create: `src/paper_reprise/setupstage.py`
- Create: `src/paper_reprise/runstage.py`
- Test: `tests/test_stages_stub.py`

No real GPU this phase. setup returns an env snapshot (stub); the ability for run to persist "what the eval command should output" is left to Plan 2. For now run provides a stub that consumes the RunResult contract and supports injecting an "executor" for pipeline testing. This way both paths (official-repo / from-scratch) will later fill the same executor interface.

- [ ] **Step 1: Write the failing test**

`tests/test_stages_stub.py`:
```python
from pathlib import Path

from paper_reprise.models import Spec, Artifact, Claim, EvalProtocol
from paper_reprise.rundir import RunDir
from paper_reprise.setupstage import run_setup, SetupResult
from paper_reprise.runstage import run_claims


def _spec():
    return Spec(
        paper="p", repo=None,
        artifacts=[Artifact(id="a1", base_model="m", method="AWQ",
                            quant_config={"wbits": 4, "seqlen": 2048})],
        claims=[Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="official", command="echo ppl",
                                                 metric="perplexity", dataset="wikitext2",
                                                 seqlen=2048),
                      expected=5.78, tolerance=0.05, source="T")],
    )


def test_setup_stub_returns_env_snapshot(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    res = run_setup(rd, _spec(), executor=None)
    assert isinstance(res, SetupResult)
    assert res.ok is True
    assert "torch" in res.env_snapshot


def test_run_claims_writes_stdout_and_returns_results(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    spec = _spec()

    def fake_executor(claim, artifact, claim_dir):
        log = Path(claim_dir) / "stdout.log"
        log.write_text("perplexity: 5.80")
        return {"stdout_path": str(log), "actual_config": {"seqlen": 2048},
                "gpu": "A100x1", "seed": 0, "minutes": 1.0}

    results, configs = run_claims(rd, spec, executor=fake_executor)
    assert results[0].stdout_path.endswith("stdout.log")
    assert configs["c1"]["seqlen"] == 2048
    assert Path(results[0].stdout_path).read_text() == "perplexity: 5.80"


def test_run_claims_marks_blocked_on_executor_error(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    spec = _spec()

    def boom_executor(claim, artifact, claim_dir):
        raise RuntimeError("kernel compile failed")

    results, _ = run_claims(rd, spec, executor=boom_executor)
    assert results[0].status == "blocked"
    assert "kernel compile failed" in results[0].block_reason
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_stages_stub.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.setupstage'`

- [ ] **Step 3: Write setupstage.py**

`src/paper_reprise/setupstage.py`:
```python
"""Setup stage: the (future) agentic env-debug loop.

Plan 1 provides a stub returning a placeholder env snapshot. Plan 2 will replace
the body with a bounded headless claude loop that builds a conda/uv env and runs
the repo's smoke test until it passes once. The signature stays stable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from paper_reprise.models import Spec
from paper_reprise.rundir import RunDir


@dataclass
class SetupResult:
    ok: bool
    env_snapshot: dict = field(default_factory=dict)
    patches: list[str] = field(default_factory=list)
    error: str = ""


def run_setup(rd: RunDir, spec: Spec,
              executor: Optional[Callable] = None) -> SetupResult:
    if executor is not None:
        return executor(rd, spec)
    # Plan 1 stub: pretend the env was built. Real impl lands in Plan 2.
    snapshot = {"torch": "stub", "transformers": "stub", "cuda": "stub"}
    (rd.root / "env_snapshot.json").write_text(json.dumps(snapshot, indent=2))
    return SetupResult(ok=True, env_snapshot=snapshot, patches=[])
```

- [ ] **Step 4: Write runstage.py**

`src/paper_reprise/runstage.py`:
```python
"""Run stage: deterministic execution of quant + eval per claim.

The actual quantization/eval is delegated to an `executor` callable (one impl per
provider: official-repo now, from-scratch later). Plan 1 wires the contract and
the persist/blocked-handling; Plan 2 supplies the real GPU executor.

executor(claim, artifact, claim_dir) -> dict with keys:
  stdout_path, actual_config, gpu, seed, minutes
"""
from __future__ import annotations

from typing import Callable

from paper_reprise.models import RunResult, Spec
from paper_reprise.rundir import RunDir


def run_claims(rd: RunDir, spec: Spec,
               executor: Callable) -> tuple[list[RunResult], dict]:
    artifacts = {a.id: a for a in spec.artifacts}
    results: list[RunResult] = []
    actual_configs: dict = {}
    for claim in spec.claims:
        claim_dir = rd.claim_dir(claim.id)
        artifact = artifacts[claim.artifact]
        try:
            out = executor(claim, artifact, claim_dir)
            results.append(RunResult(
                claim_id=claim.id, command=claim.eval_protocol.command,
                seed=out.get("seed"), gpu=out.get("gpu"), minutes=out.get("minutes"),
                stdout_path=out["stdout_path"], status="ran"))
            actual_configs[claim.id] = out.get("actual_config", {})
        except Exception as e:
            results.append(RunResult(
                claim_id=claim.id, command=claim.eval_protocol.command,
                stdout_path=str(claim_dir / "stdout.log"),
                status="blocked", block_reason=str(e)))
            actual_configs[claim.id] = {}
    return results, actual_configs
```

- [ ] **Step 5: Run the test, confirm it passes**

Run: `uv run pytest tests/test_stages_stub.py -v`
Expected: PASS, all tests green

- [ ] **Step 6: Commit**

```bash
git add src/paper_reprise/setupstage.py src/paper_reprise/runstage.py tests/test_stages_stub.py
git commit -m "feat: setup/run stage interfaces with injectable executor (stubs)"
```

---

## Task 12: Pipeline Orchestration + Gates

**Files:**
- Create: `src/paper_reprise/pipeline.py`
- Test: `tests/test_pipeline.py`

Wire the stages together. Gate 1 (spec approval) and the plan sentinel are implemented via injected callbacks, so tests can auto-approve while the real CLI pops an AskUserQuestion / waits for the user. grade uses Task 5's `grade_claim` per claim; report uses Task 6 to render.

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
import shutil
from pathlib import Path

import paper_reprise.pipeline as pipeline
from paper_reprise.setupstage import SetupResult

FIX = Path(__file__).parent / "fixtures"


def _fake_specextract(rd):
    shutil.copy(FIX / "extracted_spec.yaml", rd.root / "spec.yaml")
    import yaml
    from paper_reprise.models import Spec
    return Spec.model_validate(yaml.safe_load((rd.root / "spec.yaml").read_text()))


def _fake_setup(rd, spec):
    return SetupResult(ok=True, env_snapshot={"torch": "2.3", "transformers": "4.36",
                                              "cuda": "12.1"}, patches=[])


def _fake_executor(claim, artifact, claim_dir):
    log = Path(claim_dir) / "stdout.log"
    log.write_text("perplexity: 5.80")     # within tol of 5.78
    return {"stdout_path": str(log), "actual_config": {"seqlen": 2048},
            "gpu": "A100x1", "seed": 0, "minutes": 1.0}


def test_full_pipeline_produces_match_report(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "extract_spec", _fake_specextract)
    result = pipeline.run_pipeline(
        input_arg="2401.00001", base_dir=tmp_path, timestamp="t",
        available_hardware=["A100-80G"],
        approve_spec=lambda spec: True,           # gate 1: auto-approve
        approve_plan=lambda plan: True,
        fetch_sources=lambda rd, arxiv_id, url: None,   # skip network
        setup_executor=_fake_setup,
        run_executor=_fake_executor,
    )
    assert (result.root / "report.zh.md").exists()
    assert (result.root / "report.en.md").exists()
    assert "MATCH 1" in (result.root / "report.zh.md").read_text()


def test_pipeline_aborts_when_spec_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "extract_spec", _fake_specextract)
    result = pipeline.run_pipeline(
        input_arg="2401.00001", base_dir=tmp_path, timestamp="t",
        available_hardware=["A100-80G"],
        approve_spec=lambda spec: False,          # reject
        approve_plan=lambda plan: True,
        fetch_sources=lambda rd, arxiv_id, url: None,
        setup_executor=_fake_setup, run_executor=_fake_executor,
    )
    assert not (result.root / "report.zh.md").exists()
    assert result.aborted_at == "spec-approval"
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.pipeline'`

- [ ] **Step 3: Write the implementation**

`src/paper_reprise/pipeline.py`:
```python
"""Deterministic orchestration of the 7 stages with two gates.

Gates and side-effecting stages are injected as callables so tests run offline
and the CLI can supply interactive prompts / real executors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from paper_reprise.grade import grade_claim
from paper_reprise.ingest import normalize_input
from paper_reprise.models import IngestInfo
from paper_reprise.planstage import build_plan
from paper_reprise.report import render_reports
from paper_reprise.rundir import RunDir
from paper_reprise.runstage import run_claims
from paper_reprise.setupstage import run_setup
from paper_reprise.specextract import extract_spec


@dataclass
class PipelineResult:
    root: Path
    aborted_at: Optional[str] = None


def run_pipeline(
    input_arg: str,
    base_dir: Path,
    timestamp: str,
    available_hardware: list[str],
    approve_spec: Callable,
    approve_plan: Callable,
    fetch_sources: Callable,
    setup_executor: Optional[Callable],
    run_executor: Callable,
) -> PipelineResult:
    # --- ingest ---
    arxiv_id, url = normalize_input(input_arg)
    rd = RunDir.create(base_dir, arxiv_id=arxiv_id, timestamp=timestamp)
    fetch_sources(rd, arxiv_id, url)            # fills paper/ and repo/ (network)
    ingest = IngestInfo(arxiv_id=arxiv_id, source_url=url)
    rd.write_ingest(ingest)

    # --- specextract + gate 1 ---
    spec = extract_spec(rd)
    if spec is None:
        return PipelineResult(root=rd.root, aborted_at="specextract")
    if not approve_spec(spec):
        return PipelineResult(root=rd.root, aborted_at="spec-approval")
    rd.write_spec(spec)

    # --- plan + sentinel ---
    plan = build_plan(spec, available_hardware)
    rd.write_plan(plan)
    if plan.needs_user_decision and not approve_plan(plan):
        return PipelineResult(root=rd.root, aborted_at="plan")

    # --- setup ---
    setup = run_setup(rd, spec, executor=setup_executor)
    if not setup.ok:
        return PipelineResult(root=rd.root, aborted_at="setup")

    # --- run ---
    runs, actual_configs = run_claims(rd, spec, executor=run_executor)

    # --- grade (pure code, isolated) ---
    artifacts = {a.id: a for a in spec.artifacts}
    runs_by_claim = {r.claim_id: r for r in runs}
    grades = [grade_claim(c, artifacts[c.artifact], runs_by_claim[c.id],
                          actual_configs.get(c.id, {}))
              for c in spec.claims]

    # --- report ---
    ingest.repo = spec.repo
    zh, en = render_reports(spec, ingest, grades, runs, setup.env_snapshot,
                            patches=setup.patches)
    (rd.root / "report.zh.md").write_text(zh)
    (rd.root / "report.en.md").write_text(en)
    return PipelineResult(root=rd.root, aborted_at=None)
```

> Note: the shipped version reuses the `ingest` object built during ingest (rather
> than re-reading it from disk) and looks up runs via a `runs_by_claim` dict. The
> abort branches (specextract / spec-approval / plan / setup) each have a test.

- [ ] **Step 4: Run the test, confirm it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS, all tests green

- [ ] **Step 5: Commit**

```bash
git add src/paper_reprise/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestration with spec-approval + plan gates"
```

---

## Task 13: CLI

**Files:**
- Create: `src/paper_reprise/cli.py`
- Modify: `pyproject.toml` (add `[project.scripts]`)
- Test: `tests/test_cli.py`

The `run` command wires the real dependencies (network fetch, interactive gates, real executor) into the pipeline. This phase still uses placeholders for the network/GPU executor (replaced in Plan 2), but the CLI assembly, the `--yes` auto-approve, and the `report` re-render are testable.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
from click.testing import CliRunner

from paper_reprise.cli import cli


def test_cli_help():
    res = CliRunner().invoke(cli, ["--help"])
    assert res.exit_code == 0
    assert "run" in res.output
    assert "report" in res.output


def test_cli_run_help_lists_yes_flag():
    res = CliRunner().invoke(cli, ["run", "--help"])
    assert res.exit_code == 0
    assert "--yes" in res.output


def test_cli_report_rerenders(tmp_path, monkeypatch):
    # build a minimal run dir with spec + grades already present
    from paper_reprise.rundir import RunDir
    from paper_reprise.models import (Spec, Artifact, Claim, EvalProtocol, RepoInfo,
                                    IngestInfo)
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    spec = Spec(paper="2401.00001", repo=RepoInfo(url="u", commit="c"),
                artifacts=[Artifact(id="a1", base_model="m", method="AWQ",
                                    quant_config={"wbits": 4})],
                claims=[Claim(id="c1", artifact="a1",
                              eval_protocol=EvalProtocol(runner="official", command="c",
                                                         metric="perplexity",
                                                         dataset="wikitext2"),
                              expected=5.78, tolerance=0.05, source="T")])
    rd.write_spec(spec)
    rd.write_ingest(IngestInfo(arxiv_id="2401.00001", source_url="u",
                               repo=RepoInfo(url="u", commit="c")))
    (rd.claim_dir("c1") / "stdout.log").write_text("perplexity: 5.80")

    res = CliRunner().invoke(cli, ["report", str(rd.root)])
    assert res.exit_code == 0
    assert (rd.root / "report.zh.md").exists()
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'paper_reprise.cli'`

- [ ] **Step 3: Write the implementation**

`src/paper_reprise/cli.py`:
```python
"""paper-reprise CLI: run / resume / report."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import click

from paper_reprise.grade import grade_claim
from paper_reprise.report import render_reports
from paper_reprise.rundir import RunDir


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


@click.group()
def cli() -> None:
    """Reproduce quantization paper results."""


@cli.command()
@click.argument("input_arg")
@click.option("--base-dir", default="runs", help="where run dirs are created")
@click.option("--yes", is_flag=True, help="auto-approve all gates (non-interactive)")
def run(input_arg: str, base_dir: str, yes: bool) -> None:
    """Run the reproduction pipeline for a paper (arxiv id or url)."""
    from paper_reprise.pipeline import run_pipeline

    def approve_spec(spec):
        if yes:
            return True
        click.echo(f"\nExtracted {len(spec.claims)} claims. Review spec.yaml.")
        return click.confirm("Approve spec and continue?", default=True)

    def approve_plan(plan):
        if yes:
            return True
        click.echo(f"\nPlan flagged: {plan.decision_reason}")
        return click.confirm("Proceed anyway?", default=False)

    def fetch_sources(rd, arxiv_id, url):
        # Plan 2: fetch latex tarball + git clone. Plan 1: no-op placeholder.
        click.echo(f"[ingest] {arxiv_id} (source fetch deferred to Plan 2)")

    def run_executor(claim, artifact, claim_dir):
        raise RuntimeError("real GPU executor not implemented (Plan 2)")

    result = run_pipeline(
        input_arg=input_arg, base_dir=Path(base_dir), timestamp=_timestamp(),
        available_hardware=[], approve_spec=approve_spec, approve_plan=approve_plan,
        fetch_sources=fetch_sources, setup_executor=None, run_executor=run_executor,
    )
    if result.aborted_at:
        click.echo(f"Aborted at: {result.aborted_at}")
    else:
        click.echo(f"Done. Report: {result.root}/report.zh.md")


@cli.command()
@click.argument("run_dir")
def report(run_dir: str) -> None:
    """Re-render reports from an existing run dir."""
    rd = RunDir.open(Path(run_dir))
    spec = rd.read_spec()
    ingest = rd.read_ingest()
    if spec is None or ingest is None:
        raise click.ClickException("run dir missing spec.yaml or ingest.json")

    artifacts = {a.id: a for a in spec.artifacts}
    from paper_reprise.models import RunResult
    grades, runs = [], []
    for c in spec.claims:
        log = rd.claim_dir(c.id) / "stdout.log"
        rr = RunResult(claim_id=c.id, command=c.eval_protocol.command,
                       stdout_path=str(log),
                       status="ran" if log.exists() else "blocked",
                       block_reason=None if log.exists() else "no stdout.log")
        runs.append(rr)
        grades.append(grade_claim(c, artifacts[c.artifact], rr, actual_config={}))

    zh, en = render_reports(spec, ingest, grades, runs, env={}, patches=[])
    (rd.root / "report.zh.md").write_text(zh)
    (rd.root / "report.en.md").write_text(en)
    click.echo(f"Re-rendered: {rd.root}/report.zh.md")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Add the console-script entry**

Insert into `pyproject.toml` after the `[project]` table (after the `dependencies` array, before `[dependency-groups]`):
```toml
[project.scripts]
paper-reprise = "paper_reprise.cli:cli"
```

- [ ] **Step 5: Run the test, confirm it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS, all tests green

- [ ] **Step 6: Full suite + CLI smoke**

Run: `uv run pytest -q && uv sync && uv run paper-reprise --help`
Expected: all tests pass; `paper-reprise --help` prints the run/report subcommands

- [ ] **Step 7: Commit**

```bash
git add src/paper_reprise/cli.py pyproject.toml tests/test_cli.py uv.lock
git commit -m "feat: click CLI (run/report) with console script entrypoint"
```

---

## Task 14: End-to-End Smoke + conftest Cleanup

**Files:**
- Create: `tests/conftest.py`
- Test: reuse existing

- [ ] **Step 1: Write shared fixtures (dedup)**

`tests/conftest.py`:
```python
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
```

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS, all tasks' tests green (~50+ cases)

- [ ] **Step 3: ruff check**

Run: `uv run ruff check src/ tests/`
Expected: no errors (auto-fixable warnings via `uv run ruff check --fix`)

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: shared fixtures + full-suite green"
```

---

## Self-Review

**1. Spec coverage** — item by item against the design:

- §2 seven stages: ingest (Task 7) / specextract (Task 10) / plan (Task 8) / setup (Task 11) / run (Task 11) / grade (Task 5) / report (Task 6) / orchestration (Task 12). ✓
- §2.1 gate 1 + plan sentinel: pipeline's `approve_spec` / `approve_plan` + `build_plan.needs_user_decision` (Task 8/12). ✓
- §2.2 judge isolation: `grade.py` does not import `runstage`, reads only persisted files (Task 5). ✓
- §3.1 ingest input normalization (arxiv id/url) + repo discovery: `normalize_input` / `find_repo_url` (Task 7). latex fetch/clone explicitly deferred to Plan 2 (`fetch_sources` placeholder), not an omission. ✓
- §3.2 two-layer spec schema: `models.py`'s Artifact/Claim (Task 2). ✓
- §3.3 runner / calib_status / source: all model fields, used in grade/specextract. ✓
- §4.1 setup's single exit condition + guardrails: interface+stub (Task 11), real loop deferred to Plan 2 (noted in the file docstring). ✓
- §4.2 run prefers official reproduction command: carried by the executor interface, implemented in Plan 2. ✓
- §5.1 four verdicts: Task 5 covers them fully, including BLOCKED separated from FAIL, calib UNKNOWN, unparseable. ✓
- §5.2 bilingual two files + measured numbers + replay info: Task 6. ✓
- §6 from-scratch reserved interface: the `executor` injection point of `run_setup`/`run_claims` is the provider interface. ✓
- §7 run directory layout: Task 3 RunDir. ✓
- §8 YAGNI: no queue/DB/cron/cost-gate/LLM-grading. ✓

**Deferred to Plan 2 (not omissions, explicitly scoped out):** real latex fetch and git clone, real conda/uv setup debug loop, real GPU quant+eval executor, real setup_patches collection, real env_snapshot capture, title-based input (title → arxiv_id needs an online search). Plan 1 stubs/interfaces them all and tests the assembly.

**2. Placeholder scan** — no "TBD/TODO/fill in later" in the plan; all code steps give complete runnable code. The stubs are an intentional scope split, noted in docstrings and self-review, not placeholder debt.

**3. Type consistency** — full pass: `grade_claim(claim, artifact, run, actual_config)` defined in Task 5, called consistently in Task 12; `run_claims(rd, spec, executor) -> (list[RunResult], dict)` defined in Task 11, unpacked consistently in Task 12; `render_reports(spec, ingest, grades, runs, env, patches)` defined in Task 6, called consistently in Task 12/13; `run_headless(prompt, allowed_tools, cwd, expect_file)` defined in Task 9, called consistently in Task 10; `RunResult.stdout_path` is str, grade wraps it with `Path(run.stdout_path)` consistently. ✓
