# paper-repro 确定性骨架 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭起 paper-repro 的确定性流水线骨架——能端到端跑通 `ingest → specextract → plan → setup → run → grade → report`,其中判分(grade)、报告(report)、数据契约(models)全部 TDD 实现并锁死;specextract/setup/run 用 mock claude 和 fixture 跑通接口,真实 GPU 量化/评测留给后续 Plan 2。

**Architecture:** 确定性 Python 编排,每阶段读写带类型的 artifact 到 run 目录(方案 B)。agent 的不确定性只在 setup 阶段(本期为接口+stub)。grade 是纯代码、与执行隔离,只读落盘的原始输出 + spec,实现"过程忠实 AND 数值达标"双检与 MATCH/PARTIAL/FAIL/BLOCKED 四态。

**Tech Stack:** Python 3.12,pydantic v2(数据模型),click(CLI),pyyaml(spec),pytest(测试),uv(依赖/虚拟环境),`claude -p` headless(specextract/setup,复用 llm-paper-radar 的调用模式)。

设计文档:`docs/superpowers/specs/2026-06-19-paper-repro-agent-design.md`

---

## File Structure

```
paper-repro/
  pyproject.toml                  # 项目元数据 + 依赖 + pytest/ruff 配置
  src/paper_repro/
    __init__.py
    models.py                     # pydantic 模型:EvalProtocol/Artifact/Claim/Spec/IngestInfo/PlanReport/ClaimGrade/RunResult
    rundir.py                     # RunDir:目录布局 + artifact 读写
    parsers.py                    # 指标输出解析器:PPL / accuracy / speedup
    grade.py                      # 纯代码判分:数值 + 忠实双检 → 四态
    report.py                     # 渲染 report.zh.md / report.en.md
    ingest.py                     # 入参归一(.org/url/id)+ latex 抓取 + repo 定位
    planstage.py                  # 可行性/异常哨兵
    headless.py                   # claude -p 封装,输出文件校验(不信 exit code)
    specextract.py                # specextract 阶段:调 headless → spec.yaml → 门控
    setupstage.py                 # setup 阶段:agentic 调试循环(接口 + stub)
    runstage.py                   # run 阶段:量化+评测(接口 + stub)
    pipeline.py                   # 编排 7 阶段 + 门控
    cli.py                        # click CLI:run / resume / report
  tests/
    conftest.py                   # 共享 fixture
    fixtures/                     # 样例 .org / spec.yaml / 评测输出
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

**职责边界:** 每个文件一个清晰职责。`grade.py` 绝不 import `runstage.py`(判分与执行隔离,只通过 RunDir 落盘文件通信)。`models.py` 不依赖任何其他本包模块(纯契约)。

---

## Task 1: 项目脚手架

**Files:**
- Create: `pyproject.toml`
- Create: `src/paper_repro/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: 写 pyproject.toml**

```toml
[project]
name = "paper-repro"
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
packages = ["src/paper_repro"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 2: 建包入口**

`src/paper_repro/__init__.py`:
```python
"""paper-repro: reproduce quantization paper results."""

__version__ = "0.1.0"
```

`tests/__init__.py`:
```python
```

- [ ] **Step 3: 同步依赖、建虚拟环境**

Run: `cd /proj/xcohdstaff7/zhaolin/code/paper-repro && uv sync`
Expected: 创建 `.venv/`,装好 click/pydantic/pyyaml/httpx/pytest/ruff,无报错。

- [ ] **Step 4: 验证 import**

Run: `uv run python -c "import paper_repro; print(paper_repro.__version__)"`
Expected: 打印 `0.1.0`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/paper_repro/__init__.py tests/__init__.py uv.lock
git commit -m "chore: project scaffold with uv + pytest"
```

---

## Task 2: 数据模型(契约)

**Files:**
- Create: `src/paper_repro/models.py`
- Test: `tests/test_models.py`

这些 pydantic 模型是所有阶段之间的契约,字段名一旦定下,后续 task 必须严格沿用。

- [ ] **Step 1: 写失败测试**

`tests/test_models.py`:
```python
import pytest
from pydantic import ValidationError

from paper_repro.models import (
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

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'paper_repro.models'`

- [ ] **Step 3: 写实现**

`src/paper_repro/models.py`:
```python
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
    stdout_path: str                 # path to raw落盘 output
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

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS,9 个测试全绿

- [ ] **Step 5: Commit**

```bash
git add src/paper_repro/models.py tests/test_models.py
git commit -m "feat: typed contracts (Spec/Claim/Grade/...) with cross-ref validation"
```

---

## Task 3: RunDir 目录布局与读写

**Files:**
- Create: `src/paper_repro/rundir.py`
- Test: `tests/test_rundir.py`

- [ ] **Step 1: 写失败测试**

`tests/test_rundir.py`:
```python
from pathlib import Path

from paper_repro.models import IngestInfo, Spec, Artifact, Claim, EvalProtocol
from paper_repro.rundir import RunDir


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

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_rundir.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'paper_repro.rundir'`

- [ ] **Step 3: 写实现**

`src/paper_repro/rundir.py`:
```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_rundir.py -v`
Expected: PASS,6 个测试全绿

- [ ] **Step 5: Commit**

```bash
git add src/paper_repro/rundir.py tests/test_rundir.py
git commit -m "feat: RunDir layout + typed artifact I/O"
```

---

## Task 4: 指标解析器

**Files:**
- Create: `src/paper_repro/parsers.py`
- Test: `tests/test_parsers.py`

run 阶段把评测脚本的原始 stdout 落盘,grade 阶段用这里的解析器把数字抠出来。解析不出来要明确返回 None(让 grade 判 BLOCKED/UNPARSEABLE),绝不猜。

- [ ] **Step 1: 写失败测试**

`tests/test_parsers.py`:
```python
from paper_repro.parsers import parse_metric


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

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_parsers.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'paper_repro.parsers'`

- [ ] **Step 3: 写实现**

`src/paper_repro/parsers.py`:
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

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_parsers.py -v`
Expected: PASS,7 个测试全绿

- [ ] **Step 5: Commit**

```bash
git add src/paper_repro/parsers.py tests/test_parsers.py
git commit -m "feat: metric parsers (ppl/accuracy/speedup) returning None on miss"
```

---

## Task 5: Grade —— 纯代码判分(核心)

**Files:**
- Create: `src/paper_repro/grade.py`
- Test: `tests/test_grade.py`

这是整个系统的皇冠。两道独立检查(数值达标 + 过程忠实),四态判定。grade 只读 spec + run 落盘输出,**不 import runstage,不重跑,不知道目标值之外的执行上下文**。

判定规则(与设计 §5.1 一致):
- **MATCH** = 数值达标 AND 过程忠实
- **PARTIAL** = 数值达标但过程有偏差;或过程忠实但数值超容差(必带原因)
- **FAIL** = 数值显著偏离且无法归因(数值超容差且过程也有偏差)
- **BLOCKED** = run 没跑成 / 输出无法解析 / calib UNKNOWN 导致不可比

- [ ] **Step 1: 写失败测试**

`tests/test_grade.py`:
```python
from paper_repro.models import (
    Artifact, Claim, EvalProtocol, Spec, RunResult, RepoInfo,
)
from paper_repro.grade import grade_claim


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

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_grade.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'paper_repro.grade'`

- [ ] **Step 3: 写实现**

`src/paper_repro/grade.py`:
```python
"""Pure-code judge. Isolated from execution: reads only spec + run's落盘 output.

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

from paper_repro.models import Artifact, Claim, ClaimGrade, RunResult
from paper_repro.parsers import parse_metric

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

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_grade.py -v`
Expected: PASS,7 个测试全绿

- [ ] **Step 5: Commit**

```bash
git add src/paper_repro/grade.py tests/test_grade.py
git commit -m "feat: pure-code judge with value+faithfulness double check, 4 verdicts"
```

---

## Task 6: Report —— 中英双语渲染

**Files:**
- Create: `src/paper_repro/report.py`
- Test: `tests/test_report.py`

渲染 `report.zh.md` 和 `report.en.md`。永远用实测原始数字,绝不用 paper 数字填空;每条 claim 附复算信息。

- [ ] **Step 1: 写失败测试**

`tests/test_report.py`:
```python
from paper_repro.models import (
    Spec, Artifact, Claim, EvalProtocol, RepoInfo, ClaimGrade, RunResult, IngestInfo,
)
from paper_repro.report import render_reports


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

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'paper_repro.report'`

- [ ] **Step 3: 写实现**

`src/paper_repro/report.py`:
```python
"""Render bilingual reproduction reports (zh + en) as two markdown strings.

Iron rules (design §5.2):
  - always show measured (raw) numbers, never paper numbers as substitute
  - every claim carries replay info (command/seed/gpu/commit/env)
"""
from __future__ import annotations

from collections import Counter

from paper_repro.models import ClaimGrade, IngestInfo, RunResult, Spec


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
        measured = "—" if not g or g.measured is None else f"{g.measured:g}"
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

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_report.py -v`
Expected: PASS,4 个测试全绿

- [ ] **Step 5: Commit**

```bash
git add src/paper_repro/report.py tests/test_report.py
git commit -m "feat: bilingual report rendering (measured-only, with replay info)"
```

---

## Task 7: Ingest —— .org 解析与入参归一

**Files:**
- Create: `src/paper_repro/ingest.py`
- Create: `tests/fixtures/sample.org`
- Test: `tests/test_ingest.py`

本期重点实现可纯逻辑测试的部分:入参归一(`.org` / url / id → arxiv_id + source_url)和 repo 链接发现(从 latex/readme 文本里扒 GitHub 链接)。网络抓取(latex/clone)封装成可注入的函数,测试用 monkeypatch 打桩。

- [ ] **Step 1: 写 fixture**

`tests/fixtures/sample.org`:
```
#+title:      Ternary Mamba: Grouped QAT of W1.58A16 SSMs
#+date:       [2026-06-17 Wed 00:17]
#+source:     https://arxiv.org/abs/2606.18114
#+authors:    Alice Smith, Bob Jones

* The Problem River
Some narrative text. Code at https://github.com/example/ternary-mamba here.
```

- [ ] **Step 2: 写失败测试**

`tests/test_ingest.py`:
```python
from pathlib import Path

from paper_repro.ingest import (
    normalize_input, parse_org, find_repo_url, arxiv_id_from_url,
)

FIX = Path(__file__).parent / "fixtures"


def test_parse_org_extracts_source_and_meta():
    meta = parse_org((FIX / "sample.org").read_text())
    assert meta["source"] == "https://arxiv.org/abs/2606.18114"
    assert meta["title"].startswith("Ternary Mamba")
    assert "Alice Smith" in meta["authors"]


def test_arxiv_id_from_abs_url():
    assert arxiv_id_from_url("https://arxiv.org/abs/2606.18114") == "2606.18114"


def test_arxiv_id_from_versioned_url():
    assert arxiv_id_from_url("https://arxiv.org/abs/2401.00001v2") == "2401.00001"


def test_normalize_input_from_org_file(tmp_path):
    f = tmp_path / "x.org"
    f.write_text("#+source: https://arxiv.org/abs/2401.00001\n")
    arxiv_id, url = normalize_input(str(f))
    assert arxiv_id == "2401.00001"
    assert url == "https://arxiv.org/abs/2401.00001"


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

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'paper_repro.ingest'`

- [ ] **Step 4: 写实现**

`src/paper_repro/ingest.py`:
```python
"""Ingest stage: normalize input to (arxiv_id, source_url), discover official repo.

Network fetches (latex tarball, git clone) are isolated behind functions that
callers can patch in tests. The parsing/normalization logic is pure.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")
_GITHUB_RE = re.compile(r"https?://github\.com/[\w.\-]+/[\w.\-]+")


def arxiv_id_from_url(url: str) -> Optional[str]:
    m = _ARXIV_RE.search(url)
    return m.group(1) if m else None


def parse_org(text: str) -> dict:
    meta: dict = {"authors": []}
    for line in text.splitlines():
        m = re.match(r"#\+(\w+):\s*(.*)", line)
        if not m:
            continue
        key, val = m.group(1).lower(), m.group(2).strip()
        if key == "authors":
            meta["authors"] = [a.strip() for a in val.split(",") if a.strip()]
        else:
            meta[key] = val
    return meta


def find_repo_url(text: str) -> Optional[str]:
    m = _GITHUB_RE.search(text)
    if not m:
        return None
    return m.group(0).rstrip("/").removesuffix(".git")


def normalize_input(arg: str) -> tuple[str, str]:
    """Return (arxiv_id, source_url) from a .org path, arxiv url, or bare id."""
    p = Path(arg)
    if p.exists() and p.suffix == ".org":
        meta = parse_org(p.read_text())
        url = meta.get("source", "")
        arxiv_id = arxiv_id_from_url(url)
        if not arxiv_id:
            raise ValueError(f"no arxiv source in {arg}")
        return arxiv_id, url
    if arg.startswith("http"):
        arxiv_id = arxiv_id_from_url(arg)
        if not arxiv_id:
            raise ValueError(f"cannot extract arxiv id from {arg}")
        return arxiv_id, f"https://arxiv.org/abs/{arxiv_id}"
    if _ARXIV_RE.fullmatch(arg):
        return arg, f"https://arxiv.org/abs/{arg}"
    raise ValueError(f"unrecognized input: {arg}")
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: PASS,8 个测试全绿

- [ ] **Step 6: Commit**

```bash
git add src/paper_repro/ingest.py tests/test_ingest.py tests/fixtures/sample.org
git commit -m "feat: ingest input normalization + org parsing + repo discovery"
```

---

## Task 8: Plan —— 可行性/异常哨兵

**Files:**
- Create: `src/paper_repro/planstage.py`
- Test: `tests/test_planstage.py`

plan 默认静默放行(成本非约束)。只在两种情况标 `needs_user_decision=True`:硬件不可行,或估算与 paper 自报严重背离(质量信号,通常意味着 specextract 抽错)。

- [ ] **Step 1: 写失败测试**

`tests/test_planstage.py`:
```python
from paper_repro.models import Spec, Artifact, Claim, EvalProtocol
from paper_repro.planstage import build_plan


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

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_planstage.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'paper_repro.planstage'`

- [ ] **Step 3: 写实现**

`src/paper_repro/planstage.py`:
```python
"""Plan stage: feasibility / anomaly sentinel.

Default: silent pass (compute is not a constraint). Escalates to a user decision
only on (1) infeasible hardware, or (2) estimate wildly diverging from the paper.
"""
from __future__ import annotations

from paper_repro.models import ClaimPlan, PlanReport, Spec


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

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_planstage.py -v`
Expected: PASS,3 个测试全绿

- [ ] **Step 5: Commit**

```bash
git add src/paper_repro/planstage.py tests/test_planstage.py
git commit -m "feat: plan stage feasibility/anomaly sentinel (silent pass by default)"
```

---

## Task 9: Headless —— claude -p 封装

**Files:**
- Create: `src/paper_repro/headless.py`
- Test: `tests/test_headless.py`

复用 llm-paper-radar 的模式:`claude -p --permission-mode acceptEdits --allowedTools "..."`,prompt 经 stdin,**不信 exit code,靠输出文件是否出现来判定成功**。subprocess 调用封装成可注入,测试用 monkeypatch。

- [ ] **Step 1: 写失败测试**

`tests/test_headless.py`:
```python
from pathlib import Path

import paper_repro.headless as headless
from paper_repro.headless import run_headless


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

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_headless.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'paper_repro.headless'`

- [ ] **Step 3: 写实现**

`src/paper_repro/headless.py`:
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

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_headless.py -v`
Expected: PASS,3 个测试全绿

- [ ] **Step 5: Commit**

```bash
git add src/paper_repro/headless.py tests/test_headless.py
git commit -m "feat: headless claude -p wrapper with output-file verification"
```

---

## Task 10: SpecExtract 阶段(headless,可 mock)

**Files:**
- Create: `src/paper_repro/specextract.py`
- Create: `tests/fixtures/extracted_spec.yaml`
- Test: `tests/test_specextract.py`

specextract 构造 prompt、调 headless 生成 `spec.yaml`、校验能 parse 成 `Spec`。本期用 mock headless 写出 fixture spec 来验证装配。门控(等用户审 spec)在 pipeline 层做,这里只负责生成+校验。

- [ ] **Step 1: 写 fixture**

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

- [ ] **Step 2: 写失败测试**

`tests/test_specextract.py`:
```python
import shutil
from pathlib import Path

import paper_repro.specextract as specextract
from paper_repro.headless import HeadlessResult
from paper_repro.rundir import RunDir

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

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_specextract.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'paper_repro.specextract'`

- [ ] **Step 4: 写实现**

`src/paper_repro/specextract.py`:
```python
"""SpecExtract stage: one headless claude call → spec.yaml → validated Spec.

The approval gate (user reviews spec.yaml) is enforced by the pipeline, not here.
"""
from __future__ import annotations

from typing import Optional

import yaml

from paper_repro.headless import run_headless
from paper_repro.models import Spec
from paper_repro.rundir import RunDir

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

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_specextract.py -v`
Expected: PASS,3 个测试全绿

- [ ] **Step 6: Commit**

```bash
git add src/paper_repro/specextract.py tests/test_specextract.py tests/fixtures/extracted_spec.yaml
git commit -m "feat: specextract stage (headless → validated Spec)"
```

---

## Task 11: Setup / Run 阶段接口与 stub

**Files:**
- Create: `src/paper_repro/setupstage.py`
- Create: `src/paper_repro/runstage.py`
- Test: `tests/test_stages_stub.py`

本期不接真 GPU。setup 返回 env 快照(stub),run 按 spec 把"评测命令应有的输出"落盘的能力留给 Plan 2;现在 run 提供一个能消费 RunResult 契约的 stub,并支持注入一个"executor"以便 pipeline 测试。这样两路径(官方 repo / 从头实现)将来都填同一个 executor 接口。

- [ ] **Step 1: 写失败测试**

`tests/test_stages_stub.py`:
```python
from pathlib import Path

from paper_repro.models import Spec, Artifact, Claim, EvalProtocol
from paper_repro.rundir import RunDir
from paper_repro.setupstage import run_setup, SetupResult
from paper_repro.runstage import run_claims


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

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_stages_stub.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'paper_repro.setupstage'`

- [ ] **Step 3: 写实现 setupstage.py**

`src/paper_repro/setupstage.py`:
```python
"""Setup stage: the (future) agentic env-debug loop.

Plan 1 provides a stub returning a placeholder env snapshot. Plan 2 will replace
the body with a bounded headless claude loop that builds a conda/uv env and runs
the repo's smoke test until it passes once. The signature stays stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from paper_repro.models import Spec
from paper_repro.rundir import RunDir


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
    (rd.root / "env_snapshot.json").write_text(
        __import__("json").dumps(snapshot, indent=2))
    return SetupResult(ok=True, env_snapshot=snapshot, patches=[])
```

- [ ] **Step 4: 写实现 runstage.py**

`src/paper_repro/runstage.py`:
```python
"""Run stage: deterministic execution of quant + eval per claim.

The actual quantization/eval is delegated to an `executor` callable (one impl per
provider: official-repo now, from-scratch later). Plan 1 wires the contract and
the落盘/blocked-handling; Plan 2 supplies the real GPU executor.

executor(claim, artifact, claim_dir) -> dict with keys:
  stdout_path, actual_config, gpu, seed, minutes
"""
from __future__ import annotations

from typing import Callable

from paper_repro.models import RunResult, Spec
from paper_repro.rundir import RunDir


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

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_stages_stub.py -v`
Expected: PASS,3 个测试全绿

- [ ] **Step 6: Commit**

```bash
git add src/paper_repro/setupstage.py src/paper_repro/runstage.py tests/test_stages_stub.py
git commit -m "feat: setup/run stage interfaces with injectable executor (stubs)"
```

---

## Task 12: Pipeline 编排 + 门控

**Files:**
- Create: `src/paper_repro/pipeline.py`
- Test: `tests/test_pipeline.py`

把各阶段串起来。门控 1(spec 审批)和 plan 哨兵通过注入的回调实现,这样测试里可以自动批准,真实 CLI 里弹 AskUserQuestion / 等用户。grade 用 Task 5 的 `grade_claim` 逐条判,report 用 Task 6 渲染。

- [ ] **Step 1: 写失败测试**

`tests/test_pipeline.py`:
```python
import shutil
from pathlib import Path

import paper_repro.pipeline as pipeline
import paper_repro.specextract as specextract
from paper_repro.headless import HeadlessResult
from paper_repro.setupstage import SetupResult

FIX = Path(__file__).parent / "fixtures"


def _fake_specextract(rd):
    shutil.copy(FIX / "extracted_spec.yaml", rd.root / "spec.yaml")
    from paper_repro.models import Spec
    import yaml
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

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'paper_repro.pipeline'`

- [ ] **Step 3: 写实现**

`src/paper_repro/pipeline.py`:
```python
"""Deterministic orchestration of the 7 stages with two gates.

Gates and side-effecting stages are injected as callables so tests run offline
and the CLI can supply interactive prompts / real executors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from paper_repro.grade import grade_claim
from paper_repro.ingest import normalize_input
from paper_repro.models import IngestInfo, RepoInfo
from paper_repro.planstage import build_plan
from paper_repro.report import render_reports
from paper_repro.rundir import RunDir
from paper_repro.runstage import run_claims
from paper_repro.setupstage import run_setup
from paper_repro.specextract import extract_spec


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
    rd.write_ingest(IngestInfo(arxiv_id=arxiv_id, source_url=url))

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
    grades = [grade_claim(c, artifacts[c.artifact],
                          next(r for r in runs if r.claim_id == c.id),
                          actual_configs.get(c.id, {}))
              for c in spec.claims]

    # --- report ---
    ingest = rd.read_ingest()
    ingest.repo = spec.repo
    zh, en = render_reports(spec, ingest, grades, runs, setup.env_snapshot,
                            patches=setup.patches)
    (rd.root / "report.zh.md").write_text(zh)
    (rd.root / "report.en.md").write_text(en)
    return PipelineResult(root=rd.root, aborted_at=None)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS,2 个测试全绿

- [ ] **Step 5: Commit**

```bash
git add src/paper_repro/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestration with spec-approval + plan gates"
```

---

## Task 13: CLI

**Files:**
- Create: `src/paper_repro/cli.py`
- Modify: `pyproject.toml`(加 `[project.scripts]`)
- Test: `tests/test_cli.py`

`run` 命令把真实依赖(网络抓取、交互门控、真实 executor)接进 pipeline。本期网络/GPU executor 仍是占位(Plan 2 替换),但 CLI 装配与 `--yes` 自动批准、`report` 重渲染可测。

- [ ] **Step 1: 写失败测试**

`tests/test_cli.py`:
```python
from pathlib import Path

from click.testing import CliRunner

from paper_repro.cli import cli


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
    from paper_repro.rundir import RunDir
    from paper_repro.models import (Spec, Artifact, Claim, EvalProtocol, RepoInfo,
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

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'paper_repro.cli'`

- [ ] **Step 3: 写实现**

`src/paper_repro/cli.py`:
```python
"""paper-repro CLI: run / resume / report."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import click

from paper_repro.grade import grade_claim
from paper_repro.parsers import parse_metric
from paper_repro.report import render_reports
from paper_repro.rundir import RunDir


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
    """Run the reproduction pipeline for a paper (arxiv id / url / .org)."""
    from paper_repro.pipeline import run_pipeline

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
    from paper_repro.models import RunResult
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

- [ ] **Step 4: 加 console script 入口**

在 `pyproject.toml` 的 `[project]` 段之后插入:
```toml
[project.scripts]
paper-repro = "paper_repro.cli:cli"
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS,3 个测试全绿

- [ ] **Step 6: 全量测试 + 装 CLI 冒烟**

Run: `uv run pytest -q && uv sync && uv run paper-repro --help`
Expected: 全部测试通过;`paper-repro --help` 打印 run/report 子命令

- [ ] **Step 7: Commit**

```bash
git add src/paper_repro/cli.py pyproject.toml tests/test_cli.py uv.lock
git commit -m "feat: click CLI (run/report) with console script entrypoint"
```

---

## Task 14: 端到端冒烟 + conftest 清理

**Files:**
- Create: `tests/conftest.py`
- Test: 复用现有

- [ ] **Step 1: 写共享 fixture(去重)**

`tests/conftest.py`:
```python
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
```

- [ ] **Step 2: 跑全量测试**

Run: `uv run pytest -q`
Expected: PASS,所有 task 的测试全绿(约 50+ 用例)

- [ ] **Step 3: ruff 检查**

Run: `uv run ruff check src/ tests/`
Expected: 无 error(有自动可修的 warning 用 `uv run ruff check --fix` 修)

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: shared fixtures + full-suite green"
```

---

## Self-Review

**1. Spec coverage** — 逐条对设计:

- §2 七阶段:ingest(Task 7)/ specextract(Task 10)/ plan(Task 8)/ setup(Task 11)/ run(Task 11)/ grade(Task 5)/ report(Task 6)/ 编排(Task 12)。✓
- §2.1 门控 1 + plan 哨兵:pipeline 的 `approve_spec` / `approve_plan` + `build_plan.needs_user_decision`(Task 8/12)。✓
- §2.2 判分隔离:`grade.py` 不 import `runstage`,只读落盘文件(Task 5)。✓
- §3.1 ingest 入参三形态 + repo 发现:`normalize_input` / `find_repo_url`(Task 7)。latex 抓取/clone 明确推迟到 Plan 2(`fetch_sources` 占位),不是遗漏。✓
- §3.2 spec schema 两层:`models.py` 的 Artifact/Claim(Task 2)。✓
- §3.3 runner / calib_status / source:都是模型字段并在 grade/specextract 用到。✓
- §4.1 setup 单一退出条件 + 护栏:接口+stub(Task 11),真实循环推迟 Plan 2(已在文件 docstring 标注)。✓
- §4.2 run 优先官方复现命令:executor 接口承载,Plan 2 实现。✓
- §5.1 四态判分:Task 5 完整覆盖,含 BLOCKED 与 FAIL 分离、calib UNKNOWN、unparseable。✓
- §5.2 双语两文件 + 实测数字 + 复算信息:Task 6。✓
- §6 从头实现预留接口:`run_setup`/`run_claims` 的 `executor` 注入点即 provider 接口。✓
- §7 run 目录布局:Task 3 RunDir。✓
- §8 YAGNI:无队列/DB/cron/成本门控/LLM 判分。✓

**推迟到 Plan 2 的(非遗漏,显式划出范围):** 真实 latex 抓取与 git clone、真实 conda/uv setup 调试循环、真实 GPU 量化+评测 executor、setup_patches 实采集、env_snapshot 实采集。Plan 1 全部用接口/stub 占位并测试装配。

**2. Placeholder scan** — 计划内无 "TBD/TODO/实现细节后补";代码步骤均给出完整可运行代码。stub 是有意的范围切分,已在 docstring 和 self-review 标注,非占位符欠债。

**3. Type consistency** — 通查:`grade_claim(claim, artifact, run, actual_config)` 在 Task 5 定义、Task 12 调用一致;`run_claims(rd, spec, executor) -> (list[RunResult], dict)` Task 11 定义、Task 12 解包一致;`render_reports(spec, ingest, grades, runs, env, patches)` Task 6 定义、Task 12/13 调用一致;`run_headless(prompt, allowed_tools, cwd, expect_file)` Task 9 定义、Task 10 调用一致;`RunResult.stdout_path` 为 str,grade 用 `Path(run.stdout_path)` 包装一致。✓
