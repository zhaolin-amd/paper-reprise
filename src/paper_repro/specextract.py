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
