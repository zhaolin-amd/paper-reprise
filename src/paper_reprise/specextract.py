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
- references: list of {{method, repo_url, note}} or [] (optional)

Rules:
- runner: prefer "official" (the repo's own eval script). Use "cited-standard" if the \
paper explicitly cites a standard impl; "custom" only as last resort.
- command: when it runs an lm-eval-style task suite, write the task list as \
`${{PAPER_REPRISE_TASKS:-<the paper's tasks>}}` (e.g. \
`EVAL_TASKS=${{PAPER_REPRISE_TASKS:-arc_easy,hellaswag}}`) so a `--tasks` override can \
replace it while the paper's tasks stay the default. Likewise, when the command takes a \
GPU/process count, write it as `${{PAPER_REPRISE_GPUS:-<default>}}` (e.g. \
`NPROC=${{PAPER_REPRISE_GPUS:-1}}` or `torchrun --nproc-per-node=${{PAPER_REPRISE_GPUS:-1}}`) \
so a `--gpus` override applies. Reference the model as `$PAPER_REPRISE_MODEL` (it is \
exported into the command).
- baseline: for each model that has a quantized claim, ALSO add the paper's \
UNCOMPRESSED (FP16/BF16) baseline as its own artifact + claim. The baseline artifact \
uses method `none` and `quant_config: {{wbits: 16}}`; the claim id ends in `-baseline` \
(or `-fp16`), reuses the SAME eval_protocol metric/dataset/tasks as the quantized claim, \
and `expected` = the paper's reported baseline number for that metric. Its command \
EVALUATES THE BASE MODEL DIRECTLY (serve `$PAPER_REPRISE_MODEL` + eval) with NO \
quantization step. Rationale: a baseline that matches the paper's baseline confirms our \
eval protocol matches theirs, so the quantized gap is trustworthy; a baseline that does \
NOT match flags an eval-protocol mismatch before we trust any quantized number.
- references: the PREREQUISITE method(s) THIS paper's algorithm directly builds on / \
extends (e.g. a paper proposing "X + a 1-bit QJL residual" depends on QJL), each with \
its official repo_url when the paper states it OR you are reasonably confident it exists, \
and a short `note` on what part to look at. Include ONLY methods the current algorithm \
depends on for IMPLEMENTATION — not every citation. Omit a repo_url you are unsure of \
(better empty than a guessed URL). These are READ-ONLY disambiguation aids for a \
from-scratch reimplementation; the paper remains the source of truth and they must NOT \
be used to obtain any reported number. Use [] if none apply.
- quant_config: use the key `wbits` (not `bits`) for the weight bit-width.
- Default tolerance: perplexity 0.05, accuracy 0.5. If the paper states one, use it.
- source: pin each claim to its location, e.g. "Table 3, row 2, col W4".
- hardware: the MINIMUM hardware needed to reproduce THIS specific claim's model, \
taken from the official repo's README/docs (look for a per-model hardware table or \
requirements section) — NOT the larger setup the paper happened to report its runs on. \
Set it PER CLAIM and per model size: a small model usually needs far less than the \
paper's headline run (e.g. a repo table may list 1x H100/A100 for an 8B model even \
when the paper ran everything on 8x H200). Record whatever accelerator the source \
names — NVIDIA (e.g. H100, H200, A100) or AMD Instinct (e.g. MI300X, MI325X, MI350X, \
MI355X) — as written. When the repo gives a choice or range (e.g. "H100/H200"), pick \
the smaller / more widely available option. Use null only when neither the repo nor \
the paper indicates any specific hardware.

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
