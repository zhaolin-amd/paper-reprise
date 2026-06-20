# paper-reprise

An agent that reproduces quantization paper results. Given an arxiv paper, it prefers to
reproduce the paper's reported numbers by invoking the official repo's own scripts, and
honestly reports the gap.

Design doc: [docs/superpowers/specs/2026-06-19-paper-reprise-agent-design.md](docs/superpowers/specs/2026-06-19-paper-reprise-agent-design.md)

## How it works

A deterministic pipeline, one run directory per paper:

```
ingest → specextract → plan → setup → run → grade → report
```

- **ingest** — fetch the arxiv LaTeX source, locate and clone the official repo.
- **specextract** — read the paper + repo (headless Claude) into a machine-checkable spec
  of claims (model × config × eval protocol × expected number); stops for your approval.
- **plan** — feasibility check; flags claims whose required hardware isn't available.
- **setup** — the one agentic stage: build a conda/uv env and let Claude fix dependencies
  until the repo's own eval command passes a smoke test, under retry/timeout guardrails.
- **run** — execute each claim's eval command in that env, persist the raw output.
- **grade** — pure code, isolated from execution: a claim is MATCH only if the number is
  in tolerance AND the run was faithful; otherwise PARTIAL / FAIL / BLOCKED.
- **report** — bilingual `report.zh.md` / `report.en.md`, always the measured number,
  never the paper's, with full replay info.

## Usage

```
paper-reprise run <arxiv_id | arxiv_url | "paper title">
paper-reprise resume <run_dir>          # continue an existing run from its spec.yaml
paper-reprise report <run_dir>          # re-render the report from an existing run
```

A short alias `reprise` is also installed (e.g. `reprise run 2401.00001`).

Add `--yes` to auto-approve the gates non-interactively.

**Reproducing efficiency/accuracy at scale needs a GPU** — quantization papers run on real
models. Without a GPU the pipeline still runs through setup and reports the run stage as
BLOCKED rather than fabricating numbers.

## Status

The full pipeline is implemented and runs end to end for papers with an official repo.

Not yet implemented:

- Papers **without** an official repo (the from-scratch provider) — currently skipped.
- Faithfulness from **parsed** eval output — `actual_config` is the spec-resolved config the
  run was launched with, not values introspected from the eval log, so on the official path
  the faithfulness check leans on calibration-unknown and the setup-patch trail.
- Sandboxing of the executed repo code (it runs on the host).

## Develop

```
uv sync
uv run pytest -q
uv run ruff check src/ tests/
```
