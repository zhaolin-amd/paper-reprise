# paper-reprise

An agent that reproduces quantization paper results. Given an arxiv paper, it prefers to
reproduce the paper's reported numbers by invoking the official repo's own scripts, and
honestly reports the gap.

Design doc: [docs/superpowers/specs/2026-06-19-paper-reprise-agent-design.md](docs/superpowers/specs/2026-06-19-paper-reprise-agent-design.md)

## Status

Deterministic skeleton implemented (Plan 1). The real GPU quantization/eval executor,
latex fetch + git clone, and the agentic conda/uv setup-debug loop are deferred to Plan 2
(wired as injectable executor seams).

## Usage

```
paper-reprise run <arxiv_id | arxiv_url>
paper-reprise resume <run_dir>
paper-reprise report <run_dir>
```

A short alias `reprise` is also installed (e.g. `reprise run 2401.00001`).

Deterministic pipeline `ingest → specextract → plan → setup → run → grade → report`; the
agent only enters the setup stage (taming the rotting official-repo environment); grading
is pure code, isolated from execution.

## Develop

```
uv sync
uv run pytest -q
uv run ruff check src/ tests/
```
