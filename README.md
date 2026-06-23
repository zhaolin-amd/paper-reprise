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
  of claims (model × config × eval protocol × expected number); then presents the claims
  for you to pick which to reproduce.
- **plan** — feasibility check; flags claims whose required hardware isn't available.
- **setup** — the one agentic stage: build a conda/uv env and let Claude fix dependencies
  until the repo's own eval command passes a smoke test, under retry/timeout guardrails.
- **run** — execute each claim's eval command in that env, persist the raw output.
- **grade** — pure code, isolated from execution: a claim is MATCH only if the number is
  in tolerance AND the run was faithful; otherwise PARTIAL / FAIL / BLOCKED.
- **report** — bilingual `report.zh.md` / `report.en.md`, always the measured number,
  never the paper's, with full replay info.

### One isolated reproduction path per paper

Every paper gets its own self-contained run directory under `runs/`, and the whole
reproduction happens inside it — nothing is installed globally and nothing leaks between
papers:

- the official repo is **cloned into that run dir** (`repo/`), not shared;
- a **dedicated virtualenv** is built for that paper (`env/`), so each paper pins its own
  (often mutually incompatible) torch / transformers / CUDA stack;
- the eval scripts that actually run are the **paper's own** scripts shipped in its repo
  (e.g. `repo/scripts/eval_model.sh`) — paper-reprise invokes them, it does not reimplement
  them;
- all raw outputs, logs, env snapshot, and reports stay under that one directory.

To reproduce N papers you get N independent run dirs; deleting a run dir removes everything
that run touched. The entire `runs/` tree is gitignored.

## Run directory layout

One run = one directory `runs/<paper-name>-<arxiv_id>-<timestamp>/` (the `<paper-name>`
slug is best-effort from the arxiv title; omitted when it can't be fetched):

```
runs/<paper-name>-<arxiv_id>-<timestamp>/
├── ingest.json          # resolved arxiv id + source url (the located repo url/commit are
│                        #   recorded in spec.yaml by specextract, not here)
├── paper/               # the paper's LaTeX source, downloaded from arxiv (not OCR'd)
├── repo/                # the official repo, git-cloned here — its own eval scripts live inside
├── spec.yaml            # extracted reproduction spec: artifacts × claims × eval protocol
│                        #   (the interactive claim picker writes the chosen subset here; `resume` re-reads it)
├── spec.public.yaml     # from-scratch path only: redacted spec the implementer agent reads
│                        #   (expected/tolerance/source stripped so it can't target the number)
├── plan.json            # per-claim feasibility estimate (hardware required vs available)
├── env/                 # the dedicated conda/uv virtualenv built for this paper
├── env_snapshot.json    # frozen torch/transformers/CUDA + pip freeze (on a successful setup)
├── setup_log/           # logs from the setup loop: create_env.log, smoke_<n>.log per attempt
├── setup_patches/       # patch_<n>.txt — one line per change the setup agent made to the env/repo
├── runs/                # per-claim execution outputs (one subdir per claim id):
│   └── <claim_id>/
│       ├── stdout.log         # raw combined stdout+stderr of that claim's eval run
│       └── actual_config.json # the config the eval was launched with (for the faithfulness check)
├── report.zh.md         # Chinese reproduction report — the per-claim verdict table
│                        #   (MATCH / PARTIAL / FAIL / BLOCKED + reason), measured numbers, replay info
└── report.en.md         # English reproduction report (same content)
```

Notes:

- `paper/` and `repo/` are created for every run but stay empty if the source fetch is
  skipped or no official repo is found.
- `env_snapshot.json` is written only when setup succeeds; on failure the diagnosis is in
  `setup_log/` and the run stage is reported as BLOCKED.
- Verdicts are not persisted as a separate file — they are computed in `grade` (pure code)
  and rendered straight into the two reports; `paper-reprise report <run_dir>` re-renders
  them from `spec.yaml` + the per-claim `stdout.log` / `actual_config.json`.

## Usage

```
paper-reprise run <arxiv_id | arxiv_url | "paper title">
paper-reprise resume <run_dir>          # continue an existing run from its spec.yaml
paper-reprise report <run_dir>          # re-render the report from an existing run
```

A short alias `reprise` is also installed (e.g. `reprise run 2401.00001`).

**By default `run` presents the extracted claims interactively** — after specextract it
prints each numbered claim as a block showing the full **model × config** (model, the whole
quant config, eval protocol, target ± tolerance, hardware) and asks which to reproduce
(`"1 3"`, `"all"`, or `"q"` to abort). The spec is LLM-extracted from the
paper and may mis-pick or mis-read (e.g. required hardware), so this is your review gate; the
chosen subset is kept (orphaned artifacts pruned) and the pipeline continues automatically.
Pass `--yes` to skip selection and reproduce **all** claims end to end (non-interactive / CI).

For finer control you can still hand-edit `spec.yaml` in the run dir and `resume <run_dir>`
— `resume` runs whatever claims the file contains, without re-prompting.

**Reproducing efficiency/accuracy at scale needs a GPU** — quantization papers run on real
models. Without a GPU the pipeline still runs through setup and reports the run stage as
BLOCKED rather than fabricating numbers.

## Status

The full pipeline is implemented and runs end to end, both for papers with an official repo
and for papers **without** one — the from-scratch provider implements the paper's method as a
self-contained `impl/` (headless Claude, behind the same env-build + smoke-test + retry/timeout
guardrails as the official path) and runs that, with the grade stage isolated from it.

Not yet implemented:

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
