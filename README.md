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
├── ingest.json          # resolved arxiv id, source url, located repo (url + commit)
├── paper/               # the paper's LaTeX source, downloaded from arxiv (not OCR'd)
├── repo/                # the official repo, git-cloned here — its own eval scripts live inside
├── spec.yaml            # extracted reproduction spec: artifacts × claims × eval protocol
│                        #   (this is the human-review / hand-edit gate; `resume` re-reads it)
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

**By default `run` stops after extracting `spec.yaml`** — so you can review and edit
**which models/claims** to reproduce (the spec is LLM-extracted from the paper and may
mis-pick or mis-read, e.g. the required hardware). Edit the file, then `resume <run_dir>`
to continue. Pass `--yes` to skip the review and run end to end (non-interactive / CI).

**Reproducing efficiency/accuracy at scale needs a GPU** — quantization papers run on real
models. Without a GPU the pipeline still runs through setup and reports the run stage as
BLOCKED rather than fabricating numbers.

### Model cache

Models are read from a **shared cache first** and anything missing is **downloaded to scratch**
(never $HOME, never the read-only shared cache). Two env knobs control the paths:

| Variable | Default | Meaning |
|---|---|---|
| `PAPER_REPRISE_MODEL_BASE` | `/group/amdneuralopt/huggingface/pretrained_models` | Shared cache root, `<org>/<model>` snapshot layout. A model id (`meta-llama/Llama-3.2-1B`) resolves to its local snapshot here when `config.json` exists — avoiding re-download / re-auth for gated models. |
| `PAPER_REPRISE_DOWNLOAD_DIR` | `/scratch/$USER/pretrained_models` | Where HF downloads missing models (`HF_HUB_CACHE`). |

The defaults are site-specific; override either var to run elsewhere. An eval command may use a
`{model}` placeholder (substituted with the resolved path) and may reference `$PAPER_REPRISE_MODEL`
(exported into the eval shell, e.g. spec `extra_args: GSQ_MODEL_NAME=$PAPER_REPRISE_MODEL`).

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
