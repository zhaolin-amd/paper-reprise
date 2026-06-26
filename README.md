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
  of claims (model × config × eval protocol × expected number) — including each model's
  uncompressed **FP baseline**, so the measured baseline can be checked against the paper's
  (a baseline that matches confirms the eval protocol; one that doesn't flags a mismatch
  before trusting any quantized gap), plus any **prerequisite-method repos** the paper builds
  on (recorded as read-only references for the from-scratch path); then presents the claims
  for you to pick.
- **plan** — feasibility check; flags claims whose required hardware isn't available.
- **setup** — the one agentic stage: build a conda/uv env and let Claude fix dependencies
  until the repo's own eval command passes a smoke test, under retry/timeout guardrails.
- **run** — execute each claim's eval command in that env, persist the raw output.
- **grade** — pure code, isolated from execution: a claim is MATCH only if the number is
  in tolerance AND the run was faithful; otherwise PARTIAL / FAIL / BLOCKED.
- **report** — bilingual `report.zh.md` / `report.en.md`, always the measured number,
  never the paper's, with the harness's raw per-task results table behind any averaged
  metric and full replay info.

Papers **without** an official repo take a *from-scratch* path: instead of running a cloned
repo, headless Claude implements the paper's method as a self-contained `impl/` (from a
redacted spec, so it never sees the target number), which then flows through the same setup →
run → grade → report stages and guardrails. When the spec records prerequisite-method repos
(a paper that builds on a prior method with its own repo), they are offered to the
implementer as **read-only references** — the paper stays the source of truth and they are
never a way to back into a number.

### One isolated reproduction path per paper

Every paper gets its own self-contained run directory under `runs/`, and the whole
reproduction happens inside it — nothing is installed globally and nothing leaks between
papers:

- the official repo is **cloned into that run dir** (`repo/`), not shared;
- a **dedicated virtualenv** is built for that paper (`env/`), so each paper pins its own
  (often mutually incompatible) torch / transformers / CUDA-or-ROCm stack;
- the eval scripts that actually run are the **paper's own** scripts shipped in its repo
  (e.g. `repo/scripts/eval_model.sh`) — paper-reprise invokes them, it does not reimplement
  them;
- all raw outputs, logs, env snapshot, and reports stay under that one directory.

To reproduce N papers you get N independent run dirs; deleting a run dir removes everything
that run touched. The entire `runs/` tree is gitignored.

## Run directory layout

One run = one directory `runs/<paper-name>-<arxiv_id>-<timestamp>/` (the `<paper-name>`
slug is best-effort from the arxiv title — the authors' short name before a colon when the
title has one, e.g. `turboquant` from "TurboQuant: …"; omitted when the title can't be
fetched):

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
├── env_snapshot.json    # frozen torch/transformers + CUDA or ROCm + pip freeze (on a successful setup)
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

## Install

Prerequisites: **Python ≥ 3.11**, **git**, **uv** (or conda), and the **`claude` CLI** on
PATH — specextract and the setup loop run headless Claude (`claude -p`). Reproducing real
accuracy/efficiency numbers also needs a **GPU** — NVIDIA (CUDA) or AMD Instinct (ROCm);
the agent detects both. paper-reprise is vendor-agnostic (it builds an env and runs the
paper's own scripts), so whether a *given* paper reproduces on a given vendor depends on
that paper's repo, not on the tool. Without a GPU the run still proceeds through setup and
reports the run stage as BLOCKED rather than fabricating numbers.

From a clone (local / dev):

```
git clone https://github.com/zhaolin-amd/paper-reprise && cd paper-reprise
uv sync                                  # builds ./.venv from uv.lock
uv run paper-reprise run 2604.18556      # or: uv run reprise run 2604.18556
```

Or put the `paper-reprise` / `reprise` commands on your PATH:

```
uv tool install .                        # or: pipx install .   |   pip install -e .
paper-reprise run 2604.18556
```

## Usage

```
paper-reprise run <arxiv_id | arxiv_url | "paper title">   # full reproduction: select → setup → quantize → eval → grade → report (does NOT clean; that's separate)
paper-reprise resume <run_dir>          # continue an existing run from its spec.yaml
paper-reprise report <run_dir>          # re-render the report from an existing run
paper-reprise clean  [<run_dir>]        # free model weights + env (keep records); no arg → all runs under runs/
```

A short alias `reprise` is also installed (e.g. `reprise run 2604.18556`).

**By default `run` presents the extracted claims interactively** — after specextract it
prints each numbered claim as a block showing the full **model × config** (model, the whole
quant config, eval protocol, target ± tolerance, hardware) and asks which to reproduce
(`"1 3"`, `"all"`, or `"q"` to abort). The spec is LLM-extracted from the
paper and may mis-pick or mis-read (e.g. required hardware), so this is your review gate; the
chosen subset is kept (orphaned artifacts pruned) and the pipeline continues automatically.
Pass `--yes` to skip selection and reproduce **all** claims end to end (non-interactive / CI).

For finer control you can still hand-edit `spec.yaml` in the run dir and `resume <run_dir>`
— `resume` runs whatever claims the file contains, without re-prompting.

`run`/`resume` also accept `--tasks a,b,c` to override the eval task list and `--gpus N` to
override the GPU count: they export `PAPER_REPRISE_TASKS` / `PAPER_REPRISE_GPUS`, which eval
commands read as `${PAPER_REPRISE_TASKS:-<spec default>}` / `${PAPER_REPRISE_GPUS:-<default>}`,
so your value wins when given and the paper's default applies otherwise. (`--gpus` sets how
many GPUs; *which* ones is still `CUDA_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`.)

**Where the model is written.** A paper's repo can emit a multi-GB quantized checkpoint. To
keep `runs/` (home, small quota) light, setup symlinks `repo/runtime` (override:
`PAPER_REPRISE_REPO_OUTPUT_SUBDIR`) to a per-run scratch dir under
`/scratch/$USER/paper-reprise-models/` (override: `PAPER_REPRISE_MODELS_DIR`) — the repo
writes there transparently, no command changes; paper-reprise's own records stay in the run
dir. (Skipped if `repo/runtime` already exists as a real directory, to avoid clobbering.)

**Freeing disk.** `run`/`resume` **keep** the exported model and env by default, so you can run
a dir many times (resume more claims, re-eval) without re-quantizing. When you're done with a
run, `paper-reprise clean <run_dir>` (or just `paper-reprise clean` to sweep every run under
`runs/`) deletes the regenerable artifacts — the exported model weights (including the
symlinked scratch copy) and (unless `--no-env`) the per-run env — while keeping **all records** (logs, env
snapshot, setup patches, per-claim `stdout.log` / `actual_config`, reports). For a one-shot
run you can instead pass `run/resume --clean-models` to delete the model right after that run
(skipped if nothing verified, so a failed run's model stays for debugging).

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
