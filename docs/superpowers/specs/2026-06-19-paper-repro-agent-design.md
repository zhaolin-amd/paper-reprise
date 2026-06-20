# Quantization Paper Reproduction Agent — Design Doc

> Date: 2026-06-19
> Status: design approved, implementation plan pending
> Upstream: [llm-paper-radar](https://github.com/zhaolin-amd/llm-paper-radar) (consumes its pushed papers)

## 1. Goal and Scope

Build an agent that, given a quantization paper on arxiv or its official GitHub repo, **reproduces the results reported in the paper**. Papers come mainly from llm-paper-radar's pushes.

- Papers with an official repo: reproduce by invoking the repo's own scripts (highest fidelity, preferred).
- Papers without an official repo: implement the algorithm yourself from the paper's description (**this phase only reserves the interface, no implementation**).

### 1.1 Core Insight

The hard part is not "running code" but two things:
1. Translating a paper into a **machine-executable, auto-gradeable reproduction spec**;
2. **Honestly reporting the gap** — if it can't run, say so; never fill in paper numbers as a substitute.

Running the code itself is already highly standardized in the quantization field (PPL + lm-eval-harness + the paper's own scripts), so we can lean on the existing ecosystem.

### 1.2 Locked Design Decisions

| Dimension | Decision |
|---|---|
| Eval | Run the paper's own scripts first (official > cited-standard > custom rebuild) |
| Spec | Extract the **full eval protocol** per claim |
| Path | Official-repo path first; from-scratch is a **reserved interface** |
| Judge | **Process-faithful AND value-in-tolerance** — both required for MATCH |
| Autonomy | Semi-auto: gate 1 (spec approval) + plan anomaly sentinel |
| Deployment | Manual CLI, per-paper, file-based state |
| Tech stack | Claude Code headless (setup debugging) + conda/uv isolation |
| Compute | Multi-GPU available, cost not a constraint |
| Default claims | Extract main results only (the paper's headline/bolded ones), rest on demand |
| Default tolerance | PPL ±0.05, accuracy ±0.5% (use the paper's if it states one) |
| Report | Bilingual zh/en, split into two files `report.zh.md` / `report.en.md` |

### 1.3 Architecture Choice

Adopt a **deterministic pipeline where the agent only enters the Setup stage** (Approach B). Reasoning: the four hardest constraints — faithful+numeric judge, semi-auto gating, reproducibility, reserved from-scratch interface — all point to a deterministic skeleton that confines the agent's nondeterminism to the one genuinely open-ended step (taming the rotting official-repo environment). Grading is pure-code deterministic numeric comparison, not handed to an LLM.

## 2. Overall Architecture

One CLI invocation = one paper = one run directory. No queue, no DB, no cron.

```
paper-repro run <arxiv_id | path-to-.org | arxiv_url>
paper-repro resume <run_dir>      # resume from last interruption / gate
paper-repro report <run_dir>      # re-render the report
```

Seven deterministic stages, orchestrated in Python; each stage reads the prior stage's artifact and writes its own artifact into the run directory:

```
ingest → specextract → plan → setup → run → grade → report
                ⤷[gate 1: spec approval]   ⤷[plan: feasibility/anomaly sentinel]
```

| Stage | Nature | Responsibility | Output |
|---|---|---|---|
| **ingest** | deterministic | arxiv id → fetch LaTeX source + locate official repo; if input is a `.org`, read `#+source:` | `paper/`, `repo/`, `ingest.json` |
| **specextract** | 1 headless call | LaTeX+README → full spec → **stop, await approval** | `spec.yaml` |
| **plan** | deterministic | estimate each claim's GPU/VRAM/runtime → feasibility/anomaly check | `plan.json` |
| **setup** | agentic debug loop | build conda/uv env, fix deps until the repo's own eval command passes a smoke test | `env/`, `setup_log/`, `env_snapshot.json`, `setup_patches/` |
| **run** | deterministic | quantize per artifact, invoke eval script per claim, persist raw output | `runs/<claim_id>/` |
| **grade** | pure code | parse output, value+faithfulness double check, verdict MATCH/PARTIAL/FAIL/BLOCKED | `grades.json` |
| **report** | deterministic | render bilingual reports | `report.zh.md`, `report.en.md` |

### 2.1 Gates

- **Gate 1 (spec approval):** stop after specextract; the user reviews `spec.yaml` before continuing. Prevents a mis-extracted protocol from wasting the whole downstream run.
- **plan feasibility/anomaly sentinel:** silent pass by default (cost is not a constraint). Escalates to a single `AskUserQuestion` only in two cases:
  1. **Infeasible hardware** — a claim needs a GPU type/VRAM the environment simply doesn't have;
  2. **Estimate wildly diverges from the paper** — the plan estimate far exceeds the paper's self-reported cost (e.g. paper says 4 GPU-hours, estimate says 200), which usually means specextract got something wrong; a quality signal worth a glance before burning resources.

### 2.2 Core Isolation Principle

grade is pure code, separated from execution; it reads only the raw output persisted by the run stage and **never sees execution context beyond "the value to match"**. This is the precondition that makes "process-faithful + value-in-tolerance" grading sound and impossible for the agent to game.

## 3. Ingest and Spec Schema

### 3.1 Ingest

Normalize three input forms to an arxiv_id:
- radar's `.org` file → read `#+source:` to get the arxiv url (radar has already filtered)
- arxiv url / id → use directly

Then:
- **Fetch LaTeX source** (`arxiv.org/e-print/<id>`), do not OCR the PDF — table numbers are far more accurate from LaTeX.
- **Locate the official repo**, priority: GitHub link in the paper > PapersWithCode > GH code search (by title/method name). Candidates plus confidence are written into `ingest.json`; **if none found, `repo: null`** (future from-scratch provider; this phase simply SKIPs and notes it in the report).

```
ingest.json:
  arxiv_id, title, authors, source_url
  repo: {url, commit, confidence, evidence} | null
  latex_path, repo_path
```

### 3.2 Spec Schema (two layers: artifact + claim)

The same quantized product is often reported under multiple eval protocols, so split into **artifacts** (quantized products, reusable) and **claims** (one number = artifact × eval protocol).

```yaml
paper: 2401.xxxxx
repo: {url, commit}                    # carried from ingest, recorded in report by grade

artifacts:                             # quantized products
  - id: llama2-7b-w4g128
    base_model: meta-llama/Llama-2-7b-hf
    method: AWQ
    quant_config:                      # exactly what faithfulness grading compares item by item
      wbits: 4
      group_size: 128
      sym: false
      calib: {dataset: pile, n_samples: 128, seqlen: 512}
    calib_status: known                # known | UNKNOWN (mark explicitly if unextractable; grade treats as "incomparable")

claims:                                # one = one grading unit
  - id: c1
    artifact: llama2-7b-w4g128
    eval_protocol:                     # fully extracted, the judge's basis
      runner: official                 # official | cited-standard | custom
      command: "python eval_ppl.py --model {model} --dataset wikitext2"
      metric: perplexity
      dataset: wikitext2
      split: test
      seqlen: 2048
      stride: 2048
      few_shot: 0
      extra_args: "--use_cache false"
    expected: 5.78
    tolerance: 0.05                    # default PPL ±0.05 / acc ±0.5%; use paper's if given
    source: "Table 3, row 2, col W4"   # traceable to its location in the paper
    hardware: null                     # null for accuracy claims; pinned for efficiency, e.g. "A100-80G,bs=1,seqlen=2048"
```

### 3.3 Key Design Points

1. **The `runner` field is central**: `official` = call the repo's own script (preferred); `cited-standard` = a standard implementation the paper explicitly cites (e.g. a pinned lm-eval version); `custom` = none of the above, rebuild from the protocol, with the report flagging "unofficial implementation".
2. **The explicit honesty of `calib_status: UNKNOWN`**: the #1 cause of quantization reproduction failure is calib inconsistency. If unextractable, mark UNKNOWN, grade treats it as "incomparable", and **never silently fudges with a default**.
3. **`source` traceability**: each claim is pinned to a paper Table/row/column, so gate 1 review can verify them one by one.

### 3.4 SpecExtract (gate 1)

One headless call, fed the full LaTeX + README, producing the YAML above.
- By default **extract main results only** (the paper's headline/bolded ones), rest on demand.
- If the paper doesn't state a tolerance, use the default (PPL ±0.05, acc ±0.5%) and flag "default value, please confirm".
- **Stop** after extraction; the user reviews `spec.yaml`: check numbers/protocol/tolerance, only proceed after edits.

## 4. Setup / Run / Failure Modes

### 4.1 Setup — the only agentic stage

The only step handed to Claude Code headless, because "taming the rotting official-repo environment" is the only genuinely open-ended problem.

**Goal (single, decidable):** fix the conda/uv environment until **the repo's own eval command runs successfully once** (a smoke test, not the full run). This is a machine-decidable exit condition.

**Smoke-test input:** (a) the repo's own example/test if available; otherwise (b) fall back to one of the spec's claim commands shrunk to a tiny scale (e.g. 8 samples, 1 batch).

**Loop:**
```
build env (conda/uv) → install deps → smoke-run the eval command
   → fail → agent reads traceback → fix (pin version / add package / patch API) → retry
   → success → freeze env snapshot → exit
```

**Guardrails:**
- **Retry cap + total timeout**: on exceeding them, don't silently give up — stop, mark `setup: FAILED`, hand the full setup_log to the user.
- **Env snapshot recorded**: on success, write `pip freeze` + CUDA/torch/transformers versions into `env_snapshot.json`. This is the #1 false-negative source on the official-repo path (dependency drift); the report must record it.
- **Agent change trail**: each patch (which API line changed, which version pinned) is recorded into `setup_patches/`; it may be the very cause of a reproduction failure, so grade and the report must see it.

**Why setup and run are separate:** setup only ensures "the environment can run", never touching real experiment parameters. The agent's nondeterminism is confined to "make it runnable" and doesn't leak into "what numbers come out".

### 4.2 Run — deterministic execution

No agent after setup passes. Execute per spec, item by item:
```
for artifact in spec.artifacts:
    quantize per quant_config (call repo's quant entry or provided checkpoint)
for claim in spec.claims:
    run eval per eval_protocol.command, persist raw stdout/products to runs/<claim_id>/
    record: actual command, seed, start/end time, GPU used
```

**Prefer the official reproduction command**: many repos directly provide a quantized checkpoint or a single `python main.py --reproduce`. run prefers that over assembling parameters itself — highest fidelity, least chance of a parameter misstep. In that case quant_config degrades to a grading basis (grade uses it to check faithfulness), not an execution instruction.

**run does not parse results or grade**, it only faithfully persists raw output.

### 4.3 Quantization-specific failure modes (plant checks early)

| Failure mode | Where the check lives |
|---|---|
| calib inconsistency (split/count/seqlen) | specextract extracts calib; if unextractable mark UNKNOWN → grade "incomparable" |
| PPL convention (stride, seqlen) | eval_protocol must extract seqlen/stride; grade verifies |
| official repo deps rotted | setup debug loop + env snapshot |
| accuracy reproduced but the selling point is speedup | report separately states "which half reproduced"; efficiency claims flagged with hardware |
| agent edited repo, shifting the numbers | setup_patches trail, exposed in report |

## 5. Grade + Report

### 5.1 Grade — pure code, isolated from execution

Reads only the run's persisted raw output + spec; **no re-running, no visibility into execution context beyond "the value to match"**.

Two independent checks per claim, both must pass for MATCH:

**Check 1 · Value in tolerance**
```
parse runs/<claim_id>/ → measured
pass_value = |measured - expected| <= tolerance
unparseable → UNPARSEABLE (don't guess)
```
Parsers are written per metric type (PPL, accuracy, speedup…), extracting from known eval-script output formats.

**Check 2 · Process faithfulness**
```
compare actual config vs spec.eval_protocol / quant_config item by item:
  seqlen, stride, calib, wbits, group_size, few_shot...
pass_faithful = all key items consistent
calib_status==UNKNOWN → incomparable
setup_patches contains numerics-affecting changes → flag downgrade
```

**Three verdicts + BLOCKED:**

| Verdict | Condition |
|---|---|
| **MATCH** | value in tolerance AND process faithful |
| **PARTIAL** | value in tolerance but process diverged; or process faithful but value out of tolerance (reason required) |
| **FAIL** | value significantly off and unattributable |
| **BLOCKED** | setup failed / output unparseable / eval didn't run — "didn't run" is not "failed to reproduce", a separate state |

PARTIAL always carries a reason (which config item differs, by how much). Never "close enough counts". BLOCKED is separate from FAIL: "the environment didn't come up" and "the method didn't reproduce" are two different things.

### 5.2 Report — deterministic rendering, two bilingual files

Each paper produces `report.zh.md` and `report.en.md`; the core is a traceable, replayable table:

```markdown
# Reproduction Report: <title> (<arxiv_id>)
repo: <url>@<commit> | env: torch X / transformers Y / CUDA Z
Verdict summary: MATCH 3 / PARTIAL 1 / FAIL 0 / BLOCKED 1

| claim | model | config | metric | paper | measured | verdict | reason |
|-------|-------|--------|--------|-------|----------|---------|--------|
| c1 | Llama2-7B | W4G128 | wiki2 PPL | 5.78 | 5.80 | MATCH | — |
| c2 | Llama2-7B | W3G128 | wiki2 PPL | 6.92 | 7.41 | PARTIAL | out of tolerance by 0.49; calib n_samples unextractable, used default 128 |
| c3 | ... | | speedup | 2.1x | — | BLOCKED | setup failed: cuda kernel won't compile |

## Replay info (per claim)
c1: command `...` | seed 0 | GPU A100×1 | 18min | raw output runs/c1/stdout.log
## Setup patch trail
- pinned transformers==4.36 (repo requires 4.31 but conflicts with torch 2.x)
## Which half reproduced
accuracy 3/4 reproduced; 1 efficiency claim BLOCKED — paper's main selling point includes speedup, that part unverified
```

**Report iron rules:**
- Always the measured raw number; never fill in paper numbers as a substitute.
- Each claim carries the full replay command + seed + commit + env, so anyone can re-run it.
- "Which half reproduced" stated explicitly — accuracy reproduced ≠ speedup reproduced.
- judge logic separated from execution; verdicts traceable to grade's two checks.

### 5.3 Accumulation (lightweight)

Each reproduction produces reusable assets, landing in a directory rather than a database:
- `env_snapshot.json` + `setup_patches` → reuse directly next time for the same method/repo family.
- method adapters (AWQ/GPTQ families) → accumulate into `adapters/`, reuse the base when a variant shows up.

In the CLI form this is just "a directory layout by convention", not extra infrastructure.

## 6. From-Scratch Path (interface reserved this phase)

Papers without an official repo take the "from-scratch" path. Not implemented this phase, but the architecture reserves for it:
- **Provider interface**: the official-repo path is an `OfficialRepoProvider`, from-scratch is a `FromScratchProvider`; both implement the same interface (produce quantized products + executable eval commands), converging on the same grade/report.
- Papers ingest-marked `repo: null` are simply SKIPped this phase, with the report noting "no official repo, pending from-scratch implementation".

## 7. Run Directory Layout

```
runs/<arxiv_id>-<timestamp>/
  ingest.json
  paper/                 # LaTeX source
  repo/                  # cloned official repo
  spec.yaml
  plan.json
  env/                   # conda/uv env (or reference)
  env_snapshot.json
  setup_log/
  setup_patches/
  runs/<claim_id>/       # per-claim raw output, command, seed
  grades.json
  report.zh.md
  report.en.md
```

## 8. What We Don't Do (YAGNI)

- No job queue / DB / cron (manual CLI, file state is enough).
- No cross-paper scheduler (multi-GPU only serves claims within a single run).
- No cost-budget approval gate (resources not a constraint).
- No LLM grading (numeric comparison done in code).
- No from-scratch provider this phase (interface only).
