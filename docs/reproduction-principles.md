# Reproduction Principles

Judgment principles for reproducing a quantization paper — especially a **from-scratch**
reproduction (no official repo, where you write `impl/` yourself). Companion to the
[algorithm-overview diagram guide](algorithm-diagram-guide.md).

## 1. The paper is the source of truth — not an upstream repo

A paper often builds on a prior method that has its own repo (e.g. TurboQuant builds on
QJL). When the paper restates that method with an explicit definition, implement **that**,
even if the upstream repo uses a different convention. Reference repos are a read-only aid
to disambiguate underspecified details; the paper wins where they differ. (The spec's
`references` field surfaces such repos to the from-scratch scaffold as read-only context.)

## 2. Cross-check the algorithm against an independent closed form

Don't only check the measured metric against the paper's reported number — that's circular
(a wrong implementation can coincidentally match). Find a textbook / closed-form result
the algorithm must **also** satisfy, and assert it in a unit test.

Example: for TurboQuant the MSE distortion is the classic Lloyd–Max Gaussian-quantizer MSE
— the product quantizer must be unbiased while `mse@b=1` carries the `2/π` bias; a wrong
impl cannot silently "match" both the paper number and the closed form.

**How to apply:** in `impl/test_*.py`, validate against the independent closed form first,
the paper's number second.

## 3. Pick the cheap core-validation section, and confirm scope

`specextract` tends to grab the heaviest headline claims. For "reproduce the algorithm",
the faithful *and* cheap target is usually the paper's core empirical-validation section
(e.g. TurboQuant §4.1 distortion rates, not the full KV-cache LongBench suite). Surface the
scope — model × config, rough cost — to the user rather than defaulting to the headline
matrix.

## 4. Surface scale before a large run — even under `--yes`

`--yes` / "default is fine, no need to ask" authorizes skipping the *claim-selection* gate.
It does **not** mean silently launch an arbitrarily large job. Before kicking off a large
reproduction (multiple models, any large model like 70B, many configs/tasks), first print
the scale — N claims, biggest model, task count, rough cost — even under `--yes`. Report
it; give the user a beat to stop or narrow. When unsure, default to a sensible **subset**
(smallest model / one representative config) rather than the full matrix.

## 5. Diagnose the eval infrastructure before blaming the quantization

When measured accuracy / PPL doesn't match, check whether the **FP baseline itself** is
off first. If the BF16 baseline is already off tolerance, the quantization implementation
is not the cause — the gap is in the eval setup.

Example (MXFP4 2603.08713): PPL matched (±0.06) but `acc_norm` was consistently ~1.5–2.1
low, and the BF16 baseline was itself off by 1.55. Root cause: passing a pre-instantiated
HF model object to lm-eval instead of a string path (HFLM vs the paper's vLLM engine).
Teacher-forcing PPL is insensitive to the engine; log-likelihood ranking (`acc_norm`) is
sensitive.

## 6. Write the gap analysis into `analysis_en.md` / `analysis_zh.md`

Put the English gap analysis in `analysis_en.md` and the Chinese in `analysis_zh.md` in the
run dir — one language per file, **not** a single bilingual `analysis.md`. `paper-reprise
report` reads them and embeds each into `README.md` / `README_zh.md`. Keep it concise and
non-redundant with the auto-generated Conclusion. Suggested structure: root cause (a table
comparing paper vs reproduction setup) → evidence → why other metrics are unaffected → fix
direction.

## Appendix: driving a from-scratch reproduction by hand

When implementing a no-repo paper manually (instead of the headless from-scratch scaffold):

1. Run **ingest + specextract only** (the CLI's first two stages) so it stops before the
   Gate-1 selection and before setup — do not use `--yes`.
2. **Rewrite the run dir `spec.yaml`** to the claims you actually reproduce (confirm scope
   with the user first — see §3/§4).
3. **Hand-write `impl/`** — the conventional entrypoint is `impl/run_eval.sh <claim_id>`,
   which prints a `metric: <number>` line — plus algorithm unit tests checked against the
   paper's closed-form values (see §2).
4. **Build `env/` yourself** (`uv venv` + deps). Do not run the from-scratch setup loop:
   its scaffold step requires a turn that *modifies* `impl/` and fails on a pre-built one.
5. **Drive run → grade → report** with a small driver that calls the from-scratch run
   executor per claim, then `paper-reprise report <run_dir>`. Open the RunDir with an
   **absolute** path — the eval env puts `env/bin` on PATH relatively and `run_eval.sh`
   `cd`s into `impl/`, so a relative run dir breaks `python` resolution.
