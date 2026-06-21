# Interactive Claim Selection at Gate 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** At gate 1, instead of stopping and telling the user to edit spec.yaml manually, display a numbered table of all extracted claims and let the user type which numbers to reproduce — keeping the selected subset and pruning orphaned artifacts from the spec in-place before continuing.

**Architecture:** Add `spec_selection_prompt(spec, run_dir) -> bool` to `cli.py`. It prints a claim table, reads a text selection (numbers / "all" / "q"), mutates `spec.claims` and `spec.artifacts` in-place to keep only the chosen claims and their referenced artifacts, and returns True to continue or False to abort. The pipeline's `approve_spec` callback in the `run` command calls this function. `--yes` bypasses it entirely (keep all claims). The pipeline contract (`approve_spec` returns bool, `rd.write_spec(spec)` is called after on True) is untouched — no changes to `pipeline.py`.

**Tech Stack:** Python 3.11, click (already in use for all CLI interaction), pytest + click.testing.CliRunner for offline tests. `from __future__ import annotations`.

## Global Constraints

- Do not change `pipeline.py` — the `approve_spec(spec) -> bool` contract is the seam.
- `--yes` flag must keep all claims unchanged (existing CI / non-interactive behaviour).
- Selecting 0 claims must abort with a clear message — an empty spec causes downstream errors.
- `resume` command is unaffected — it reads the spec already on disk, no selection needed.
- All tests must stay offline (no GPU, no real `claude` subprocess, no network).
- Run: `uv run pytest -q` — must stay green. Run: `uv run ruff check src/ tests/` — must stay clean.
- Baseline: 169 tests passing on `main`.

---

## File Structure

```
src/paper_reprise/
  cli.py          # MODIFY — add spec_selection_prompt(); update approve_spec in run cmd

tests/
  test_cli.py     # MODIFY — add tests for selection logic and wired integration
```

No new files. No changes to `pipeline.py`, `models.py`, or any other module.

---

## Task 1: `spec_selection_prompt` — pure helper + tests (TDD)

**Files:**
- Modify: `src/paper_reprise/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `spec_selection_prompt(spec: Spec, run_dir: Path) -> bool`
  - Mutates `spec.claims` and `spec.artifacts` in-place.
  - Returns `True` if ≥1 claim selected and user confirmed; `False` to abort.
- Consumes: `Spec` (from `paper_reprise.models`), `Path` (stdlib), `click.echo` / `click.prompt`.

**What the function prints (example with 3 claims):**

```
Extracted 3 claims — pick which to reproduce:

  #  claim-id                    model                   config    metric      expected  hardware
  1  llama3-8b-2bit-avg-acc      Llama-3.1-8B-Instruct   W2G128    avg_acc     68.55     1x A100
  2  llama3-8b-3bit-avg-acc      Llama-3.1-8B-Instruct   W3G128    avg_acc     72.32     1x A100
  3  llama3-70b-2bit-avg-acc     Llama-3.1-70B-Instruct  W2G128    avg_acc     75.57     4x H100

Enter numbers (e.g. "1 3"), "all", or "q" to abort [all]:
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
# ── spec_selection_prompt ────────────────────────────────────────────────────

def _make_spec(n_claims=3):
    """Build a Spec with n_claims each referencing a distinct artifact."""
    from paper_reprise.models import Spec, Artifact, Claim, EvalProtocol
    artifacts = [
        Artifact(id=f"art-{i}", base_model=f"org/Model-{i}B",
                 method="GSQ", quant_config={"wbits": 2, "group_size": 128})
        for i in range(n_claims)
    ]
    claims = [
        Claim(id=f"c{i}", artifact=f"art-{i}",
              eval_protocol=EvalProtocol(runner="official", command=f"eval {i}",
                                         metric="avg_acc", dataset="arc"),
              expected=70.0 + i, tolerance=0.5, source=f"Table 1 row {i}",
              hardware="1x A100")
        for i in range(n_claims)
    ]
    return Spec(paper="2401.00001", repo=None, artifacts=artifacts, claims=claims)


def test_selection_all_keeps_everything(tmp_path):
    from paper_reprise.cli import spec_selection_prompt
    spec = _make_spec(3)
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,  # we call via the helper directly in isolation
            catch_exceptions=False,
            input="all\n",
        )
    # Call directly (not via CLI invoke) because spec_selection_prompt is a
    # standalone function; we test it in isolation.
    spec2 = _make_spec(3)
    from unittest.mock import patch
    with patch("paper_reprise.cli.click.prompt", return_value="all"):
        from paper_reprise.cli import spec_selection_prompt
        kept = spec_selection_prompt(spec2, tmp_path)
    assert kept is True
    assert len(spec2.claims) == 3
    assert len(spec2.artifacts) == 3


def test_selection_subset_keeps_chosen_and_prunes_orphans(tmp_path):
    from paper_reprise.cli import spec_selection_prompt
    from unittest.mock import patch
    spec = _make_spec(3)
    with patch("paper_reprise.cli.click.prompt", return_value="1 3"):
        kept = spec_selection_prompt(spec, tmp_path)
    assert kept is True
    assert [c.id for c in spec.claims] == ["c0", "c2"]
    # artifact for c1 (art-1) must be pruned — no claim references it anymore
    remaining_artifact_ids = {a.id for a in spec.artifacts}
    assert "art-0" in remaining_artifact_ids
    assert "art-2" in remaining_artifact_ids
    assert "art-1" not in remaining_artifact_ids


def test_selection_zero_claims_aborts(tmp_path):
    from paper_reprise.cli import spec_selection_prompt
    from unittest.mock import patch
    spec = _make_spec(2)
    # "5" is out of range — no valid claim selected
    with patch("paper_reprise.cli.click.prompt", return_value="5"):
        kept = spec_selection_prompt(spec, tmp_path)
    assert kept is False
    # spec is unchanged (abort before mutation)
    assert len(spec.claims) == 2


def test_selection_q_aborts(tmp_path):
    from paper_reprise.cli import spec_selection_prompt
    from unittest.mock import patch
    spec = _make_spec(2)
    with patch("paper_reprise.cli.click.prompt", return_value="q"):
        kept = spec_selection_prompt(spec, tmp_path)
    assert kept is False
    assert len(spec.claims) == 2  # unchanged


def test_selection_single_claim(tmp_path):
    from paper_reprise.cli import spec_selection_prompt
    from unittest.mock import patch
    spec = _make_spec(3)
    with patch("paper_reprise.cli.click.prompt", return_value="2"):
        kept = spec_selection_prompt(spec, tmp_path)
    assert kept is True
    assert len(spec.claims) == 1
    assert spec.claims[0].id == "c1"   # claim index 2 → c1 (1-based)
    assert len(spec.artifacts) == 1
    assert spec.artifacts[0].id == "art-1"
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
uv run pytest tests/test_cli.py -k "selection" -v
```

Expected: `ImportError: cannot import name 'spec_selection_prompt' from 'paper_reprise.cli'`

- [ ] **Step 3: Implement `spec_selection_prompt` in `cli.py`**

Add after the `_run_executor()` function and before `_timestamp()`:

```python
def _claim_row(i: int, claim, artifacts: dict) -> str:
    """One display row for the claim table."""
    art = artifacts.get(claim.artifact)
    model = art.base_model.split("/")[-1] if art else "?"
    wbits = art.quant_config.get("wbits", "?") if art else "?"
    group = art.quant_config.get("group_size", "") if art else ""
    config = f"W{wbits}G{group}" if group else f"W{wbits}"
    hw = claim.hardware or "—"
    return (f"  {i:>2}  {claim.id:<30} {model:<24} {config:<9} "
            f"{claim.eval_protocol.metric:<11} {claim.expected:<9.2f} {hw}")


def spec_selection_prompt(spec: "Spec", run_dir: "Path") -> bool:
    """Print the extracted claims as a numbered table and ask the user to select
    which to reproduce. Mutates spec.claims and spec.artifacts in-place to keep
    only the chosen subset and prune orphaned artifacts. Returns False to abort."""
    from paper_reprise.models import Spec  # local to avoid top-level circular risk
    claims = spec.claims
    artifacts = {a.id: a for a in spec.artifacts}

    header = (f"  {'#':>2}  {'claim-id':<30} {'model':<24} {'config':<9} "
              f"{'metric':<11} {'expected':<9} {'hardware'}")
    click.echo(f"\nExtracted {len(claims)} claims — pick which to reproduce:\n")
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))
    for i, c in enumerate(claims, 1):
        click.echo(_claim_row(i, c, artifacts))

    raw = click.prompt(
        '\nEnter numbers (e.g. "1 3"), "all", or "q" to abort',
        default="all",
    )
    raw = raw.strip().lower()
    if raw == "q":
        click.echo("Aborted.")
        return False
    if raw == "all":
        indices = set(range(len(claims)))
    else:
        chosen = []
        for tok in raw.split():
            if tok.isdigit():
                n = int(tok)
                if 1 <= n <= len(claims):
                    chosen.append(n - 1)
        indices = set(chosen)

    if not indices:
        click.echo("No valid claims selected — aborting.")
        return False

    selected_claims = [claims[i] for i in sorted(indices)]
    referenced_artifacts = {c.artifact for c in selected_claims}
    selected_artifacts = [a for a in spec.artifacts if a.id in referenced_artifacts]

    spec.claims = selected_claims
    spec.artifacts = selected_artifacts
    click.echo(f"Selected {len(selected_claims)} claim(s). Continuing…")
    return True
```

Also add the `Spec` and `Path` imports at the top of the function docstring; the function uses string annotations so no import needed at module level. However `Path` IS already imported at module level (`from pathlib import Path` — check cli.py). `Spec` is only used in the type hint string so no import needed.

- [ ] **Step 4: Run the selection tests**

```bash
uv run pytest tests/test_cli.py -k "selection" -v
```

Expected: all 5 selection tests PASS.

- [ ] **Step 5: Run the full suite**

```bash
uv run pytest -q
```

Expected: 174 passed (169 baseline + 5 new). No failures.

- [ ] **Step 6: Commit**

```bash
git add src/paper_reprise/cli.py tests/test_cli.py
git commit -m "feat(cli): spec_selection_prompt — interactive claim picker at gate 1"
```

---

## Task 2: Wire `spec_selection_prompt` into the `run` command

**Files:**
- Modify: `src/paper_reprise/cli.py` — update `approve_spec` in `run`; update the post-run output messages
- Modify: `tests/test_cli.py` — add one integration test

**Interfaces:**
- Consumes: `spec_selection_prompt(spec, run_dir) -> bool` (from Task 1)

**What changes in `run`:**

Currently:
```python
def approve_spec(spec):
    return yes
```
After:
```python
def approve_spec(spec):
    if yes:
        return True
    return spec_selection_prompt(spec, result_root)
```

But `result_root` isn't known yet when `approve_spec` is defined (it's returned by `run_pipeline`). The solution: capture via closure over `rd`'s root, which we can reconstruct from `result.root` — but `approve_spec` is called INSIDE `run_pipeline` before it returns. 

**Correct approach:** Use a mutable container (`nonlocal` or a list) to capture the run dir path once `run_pipeline` starts. But actually simpler: `approve_spec` already receives `spec` which doesn't have the run dir. The run dir can be passed as a closure over a mutable variable that the pipeline writes into. But `run_pipeline` doesn't expose it mid-run.

**Even simpler:** `spec_selection_prompt` needs `run_dir` only to label the output (the run dir name). We can pass `Path(base_dir)` or omit the path from the display entirely. Looking at `spec_selection_prompt` — it only uses `run_dir` in the label: we can replace that with the arxiv_id label from the closure. 

Change `spec_selection_prompt(spec, run_dir: Path) -> bool` to `spec_selection_prompt(spec, label: str) -> bool` where label is a display string. The `run` command passes `arxiv_id` or `paper_name`:

```python
display_label = paper_name or arxiv_id

def approve_spec(spec):
    if yes:
        return True
    return spec_selection_prompt(spec, display_label)
```

This is cleaner and avoids any closure problem. Update Task 1's implementation and tests accordingly (the parameter is now `label: str` not `run_dir: Path`).

**Updated signature:** `spec_selection_prompt(spec: Spec, label: str) -> bool`

The header line becomes: `click.echo(f"\nExtracted {len(claims)} claims from {label} — pick which to reproduce:\n")`

**Post-selection output**: currently the `run` command has a special block:
```python
if result.aborted_at == "spec-approval":
    spec = RunDir.open(result.root).read_spec()
    n = len(spec.claims) if spec else "?"
    click.echo(f"\nExtracted {n} claims into {result.root}/spec.yaml")
    click.echo("Review/edit which models & claims to reproduce, then continue with:")
    click.echo(f"  paper-reprise resume {result.root}")
```

With interactive selection, the user has already picked claims interactively, so the pipeline continues rather than aborting at `spec-approval` — this block is only reached if the user quits (`q` or 0 claims). Update its message to reflect that:
```python
if result.aborted_at == "spec-approval":
    click.echo(f"\nRun aborted at spec selection. Run dir: {result.root}")
    click.echo("To retry: paper-reprise resume " + str(result.root))
```

- [ ] **Step 1: Update `spec_selection_prompt` signature from `run_dir: Path` to `label: str`**

In `cli.py`, change the function signature and the one internal use:

Old:
```python
def spec_selection_prompt(spec: "Spec", run_dir: "Path") -> bool:
    ...
    click.echo(f"\nExtracted {len(claims)} claims — pick which to reproduce:\n")
```

New:
```python
def spec_selection_prompt(spec: "Spec", label: str) -> bool:
    ...
    click.echo(f"\nExtracted {len(claims)} claims from {label} — pick which to reproduce:\n")
```

- [ ] **Step 2: Update Task 1's tests to pass `label` string instead of `tmp_path`**

In `tests/test_cli.py`, replace every `spec_selection_prompt(spec, tmp_path)` with `spec_selection_prompt(spec, "test-paper")`.

The 5 selection tests still exercise the same logic — only the display label changes.

- [ ] **Step 3: Confirm Task 1 tests still pass after signature change**

```bash
uv run pytest tests/test_cli.py -k "selection" -v
```

Expected: 5 PASS.

- [ ] **Step 4: Wire into the `run` command**

In `cli.py`, replace:
```python
    def approve_spec(spec):
        return yes
```

With:
```python
    display_label = paper_name or arxiv_id

    def approve_spec(spec):
        if yes:
            return True
        return spec_selection_prompt(spec, display_label)
```

And replace the `spec-approval` output block:
```python
    if result.aborted_at == "spec-approval":
        spec = RunDir.open(result.root).read_spec()
        n = len(spec.claims) if spec else "?"
        click.echo(f"\nExtracted {n} claims into {result.root}/spec.yaml")
        click.echo("Review/edit which models & claims to reproduce, then continue with:")
        click.echo(f"  paper-reprise resume {result.root}")
```

With:
```python
    if result.aborted_at == "spec-approval":
        click.echo(f"\nAborted at claim selection. Run dir: {result.root}")
        click.echo(f"To retry: paper-reprise resume {result.root}")
```

- [ ] **Step 5: Write the integration test**

Append to `tests/test_cli.py`:

```python
def test_cli_run_interactive_selection_filters_spec_and_continues(tmp_path, monkeypatch):
    """Gate 1 now asks the user to select claims; selected subset gets written to
    spec.yaml and the pipeline continues (no spec-approval abort)."""
    import paper_reprise.pipeline as pipeline_mod
    from paper_reprise.models import Spec, Artifact, Claim, EvalProtocol

    captured_spec = {}

    def fake_extract_spec(rd):
        spec = Spec(
            paper="2401.00001", repo=None,
            artifacts=[
                Artifact(id="a1", base_model="org/M1", method="GSQ",
                         quant_config={"wbits": 2}),
                Artifact(id="a2", base_model="org/M2", method="GSQ",
                         quant_config={"wbits": 3}),
            ],
            claims=[
                Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="official", command="eval1",
                                                  metric="ppl", dataset="wiki"),
                      expected=5.8, tolerance=0.05, source="T1"),
                Claim(id="c2", artifact="a2",
                      eval_protocol=EvalProtocol(runner="official", command="eval2",
                                                  metric="ppl", dataset="wiki"),
                      expected=6.1, tolerance=0.05, source="T2"),
            ],
        )
        return spec

    def fake_finish_pipeline(rd, spec, ingest, **kwargs):
        captured_spec["claims"] = [c.id for c in spec.claims]
        captured_spec["artifacts"] = [a.id for a in spec.artifacts]
        return pipeline_mod.PipelineResult(root=rd.root, aborted_at=None)

    monkeypatch.setattr(pipeline_mod, "extract_spec", fake_extract_spec)
    monkeypatch.setattr(pipeline_mod, "_finish_pipeline", fake_finish_pipeline)
    monkeypatch.setattr(cli_mod, "make_fetch_sources",
                        lambda **k: (lambda rd, arxiv_id, url: None))

    from unittest.mock import patch
    with patch("paper_reprise.cli.click.prompt", return_value="1"):
        res = CliRunner().invoke(
            cli_mod.cli,
            ["run", "2401.00001", "--base-dir", str(tmp_path)],
        )

    assert res.exit_code == 0
    assert "spec-approval" not in res.output   # did NOT abort
    # Only the first claim (and its artifact) made it through
    assert captured_spec["claims"] == ["c1"]
    assert captured_spec["artifacts"] == ["a1"]
    # Second artifact was pruned (no claim references it)
    assert "a2" not in captured_spec["artifacts"]
```

- [ ] **Step 6: Run the new integration test**

```bash
uv run pytest tests/test_cli.py::test_cli_run_interactive_selection_filters_spec_and_continues -v
```

Expected: PASS.

- [ ] **Step 7: Run the full test suite + ruff**

```bash
uv run pytest -q && uv run ruff check src/ tests/
```

Expected: 175 passed. "All checks passed."

- [ ] **Step 8: Commit**

```bash
git add src/paper_reprise/cli.py tests/test_cli.py
git commit -m "feat(cli): wire interactive claim selection into gate 1 of run command"
```

---

## Self-Review

**1. Spec coverage:**
- ✅ Gate 1 presents a numbered claim table — `_claim_row` builds each row, the loop prints them
- ✅ User types numbers / "all" / "q" — all three branches handled in `spec_selection_prompt`
- ✅ `--yes` bypasses selection entirely — `if yes: return True` before calling `spec_selection_prompt`
- ✅ 0 valid claims selected → abort with message, spec unchanged
- ✅ Orphaned artifacts pruned — `referenced_artifacts = {c.artifact for c in selected_claims}`
- ✅ `resume` unaffected — no changes to `resume_pipeline` or `resume` command
- ✅ Pipeline contract unchanged — no changes to `pipeline.py`

**2. Placeholder scan:** No TBD/TODO in the plan. All code blocks are complete.

**3. Type consistency:**
- `spec_selection_prompt(spec: "Spec", label: str) -> bool` — used identically in Task 1 tests and Task 2 wiring
- `_claim_row(i: int, claim, artifacts: dict) -> str` — used only inside `spec_selection_prompt`
- `spec.claims = selected_claims` / `spec.artifacts = selected_artifacts` — Pydantic v2 allows field assignment (no `frozen=True` in the model config)
