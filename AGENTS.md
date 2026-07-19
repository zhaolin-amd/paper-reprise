# AGENTS.md — paper-reprise

Repo-level guidance for coding agents (Claude Code, etc.) operating in this repo.

## Always run `verify-fromscratch-fidelity` — don't wait to be asked

**Proactively invoke the `verify-fromscratch-fidelity` skill**
(`.claude/skills/verify-fromscratch-fidelity/SKILL.md`) — do not treat it as
something the user has to explicitly request. Its two parts have different scope:

- **Part A** (paper-internal fidelity, no official repo to diff against) — run on
  any run dir with an `impl/` (the from-scratch path):
  - **before you report a MATCH as trustworthy** — a MATCH from `grade` only means
    the final number is in tolerance, not that the mechanism is right (see the
    skill's Gotchas);
  - investigating **any PARTIAL/FAIL** claim, before proposing a fix.
- **Part B** (kernel-composition correctness) — run on **any run, official-repo or
  from-scratch**, whenever the eval path wires in an imported or modified kernel
  (Quark MXFP4/MXFP8, CUTLASS, or any vendor kernel/library used instead of the
  paper's own unmodified code computing the numeric primitive). This explicitly
  includes an **official-repo run where the setup loop patched or swapped a kernel**
  to get the repo's own eval command running — check `setup_patches/patch_*.txt` for
  anything touching kernel/CUDA/Triton code, a custom op, or a vendored quantization
  library, and run Part B against it even though `impl/` doesn't exist there. Run
  Part B even if nobody flagged the kernel coupling as suspicious: this class of bug
  (correct algorithm logic, miscoupled to the kernel that actually executes it) does
  not show up as a crash, and end2end numbers alone can't localize it — nor can an
  otherwise-clean setup-patch trail, since a patch that swaps in a working-but-
  different kernel still reads like a normal dependency fix.

Concretely: after `setup`/`run` succeeds on ANY run, check `impl/` (from-scratch) or
`setup_patches/patch_*.txt` (official-repo) for calls into or changes touching an
external kernel/library before considering the run done. If found, confirm a test
actually exercises that coupling (contract check, boundary diff against a pure
reference, round-trip, version pin in `env_snapshot.json`) — if it doesn't, run the
missing checks yourself and add the tests (in `impl/test_*.py`, or inside `repo/`
following that repo's own test conventions), the same way you would for any other
gap the skill's checklist surfaces. Don't wait for the user to say "verify this" or
name the kernel explicitly.

This is a standing instruction, not a one-off: it applies to every run in this repo,
not just the one it was first written for.

## Pull before you push

**Always `git pull` (fast-forward, or `--rebase` if you have local commits) before
`git push`** — never push straight from a local checkout that hasn't just synced with
`origin/main`. Multiple sessions/agents work in this repo; pushing on a stale base risks a
rejected push at best, or a diverged/overwritten remote history at worst.

- **Why:** caught in practice — local was 24 commits behind `origin/main` at the start of a
  session that then went on to commit and almost pushed on that stale base.
- **How to apply:** before any `git push`, run `git fetch` + check `git status` /
  `git log HEAD..origin/main`; if behind, pull first (fast-forward when there are no local
  commits yet, `--rebase` when there are). This applies to every push, not just the first
  one in a session — a session can fall behind mid-way if another agent/human pushes while
  you're working.
